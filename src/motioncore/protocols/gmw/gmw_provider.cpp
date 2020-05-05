// MIT License
//
// Copyright (c) 2020 Lennart Braun
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "gmw_provider.h"
#include <unordered_map>

#include "base/gate_register.h"
#include "communication/communication_layer.h"
#include "communication/fbs_headers/gmw_message_generated.h"
#include "communication/message.h"
#include "communication/message_handler.h"
#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/mt_provider.h"
#include "utility/constants.h"
#include "utility/logger.h"
#include "gate.h"
#include "wire.h"

namespace MOTION::proto::gmw {

struct GMWMessageHandler : public Communication::MessageHandler {
  GMWMessageHandler(std::size_t num_parties, std::shared_ptr<Logger> logger);
  void received_message(std::size_t, std::vector<std::uint8_t> &&raw_message) override;

  enum class MsgType { bit, uint8, uint16, uint32, uint64 };

  std::unordered_map<std::size_t, std::pair<

  std::vector<
  std::unordered_map<std::size_t,
                     std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>>>>
      bits_promises_;
  std::shared_ptr<Logger> logger_;
};

GMWMessageHandler::GMWMessageHandler(std::size_t num_parties, std::shared_ptr<Logger> logger)
    : bits_promises_(num_parties), logger_(logger) {}

void GMWMessageHandler::received_message(std::size_t party_id,
                                         std::vector<std::uint8_t> &&raw_message) {
  assert(!raw_message.empty());
  flatbuffers::Verifier verifier(raw_message.data(), raw_message.size());
  auto message = Communication::GetMessage(raw_message.data());
  if (!message->Verify(verifier)) {
    throw std::runtime_error("received malformed Message");
    // TODO: log and drop instead
  }
  auto message_type = message->message_type();
  switch (message_type) {
    case Communication::MessageType::GMWGate: {
      flatbuffers::Verifier verifier(message->payload()->data(), message->payload()->size());
      auto gate_message =
          flatbuffers::GetRoot<MOTION::Communication::GMWGateMessage>(message->payload()->data());
      if (!gate_message->Verify(verifier)) {
        throw std::runtime_error("received malformed GMWGateMessage");
        // TODO: log and drop instead
      }
      auto gate_id = gate_message->gate_id();
      auto payload = gate_message->payload();
      auto& promise_map = bits_promises_[party_id];
      if (auto it = promise_map.find(gate_id); it != promise_map.end()) {
        auto &[num_bits, promise] = it->second;
        auto num_bytes = (num_bits + 7) / 8;
        if (payload->size() == num_bytes) {
          promise.set_value(ENCRYPTO::BitVector(payload->data(), num_bits));
        } else if (logger_) {
          logger_->LogError(fmt::format(
              "received GMWGateMessage for gate {} of size {} while expecting size {}, dropping",
              gate_id, payload->size(), num_bytes));
        }
      } else {
        logger_->LogError(
            fmt::format("received unexpected GMWGateMessage for gate {}, dropping", gate_id));
      }
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
}

GMWProvider::GMWProvider(Communication::CommunicationLayer& communication_layer,
                         GateRegister& gate_register,
                         Crypto::MotionBaseProvider& motion_base_provider, MTProvider& mt_provider,
                         std::shared_ptr<Logger> logger)
    : communication_layer_(communication_layer),
      gate_register_(gate_register),
      motion_base_provider_(motion_base_provider),
      mt_provider_(mt_provider),
      message_handler_(std::make_unique<GMWMessageHandler>(communication_layer_.get_num_parties(), logger)),
      my_id_(communication_layer_.get_my_id()),
      num_parties_(communication_layer_.get_num_parties()),
      next_input_id_(0),
      logger_(std::move(logger)) {
  // TODO
  communication_layer_.register_message_handler([this](auto) { return message_handler_; },
                                                {Communication::MessageType::GMWGate});
}

GMWProvider::~GMWProvider() = default;

void GMWProvider::setup() {
  motion_base_provider_.wait_for_setup();
  mt_provider_.WaitFinished();
  set_setup_ready();
}

bool GMWProvider::is_my_job(std::size_t gate_id) const noexcept {
  return my_id_ == (gate_id % num_parties_);
}

std::size_t GMWProvider::get_next_input_id(std::size_t num_inputs) noexcept {
  auto next_id = next_input_id_;
  next_input_id_ += num_inputs;
  return next_id;
}

static BooleanGMWWireVector cast_wires(std::vector<std::shared_ptr<NewWire>> wires) {
  BooleanGMWWireVector result(wires.size());
  std::transform(std::begin(wires), std::end(wires), std::begin(result),
                 [](auto& w) { return std::dynamic_pointer_cast<BooleanGMWWire>(w); });
  return result;
}

static std::vector<std::shared_ptr<NewWire>> cast_wires(BooleanGMWWireVector&& wires) {
  return std::vector<std::shared_ptr<NewWire>>(std::begin(wires), std::end(wires));
}

template <typename T>
static ArithmeticGMWWireP<T> cast_arith_wire(std::shared_ptr<NewWire> wire) {
  return std::dynamic_pointer_cast<ArithmeticGMWWire<T>>(wire);
}

template <typename T>
static std::shared_ptr<NewWire> cast_arith_wire(ArithmeticGMWWireP<T>&& wire) {
  return std::shared_ptr<NewWire>(wire);
}

// Boolean inputs/outputs

std::pair<ENCRYPTO::ReusableFiberPromise<BitValues>, WireVector>
GMWProvider::make_boolean_input_gate_my(std::size_t input_owner, std::size_t num_wires,
                                        std::size_t num_simd) {
  if (input_owner != my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  BooleanGMWWireVector output;
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> promise;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanGMWInputGateSender>(gate_id, *this, num_wires, num_simd,
                                                          promise.get_future());
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return {std::move(promise), cast_wires(std::move(output))};
}

WireVector GMWProvider::make_boolean_input_gate_other(std::size_t input_owner,
                                                      std::size_t num_wires, std::size_t num_simd) {
  if (input_owner == my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  BooleanGMWWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanGMWInputGateReceiver>(gate_id, *this, num_wires, num_simd,
                                                            input_owner);
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return cast_wires(std::move(output));
}

ENCRYPTO::ReusableFiberFuture<BitValues> GMWProvider::make_boolean_output_gate_my(
    std::size_t output_owner, const WireVector& in) {
  if (output_owner != ALL_PARTIES && output_owner != my_id_) {
    throw std::logic_error("trying to create output gate for wrong party");
  }
  auto gate_id = gate_register_.get_next_gate_id();
  auto input = cast_wires(in);
  auto gate =
      std::make_unique<BooleanGMWOutputGate>(gate_id, *this, std::move(input), output_owner);
  auto future = gate->get_output_future();
  gate_register_.register_gate(std::move(gate));
  return future;
}

void GMWProvider::make_boolean_output_gate_other(std::size_t output_owner, const WireVector& in) {
  if (output_owner == ALL_PARTIES || output_owner == my_id_) {
    throw std::logic_error("trying to create output gate for wrong party");
  }
  auto gate_id = gate_register_.get_next_gate_id();
  auto input = cast_wires(in);
  auto gate =
      std::make_unique<BooleanGMWOutputGate>(gate_id, *this, std::move(input), output_owner);
  gate_register_.register_gate(std::move(gate));
}

// arithmetic inputs/outputs

template <typename T>
std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<T>>, WireVector>
GMWProvider::basic_make_arithmetic_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  if (input_owner != my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  ENCRYPTO::ReusableFiberPromise<std::vector<T>> promise;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<ArithmeticGMWInputGateSender<T>>(gate_id, *this, num_simd,
                                                                promise.get_future());
  auto output = gate->get_output_wire();
  gate_register_.register_gate(std::move(gate));
  return {std::move(promise), {cast_arith_wire(std::move(output))}};
}

std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint8_t>>, WireVector>
GMWProvider::make_arithmetic_8_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_my<std::uint8_t>(input_owner, num_simd);
}
std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint16_t>>, WireVector>
GMWProvider::make_arithmetic_16_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_my<std::uint16_t>(input_owner, num_simd);
}
std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint32_t>>, WireVector>
GMWProvider::make_arithmetic_32_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_my<std::uint32_t>(input_owner, num_simd);
}
std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint64_t>>, WireVector>
GMWProvider::make_arithmetic_64_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_my<std::uint64_t>(input_owner, num_simd);
}

template <typename T>
WireVector GMWProvider::basic_make_arithmetic_input_gate_other(std::size_t input_owner,
                                                               std::size_t num_simd) {
  if (input_owner == my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate =
      std::make_unique<ArithmeticGMWInputGateReceiver<T>>(gate_id, *this, num_simd, input_owner);
  auto output = gate->get_output_wire();
  gate_register_.register_gate(std::move(gate));
  return {cast_arith_wire(std::move(output))};
}

WireVector GMWProvider::make_arithmetic_8_input_gate_other(std::size_t input_owner,
                                              std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_other<std::uint8_t>(input_owner, num_simd);
}
WireVector GMWProvider::make_arithmetic_16_input_gate_other(std::size_t input_owner,
                                               std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_other<std::uint16_t>(input_owner, num_simd);
}
WireVector GMWProvider::make_arithmetic_32_input_gate_other(std::size_t input_owner,
                                               std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_other<std::uint32_t>(input_owner, num_simd);
}
WireVector GMWProvider::make_arithmetic_64_input_gate_other(std::size_t input_owner,
                                               std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_other<std::uint64_t>(input_owner, num_simd);
}

template <typename T>
ENCRYPTO::ReusableFiberFuture<IntegerValues<T>>
GMWProvider::basic_make_arithmetic_output_gate_my(std::size_t output_owner, const WireVector& in) {
  if (output_owner != ALL_PARTIES && output_owner != my_id_) {
    throw std::logic_error("trying to create output gate for wrong party");
  }
  if (in.size() != 1) {
    throw std::logic_error("invalid number of wires for arithmetic gate");
  }
  auto input = cast_arith_wire<T>(in[0]);
  if (input == nullptr) {
    throw std::logic_error("wrong wire type");
  }
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate =
      std::make_unique<ArithmeticGMWOutputGate<T>>(gate_id, *this, std::move(input), output_owner);
  auto future = gate->get_output_future();
  gate_register_.register_gate(std::move(gate));
  return future;
}

ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint8_t>>
GMWProvider::make_arithmetic_8_output_gate_my(std::size_t output_owner, const WireVector& in) {
  return basic_make_arithmetic_output_gate_my<std::uint8_t>(output_owner, in);
}
ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint16_t>>
GMWProvider::make_arithmetic_16_output_gate_my(std::size_t output_owner, const WireVector& in) {
  return basic_make_arithmetic_output_gate_my<std::uint16_t>(output_owner, in);
}
ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint32_t>>
GMWProvider::make_arithmetic_32_output_gate_my(std::size_t output_owner, const WireVector& in) {
  return basic_make_arithmetic_output_gate_my<std::uint32_t>(output_owner, in);
}
ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint64_t>>
GMWProvider::make_arithmetic_64_output_gate_my(std::size_t output_owner, const WireVector& in) {
  return basic_make_arithmetic_output_gate_my<std::uint64_t>(output_owner, in);
}

void GMWProvider::make_arithmetic_output_gate_other(std::size_t output_owner, const WireVector& in) {
  if (output_owner == ALL_PARTIES || output_owner == my_id_) {
    throw std::logic_error("trying to create output gate for wrong party");
  }
  if (in.size() != 1) {
    throw std::logic_error("invalid number of wires for arithmetic gate");
  }
  std::unique_ptr<NewGate> gate;
  auto gate_id = gate_register_.get_next_gate_id();
  switch (in[0]->get_bit_size()) {
    case 8: {
      gate = std::make_unique<ArithmeticGMWOutputGate<std::uint8_t>>(
          gate_id, *this, cast_arith_wire<std::uint8_t>(in[0]), output_owner);
      break;
    }
    case 16: {
      gate = std::make_unique<ArithmeticGMWOutputGate<std::uint16_t>>(
          gate_id, *this, cast_arith_wire<std::uint16_t>(in[0]), output_owner);
      break;
    }
    case 32: {
      gate = std::make_unique<ArithmeticGMWOutputGate<std::uint32_t>>(
          gate_id, *this, cast_arith_wire<std::uint32_t>(in[0]), output_owner);
      break;
    }
    case 64: {
      gate = std::make_unique<ArithmeticGMWOutputGate<std::uint64_t>>(
          gate_id, *this, cast_arith_wire<std::uint64_t>(in[0]), output_owner);
      break;
    }
    default: {
      throw std::logic_error("unsupprted bit size");
    }
  }
  gate_register_.register_gate(std::move(gate));
}

std::vector<std::shared_ptr<NewWire>> GMWProvider::make_unary_gate(
    ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>& in_a) {
  // TODO
  auto input_a = cast_wires(in_a);
  BooleanGMWWireVector output;

  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::INV:
      output = make_inv_gate(std::move(input_a));
      break;
    default:
      throw std::logic_error(fmt::format("GMW does not support the unary operation {}", op));
  }

  return cast_wires(std::move(output));
}

std::vector<std::shared_ptr<NewWire>> GMWProvider::make_binary_gate(
    ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>& in_a,
    const std::vector<std::shared_ptr<NewWire>>& in_b) {
  auto input_a = cast_wires(in_a);
  auto input_b = cast_wires(in_b);
  BooleanGMWWireVector output;

  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::XOR:
      output = make_xor_gate(std::move(input_a), std::move(input_b));
      break;
    case ENCRYPTO::PrimitiveOperationType::AND:
      output = make_and_gate(std::move(input_a), std::move(input_b));
      break;
    default:
      throw std::logic_error(fmt::format("GMW does not support the binary operation {}", op));
  }

  return cast_wires(std::move(output));
}

BooleanGMWWireVector GMWProvider::make_inv_gate(BooleanGMWWireVector&& in_a) {
  BooleanGMWWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanGMWINVGate>(gate_id, *this, std::move(in_a));
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return output;
}

BooleanGMWWireVector GMWProvider::make_xor_gate(BooleanGMWWireVector&& in_a, BooleanGMWWireVector&& in_b) {
  BooleanGMWWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanGMWXORGate>(gate_id, std::move(in_a), std::move(in_b));
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return output;
}

BooleanGMWWireVector GMWProvider::make_and_gate(BooleanGMWWireVector&& in_a, BooleanGMWWireVector&& in_b) {
  BooleanGMWWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanGMWANDGate>(gate_id, *this, std::move(in_a), std::move(in_b));
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return output;
}

static flatbuffers::FlatBufferBuilder build_gmw_gate_message(std::size_t gate_id,
                                                             const std::uint8_t *message,
                                                             std::size_t size) {
  flatbuffers::FlatBufferBuilder builder;
  auto vector = builder.CreateVector(message, size);
  auto root = Communication::CreateGMWGateMessage(builder, gate_id, vector);
  builder.Finish(root);
  return Communication::BuildMessage(Communication::MessageType::GMWGate,
                                     builder.GetBufferPointer(), builder.GetSize());
}

static flatbuffers::FlatBufferBuilder build_gmw_gate_message(std::size_t gate_id,
                                                             const ENCRYPTO::BitVector<> &message) {
  const auto &vector = message.GetData();
  return build_gmw_gate_message(gate_id, reinterpret_cast<const std::uint8_t *>(vector.data()),
                                vector.size());
}

void GMWProvider::broadcast_bits_message(std::size_t gate_id,
                                         ENCRYPTO::BitVector<> &&message) const {
  broadcast_bits_message(gate_id, message);
}

void GMWProvider::broadcast_bits_message(std::size_t gate_id,
                                         const ENCRYPTO::BitVector<> &message) const {
  communication_layer_.broadcast_message(build_gmw_gate_message(gate_id, message));
}

void GMWProvider::send_bits_message(std::size_t party_id, std::size_t gate_id,
                                    ENCRYPTO::BitVector<> &&message) const {
  send_bits_message(party_id, gate_id, message);
}

void GMWProvider::send_bits_message(std::size_t party_id, std::size_t gate_id,
                                    const ENCRYPTO::BitVector<> &message) const {
  communication_layer_.send_message(party_id, build_gmw_gate_message(gate_id, message));
}

[[nodiscard]] std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>>
GMWProvider::register_for_bits_messages(std::size_t gate_id, std::size_t num_bits) {
  auto& mh = *message_handler_;
  std::vector<ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>> promises(num_parties_);
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>> futures;
  std::transform(std::begin(promises), std::end(promises), std::back_inserter(futures),
                 [](auto& p) { return p.get_future(); });
  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    auto [_, success] = mh.bits_promises_.at(party_id).insert(
        {gate_id, std::make_pair(num_bits, std::move(promises.at(party_id)))});
    if (!success) {
      throw std::logic_error(
          fmt::format("tried to register twice for yao message for gate {}", gate_id));
    }
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for bits messages of size {}", gate_id, num_bits));
    }
  }
  return futures;
}

}  // namespace MOTION::proto::gmw
