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

#include <cstdint>
#include <unordered_map>
#include "gmw_provider.h"

#include "base/gate_register.h"
#include "communication/communication_layer.h"
#include "communication/fbs_headers/gmw_message_generated.h"
#include "communication/message.h"
#include "communication/message_handler.h"
#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/mt_provider.h"
#include "crypto/multiplication_triple/sp_provider.h"
#include "gate.h"
#include "utility/constants.h"
#include "utility/logger.h"
#include "utility/meta.hpp"
#include "wire.h"

namespace MOTION::proto::gmw {

struct GMWMessageHandler : public Communication::MessageHandler {
  GMWMessageHandler(std::size_t num_parties, std::shared_ptr<Logger> logger);
  void received_message(std::size_t, std::vector<std::uint8_t>&& raw_message) override;

  enum class MsgValueType { bit, uint8, uint16, uint32, uint64 };

  // gate_id -> (size, type)
  std::unordered_map<std::size_t, std::pair<std::size_t, MsgValueType>> expected_messages_;

  // [gate_id -> promise]
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>>>
      bits_promises_;
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint8_t>>>>
      uint8_promises_;
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint16_t>>>>
      uint16_promises_;
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint32_t>>>>
      uint32_promises_;
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint64_t>>>>
      uint64_promises_;

  template <typename T>
  std::vector<std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<T>>>>&
  get_promise_map();

  std::shared_ptr<Logger> logger_;
};

template <>
std::vector<
    std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint8_t>>>>&
GMWMessageHandler::get_promise_map() {
  return uint8_promises_;
}
template <>
std::vector<
    std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint16_t>>>>&
GMWMessageHandler::get_promise_map() {
  return uint16_promises_;
}
template <>
std::vector<
    std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint32_t>>>>&
GMWMessageHandler::get_promise_map() {
  return uint32_promises_;
}
template <>
std::vector<
    std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint64_t>>>>&
GMWMessageHandler::get_promise_map() {
  return uint64_promises_;
}

GMWMessageHandler::GMWMessageHandler(std::size_t num_parties, std::shared_ptr<Logger> logger)
    : bits_promises_(num_parties),
      uint8_promises_(num_parties),
      uint16_promises_(num_parties),
      uint32_promises_(num_parties),
      uint64_promises_(num_parties),
      logger_(logger) {}

void GMWMessageHandler::received_message(std::size_t party_id,
                                         std::vector<std::uint8_t>&& raw_message) {
  assert(!raw_message.empty());
  auto message = Communication::GetMessage(raw_message.data());
  {
    flatbuffers::Verifier verifier(raw_message.data(), raw_message.size());
    if (!message->Verify(verifier)) {
      throw std::runtime_error("received malformed Message");
      // TODO: log and drop instead
    }
  }

  auto message_type = message->message_type();
  if (message_type != Communication::MessageType::GMWGate) {
    throw std::logic_error(fmt::format("GMWMessageHandle: received unexpected message of type {}",
                                       EnumNameMessageType(message_type)));
  }

  auto gate_message =
      flatbuffers::GetRoot<MOTION::Communication::GMWGateMessage>(message->payload()->data());
  {
    flatbuffers::Verifier verifier(message->payload()->data(), message->payload()->size());
    if (!gate_message->Verify(verifier)) {
      throw std::runtime_error("received malformed GMWGateMessage");
      // TODO: log and drop instead
    }
  }
  auto gate_id = gate_message->gate_id();
  auto payload = gate_message->payload();
  auto it = expected_messages_.find(gate_id);
  if (it == expected_messages_.end()) {
    logger_->LogError(
        fmt::format("received unexpected GMWGateMessage for gate {}, dropping", gate_id));
    return;
  }
  auto expected_size = it->second.first;
  auto type = it->second.second;

  auto set_value_helper = [this, party_id, gate_id, expected_size, payload](auto& map_vec,
                                                                            auto type_tag) {
    auto byte_size = expected_size * sizeof(type_tag);
    if (byte_size != payload->size()) {
      logger_->LogError(fmt::format(
          "received GMWGateMessage for gate {} of size {} while expecting size {}, dropping",
          gate_id, payload->size(), byte_size));
      return;
    }
    auto& promise_map = map_vec[party_id];
    auto& promise = promise_map.at(gate_id);
    auto ptr = reinterpret_cast<const decltype(type_tag)*>(payload->data());
    promise.set_value(std::vector(ptr, ptr + expected_size));
  };

  switch (type) {
    case MsgValueType::bit: {
      auto byte_size = Helpers::Convert::BitsToBytes(expected_size);
      if (byte_size != payload->size()) {
        logger_->LogError(fmt::format(
            "received GMWGateMessage for gate {} of size {} while expecting size {}, dropping",
            gate_id, payload->size(), byte_size));
        return;
      }
      auto& promise = bits_promises_[party_id].at(gate_id);
      promise.set_value(ENCRYPTO::BitVector(payload->data(), expected_size));
      break;
    }
    case MsgValueType::uint8: {
      set_value_helper(uint8_promises_, std::uint8_t{});
      break;
    }
    case MsgValueType::uint16: {
      set_value_helper(uint16_promises_, std::uint16_t{});
      break;
    }
    case MsgValueType::uint32: {
      set_value_helper(uint32_promises_, std::uint32_t{});
      break;
    }
    case MsgValueType::uint64: {
      set_value_helper(uint64_promises_, std::uint64_t{});
      break;
    }
  }
}

GMWProvider::GMWProvider(Communication::CommunicationLayer& communication_layer,
                         GateRegister& gate_register,
                         Crypto::MotionBaseProvider& motion_base_provider, MTProvider& mt_provider, SPProvider& sp_provider,
                         std::shared_ptr<Logger> logger)
    : communication_layer_(communication_layer),
      gate_register_(gate_register),
      motion_base_provider_(motion_base_provider),
      mt_provider_(mt_provider),
      sp_provider_(sp_provider),
      message_handler_(
          std::make_unique<GMWMessageHandler>(communication_layer_.get_num_parties(), logger)),
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
  sp_provider_.WaitFinished();
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
  auto ptr = std::dynamic_pointer_cast<ArithmeticGMWWire<T>>(wire);
  assert(ptr);
  return ptr;
}

template <typename T>
static std::shared_ptr<NewWire> cast_arith_wire(ArithmeticGMWWireP<T> wire) {
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
ENCRYPTO::ReusableFiberFuture<IntegerValues<T>> GMWProvider::basic_make_arithmetic_output_gate_my(
    std::size_t output_owner, const WireVector& in) {
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

void GMWProvider::make_arithmetic_output_gate_other(std::size_t output_owner,
                                                    const WireVector& in) {
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
  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::INV:
      return make_inv_gate(in_a);
    case ENCRYPTO::PrimitiveOperationType::NEG:
      return make_neg_gate(in_a);
    case ENCRYPTO::PrimitiveOperationType::SQR:
      return make_sqr_gate(in_a);
    default:
      throw std::logic_error(fmt::format("GMW does not support the unary operation {}", op));
  }
}

std::vector<std::shared_ptr<NewWire>> GMWProvider::make_binary_gate(
    ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>& in_a,
    const std::vector<std::shared_ptr<NewWire>>& in_b) {
  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::XOR:
      return make_xor_gate(in_a, in_b);
    case ENCRYPTO::PrimitiveOperationType::AND:
      return make_and_gate(in_a, in_b);
    case ENCRYPTO::PrimitiveOperationType::ADD:
      return make_add_gate(in_a, in_b);
    case ENCRYPTO::PrimitiveOperationType::MUL:
      return make_mul_gate(in_a, in_b);
    default:
      throw std::logic_error(fmt::format("GMW does not support the binary operation {}", op));
  }
}

WireVector GMWProvider::make_inv_gate(const WireVector& in_a) {
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanGMWINVGate>(gate_id, *this, cast_wires(in_a));
  auto output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return cast_wires(std::move(output));
}

WireVector GMWProvider::make_xor_gate(const WireVector&in_a,
                                                const WireVector& in_b) {
  BooleanGMWWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanGMWXORGate>(gate_id, cast_wires(in_a), cast_wires(in_b));
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return cast_wires(std::move(output));
}

WireVector GMWProvider::make_and_gate(const WireVector& in_a,
                                                const WireVector& in_b) {
  BooleanGMWWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanGMWANDGate>(gate_id, *this, cast_wires(in_a), cast_wires(in_b));
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return cast_wires(std::move(output));
}

static std::size_t check_arithmetic_wire(const WireVector& in) {
  if (in.size() != 1) {
    throw std::logic_error("arithmetic operations support single wires only");
  }
  return in[0]->get_bit_size();
}

template <template <typename> class UnaryGate, typename T>
WireVector GMWProvider::make_arithmetic_unary_gate(const NewWireP& in_a) {
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<UnaryGate<T>>(
      gate_id, *this, cast_arith_wire<T>(in_a));
  auto output = {cast_arith_wire(gate->get_output_wire())};
  gate_register_.register_gate(std::move(gate));
  return output;
}

template <template <typename> class UnaryGate>
WireVector GMWProvider::make_arithmetic_unary_gate(const WireVector& in_a) {
  auto bit_size = check_arithmetic_wire(in_a);
  switch (bit_size) {
    case 8:
      return make_arithmetic_unary_gate<UnaryGate, std::uint8_t>(in_a[0]);
    case 16:
      return make_arithmetic_unary_gate<UnaryGate, std::uint16_t>(in_a[0]);
    case 32:
      return make_arithmetic_unary_gate<UnaryGate, std::uint32_t>(in_a[0]);
    case 64:
      return make_arithmetic_unary_gate<UnaryGate, std::uint64_t>(in_a[0]);
    default:
      throw std::logic_error(fmt::format("unexpected bit size {}", bit_size));
  }
}

static std::size_t check_arithmetic_wires(const WireVector& in_a, const WireVector& in_b) {
  if (in_a.size() != 1 || in_b.size() != 1) {
    throw std::logic_error("arithmetic operations support single wires only");
  }
  auto bit_size = in_a[0]->get_bit_size();
  if (bit_size != in_b[0]->get_bit_size()) {
    throw std::logic_error("different bit sizes on wires");
  }
  return bit_size;
}

template <template <typename> class BinaryGate, typename T>
WireVector GMWProvider::make_arithmetic_binary_gate(const NewWireP& in_a, const NewWireP& in_b) {
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BinaryGate<T>>(
      gate_id, *this, cast_arith_wire<T>(in_a), cast_arith_wire<T>(in_b));
  auto output = {cast_arith_wire(gate->get_output_wire())};
  gate_register_.register_gate(std::move(gate));
  return output;
}

template <template <typename> class BinaryGate>
WireVector GMWProvider::make_arithmetic_binary_gate(const WireVector& in_a,
                                                    const WireVector& in_b) {
  auto bit_size = check_arithmetic_wires(in_a, in_b);
  switch (bit_size) {
    case 8:
      return make_arithmetic_binary_gate<BinaryGate, std::uint8_t>(in_a[0], in_b[0]);
    case 16:
      return make_arithmetic_binary_gate<BinaryGate, std::uint16_t>(in_a[0], in_b[0]);
    case 32:
      return make_arithmetic_binary_gate<BinaryGate, std::uint32_t>(in_a[0], in_b[0]);
    case 64:
      return make_arithmetic_binary_gate<BinaryGate, std::uint64_t>(in_a[0], in_b[0]);
    default:
      throw std::logic_error(fmt::format("unexpected bit size {}", bit_size));
  }
}

WireVector GMWProvider::make_neg_gate(const WireVector& in) {
  return make_arithmetic_unary_gate<ArithmeticGMWNEGGate>(in);
}

WireVector GMWProvider::make_add_gate(const WireVector& in_a, const WireVector& in_b) {
  return make_arithmetic_binary_gate<ArithmeticGMWADDGate>(in_a, in_b);
}

WireVector GMWProvider::make_mul_gate(const WireVector& in_a, const WireVector& in_b) {
  return make_arithmetic_binary_gate<ArithmeticGMWMULGate>(in_a, in_b);
}

WireVector GMWProvider::make_sqr_gate(const WireVector& in) {
  return make_arithmetic_unary_gate<ArithmeticGMWSQRGate>(in);
}

static flatbuffers::FlatBufferBuilder build_gmw_gate_message(std::size_t gate_id,
                                                             const std::uint8_t* message,
                                                             std::size_t size) {
  flatbuffers::FlatBufferBuilder builder;
  auto vector = builder.CreateVector(message, size);
  auto root = Communication::CreateGMWGateMessage(builder, gate_id, vector);
  builder.Finish(root);
  return Communication::BuildMessage(Communication::MessageType::GMWGate,
                                     builder.GetBufferPointer(), builder.GetSize());
}

template <typename T>
static flatbuffers::FlatBufferBuilder build_gmw_gate_message(std::size_t gate_id,
                                                             const std::vector<T>& vector) {
  return build_gmw_gate_message(gate_id, reinterpret_cast<const std::uint8_t*>(vector.data()),
                                sizeof(T) * vector.size());
}

static flatbuffers::FlatBufferBuilder build_gmw_gate_message(std::size_t gate_id,
                                                             const ENCRYPTO::BitVector<>& message) {
  return build_gmw_gate_message(gate_id, message.GetData());
}

void GMWProvider::broadcast_bits_message(std::size_t gate_id,
                                         const ENCRYPTO::BitVector<>& message) const {
  communication_layer_.broadcast_message(build_gmw_gate_message(gate_id, message));
}

void GMWProvider::send_bits_message(std::size_t party_id, std::size_t gate_id,
                                    const ENCRYPTO::BitVector<>& message) const {
  communication_layer_.send_message(party_id, build_gmw_gate_message(gate_id, message));
}

[[nodiscard]] std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>>
GMWProvider::register_for_bits_messages(std::size_t gate_id, std::size_t num_bits) {
  auto& mh = *message_handler_;
  std::vector<ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>> promises(num_parties_);
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>> futures;
  std::transform(std::begin(promises), std::end(promises), std::back_inserter(futures),
                 [](auto& p) { return p.get_future(); });
  auto [_, success] = mh.expected_messages_.insert(
      {gate_id, std::make_pair(num_bits, GMWMessageHandler::MsgValueType::bit)});
  if (!success) {
    throw std::logic_error(
        fmt::format("tried to register twice for GMWGate message for gate {}", gate_id));
  }
  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    auto& promise_map = mh.bits_promises_.at(party_id);
    auto [_, success] = promise_map.insert({gate_id, std::move(promises.at(party_id))});
    assert(success);
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for bits messages of size {}", gate_id, num_bits));
    }
  }
  return futures;
}

template <typename T>
constexpr static GMWMessageHandler::MsgValueType get_msg_value_type() {
  if constexpr (std::is_same_v<T, std::uint8_t>) {
    return GMWMessageHandler::MsgValueType::uint8;
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return GMWMessageHandler::MsgValueType::uint16;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return GMWMessageHandler::MsgValueType::uint32;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return GMWMessageHandler::MsgValueType::uint64;
  }
}

template <typename T>
void GMWProvider::broadcast_ints_message(std::size_t gate_id,
                                         const std::vector<T>& message) const {
  communication_layer_.broadcast_message(build_gmw_gate_message(gate_id, message));
}

template void GMWProvider::broadcast_ints_message(std::size_t,
                                                  const std::vector<std::uint8_t>&) const;
template void GMWProvider::broadcast_ints_message(std::size_t,
                                                  const std::vector<std::uint16_t>&) const;
template void GMWProvider::broadcast_ints_message(std::size_t,
                                                  const std::vector<std::uint32_t>&) const;
template void GMWProvider::broadcast_ints_message(std::size_t,
                                                  const std::vector<std::uint64_t>&) const;

template <typename T>
void GMWProvider::send_ints_message(std::size_t party_id, std::size_t gate_id,
                                    const std::vector<T>& message) const {
  communication_layer_.send_message(party_id, build_gmw_gate_message(gate_id, message));
}

template void GMWProvider::send_ints_message(std::size_t, std::size_t,
                                             const std::vector<std::uint8_t>&) const;
template void GMWProvider::send_ints_message(std::size_t, std::size_t,
                                             const std::vector<std::uint16_t>&) const;
template void GMWProvider::send_ints_message(std::size_t, std::size_t,
                                             const std::vector<std::uint32_t>&) const;
template void GMWProvider::send_ints_message(std::size_t, std::size_t,
                                             const std::vector<std::uint64_t>&) const;

template <typename T>
[[nodiscard]] std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<T>>>
GMWProvider::register_for_ints_messages(std::size_t gate_id, std::size_t num_elements) {
  auto& mh = *message_handler_;
  std::vector<ENCRYPTO::ReusableFiberPromise<std::vector<T>>> promises(num_parties_);
  std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<T>>> futures;
  std::transform(std::begin(promises), std::end(promises), std::back_inserter(futures),
                 [](auto& p) { return p.get_future(); });
  GMWMessageHandler::MsgValueType type = get_msg_value_type<T>();
  auto [_, success] = mh.expected_messages_.insert({gate_id, std::make_pair(num_elements, type)});
  if (!success) {
    throw std::logic_error(
        fmt::format("tried to register twice for GMWGate message for gate {}", gate_id));
  }
  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    auto& promise_map = mh.get_promise_map<T>().at(party_id);
    auto [_, success] = promise_map.insert({gate_id, std::move(promises.at(party_id))});
    assert(success);
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for int messages of size {}", gate_id, num_elements));
    }
  }
  return futures;
}

template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint8_t>>>
    GMWProvider::register_for_ints_messages(std::size_t, std::size_t);
template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint16_t>>>
    GMWProvider::register_for_ints_messages(std::size_t, std::size_t);
template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint32_t>>>
    GMWProvider::register_for_ints_messages(std::size_t, std::size_t);
template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>>>
    GMWProvider::register_for_ints_messages(std::size_t, std::size_t);

}  // namespace MOTION::proto::gmw
