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

#include "yao_provider.h"

#include <fmt/format.h>
#include <memory>

#include "base/gate_register.h"
#include "communication/communication_layer.h"
#include "communication/fbs_headers/yao_message_generated.h"
#include "communication/message.h"
#include "communication/message_handler.h"
#include "conversion.h"
#include "crypto/garbling/half_gates.h"
#include "gate.h"
#include "protocols/gmw/wire.h"
#include "utility/constants.h"
#include "utility/logger.h"
#include "utility/typedefs.h"

namespace MOTION::proto::yao {

struct YaoMessageHandler : public Communication::MessageHandler {
  YaoMessageHandler(std::shared_ptr<Logger> logger);
  void received_message(std::size_t, std::vector<std::uint8_t> &&raw_message) override;

  ENCRYPTO::ReusableFiberPromise<Crypto::garbling::HalfGatePublicData> hg_public_data_promise_;
  ENCRYPTO::ReusableFiberFuture<Crypto::garbling::HalfGatePublicData> hg_public_data_future_;
  std::unordered_map<
      std::size_t,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::block128_vector>>>
      blocks_promises_;
  std::unordered_map<std::size_t,
                     std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>>>
      bits_promises_;
  std::shared_ptr<Logger> logger_;
};

YaoMessageHandler::YaoMessageHandler(std::shared_ptr<Logger> logger)
    : hg_public_data_future_(hg_public_data_promise_.get_future()), logger_(logger) {}

void YaoMessageHandler::received_message([[maybe_unused]] std::size_t party_id,
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
    case Communication::MessageType::YaoSetup: {
      flatbuffers::Verifier verifier(message->payload()->data(), message->payload()->size());
      auto setup_message =
          flatbuffers::GetRoot<MOTION::Communication::YaoSetupMessage>(message->payload()->data());
      if (!setup_message->Verify(verifier)) {
        throw std::runtime_error("received malformed YaoSetupMessage");
        // TODO: log and drop instead
      }
      Crypto::garbling::HalfGatePublicData public_data;
      public_data.aes_key.load_from_memory(
          reinterpret_cast<const std::byte *>(setup_message->aes_key()->data()));
      public_data.hash_key.load_from_memory(
          reinterpret_cast<const std::byte *>(setup_message->hash_key()->data()));
      try {
        hg_public_data_promise_.set_value(std::move(public_data));
      } catch (std::future_error &e) {
        // TODO: log and drop instead
        throw std::runtime_error(
            fmt::format("received second YaoSetupMessage from party {}: {}", party_id, e.what()));
      }
      break;
    }
    case Communication::MessageType::YaoGate: {
      flatbuffers::Verifier verifier(message->payload()->data(), message->payload()->size());
      auto gate_message =
          flatbuffers::GetRoot<MOTION::Communication::YaoGateMessage>(message->payload()->data());
      if (!gate_message->Verify(verifier)) {
        throw std::runtime_error("received malformed YaoGateMessage");
        // TODO: log and drop instead
      }
      auto gate_id = gate_message->gate_id();
      auto payload = gate_message->payload();
      if (auto it = blocks_promises_.find(gate_id); it != blocks_promises_.end()) {
        auto &[num_blocks, promise] = it->second;
        if (payload->size() == num_blocks * 16) {
          promise.set_value(ENCRYPTO::block128_vector(num_blocks, payload->data()));
        } else if (logger_) {
          logger_->LogError(fmt::format(
              "received YaoGateMessage for gate {} of size {} while expecting size {}, dropping",
              gate_id, payload->size(), num_blocks * 16));
        }
      } else if (auto it = bits_promises_.find(gate_id); it != bits_promises_.end()) {
        auto &[num_bits, promise] = it->second;
        auto num_bytes = (num_bits + 7) / 8;
        if (payload->size() == num_bytes) {
          promise.set_value(ENCRYPTO::BitVector(payload->data(), num_bits));
        } else if (logger_) {
          logger_->LogError(fmt::format(
              "received YaoGateMessage for gate {} of size {} while expecting size {}, dropping",
              gate_id, payload->size(), num_bytes));
        }
      } else {
        logger_->LogError(
            fmt::format("received unexpected YaoGateMessage for gate {}, dropping", gate_id));
      }
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
}

YaoProvider::YaoProvider(Communication::CommunicationLayer &communication_layer,
                         GateRegister &gate_register,
                         Crypto::MotionBaseProvider &motion_base_provider,
                         ENCRYPTO::ObliviousTransfer::OTProvider &ot_provider,
                         std::shared_ptr<Logger> logger)
    : communication_layer_(communication_layer),
      gate_register_(gate_register),
      motion_base_provider_(motion_base_provider),
      ot_provider_(ot_provider),
      hg_garbler_(nullptr),
      hg_evaluator_(nullptr),
      message_handler_(std::make_unique<YaoMessageHandler>(logger)),
      my_id_(communication_layer_.get_my_id()),
      role_((my_id_ == 0) ? Role::garbler : Role::evaluator),
      setup_ran_(false),
      logger_(std::move(logger)) {
  if (communication_layer.get_num_parties() != 2) {
    throw std::logic_error("Yao is a two party protocol");
  }
  if (role_ == Role::garbler) {
    communication_layer_.register_message_handler([this](auto) { return message_handler_; },
                                                  {Communication::MessageType::YaoGate});
  } else {
    communication_layer_.register_message_handler(
        [this](auto) { return message_handler_; },
        {Communication::MessageType::YaoSetup, Communication::MessageType::YaoGate});
  }
}

YaoProvider::~YaoProvider() = default;

static YaoWireVector cast_wires(std::vector<std::shared_ptr<NewWire>> wires) {
  YaoWireVector result(wires.size());
  std::transform(std::begin(wires), std::end(wires), std::begin(result),
                 [](auto &w) { return std::dynamic_pointer_cast<YaoWire>(w); });
  return result;
}

static std::vector<std::shared_ptr<NewWire>> cast_wires(YaoWireVector &&wires) {
  return std::vector<std::shared_ptr<NewWire>>(std::begin(wires), std::end(wires));
}

std::pair<ENCRYPTO::ReusableFiberPromise<BitValues>, WireVector>
YaoProvider::make_boolean_input_gate_my(std::size_t input_owner, std::size_t num_wires,
                                        std::size_t num_simd) {
  if (input_owner != my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  YaoWireVector output;
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> promise;
  auto gate_id = gate_register_.get_next_gate_id();
  std::unique_ptr<detail::BasicYaoInputGate> gate;
  if (role_ == Role::garbler) {
    gate = std::make_unique<YaoInputGateGarbler>(gate_id, *this, num_wires, num_simd,
                                                 promise.get_future());
  } else {
    gate = std::make_unique<YaoInputGateEvaluator>(gate_id, *this, num_wires, num_simd,
                                                   promise.get_future());
  }
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return {std::move(promise), cast_wires(std::move(output))};
}

WireVector YaoProvider::make_boolean_input_gate_other(std::size_t input_owner,
                                                      std::size_t num_wires, std::size_t num_simd) {
  if (input_owner == my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  YaoWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  std::unique_ptr<detail::BasicYaoInputGate> gate;
  if (role_ == Role::garbler) {
    gate = std::make_unique<YaoInputGateGarbler>(gate_id, *this, num_wires, num_simd);
  } else {
    gate = std::make_unique<YaoInputGateEvaluator>(gate_id, *this, num_wires, num_simd);
  }
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return cast_wires(std::move(output));
}

ENCRYPTO::ReusableFiberFuture<BitValues> YaoProvider::make_boolean_output_gate_my(
    std::size_t output_owner, const WireVector &in) {
  if (output_owner == 1 - my_id_) {
    throw std::logic_error("trying to create output gate for wrong party");
  }
  auto output_recipient =
      (output_owner == ALL_PARTIES) ? OutputRecipient::both : static_cast<OutputRecipient>(my_id_);
  auto gate_id = gate_register_.get_next_gate_id();
  auto input = cast_wires(in);
  std::unique_ptr<detail::BasicYaoOutputGate> gate;
  if (role_ == Role::garbler) {
    gate =
        std::make_unique<YaoOutputGateGarbler>(gate_id, *this, std::move(input), output_recipient);
  } else {
    gate = std::make_unique<YaoOutputGateEvaluator>(gate_id, *this, std::move(input),
                                                    output_recipient);
  }
  auto future = gate->get_output_future();
  gate_register_.register_gate(std::move(gate));
  return future;
}

void YaoProvider::make_boolean_output_gate_other(std::size_t output_owner, const WireVector &in) {
  if (output_owner != 1 - my_id_) {
    throw std::logic_error("trying to create output gate for wrong party");
  }
  auto output_recipient = static_cast<OutputRecipient>(1 - my_id_);
  auto gate_id = gate_register_.get_next_gate_id();
  auto input = cast_wires(in);
  std::unique_ptr<detail::BasicYaoOutputGate> gate;
  if (role_ == Role::garbler) {
    gate =
        std::make_unique<YaoOutputGateGarbler>(gate_id, *this, std::move(input), output_recipient);
  } else {
    gate = std::make_unique<YaoOutputGateEvaluator>(gate_id, *this, std::move(input),
                                                    output_recipient);
  }
  gate_register_.register_gate(std::move(gate));
}

std::vector<std::shared_ptr<NewWire>> YaoProvider::make_unary_gate(
    ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>> &in_a) {
  auto input_a = cast_wires(in_a);
  YaoWireVector output;

  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::INV:
      output = make_inv_gate(std::move(input_a));
      break;
    default:
      throw std::logic_error(fmt::format("Yao does not support the unary operation {}", op));
  }

  return cast_wires(std::move(output));
}

std::vector<std::shared_ptr<NewWire>> YaoProvider::make_binary_gate(
    ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>> &in_a,
    const std::vector<std::shared_ptr<NewWire>> &in_b) {
  auto input_a = cast_wires(in_a);
  auto input_b = cast_wires(in_b);
  YaoWireVector output;

  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::XOR:
      output = make_xor_gate(std::move(input_a), std::move(input_b));
      break;
    case ENCRYPTO::PrimitiveOperationType::AND:
      output = make_and_gate(std::move(input_a), std::move(input_b));
      break;
    default:
      throw std::logic_error(fmt::format("Yao does not support the binary operation {}", op));
  }

  return cast_wires(std::move(output));
}

YaoWireVector YaoProvider::make_inv_gate(YaoWireVector &&in_a) {
  if (role_ == Role::garbler) {
    auto num_wires = in_a.size();
    auto num_simd = in_a.at(0)->get_num_simd();
    YaoWireVector output(num_wires);
    for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
      auto wire = std::make_shared<YaoWire>(num_simd);
      wire->get_keys() = in_a.at(wire_i)->get_keys() ^ get_global_offset();
      output.at(wire_i) = std::move(wire);
    }
    return output;
  } else {
    return std::move(in_a);
  }
}

YaoWireVector YaoProvider::make_xor_gate(YaoWireVector &&in_a, YaoWireVector &&in_b) {
  YaoWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  std::unique_ptr<detail::BasicYaoBinaryGate> gate;
  if (role_ == Role::garbler) {
    gate = std::make_unique<YaoXORGateGarbler>(gate_id, *this, std::move(in_a), std::move(in_b));
  } else {
    gate = std::make_unique<YaoXORGateEvaluator>(gate_id, *this, std::move(in_a), std::move(in_b));
  }
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return output;
}

YaoWireVector YaoProvider::make_and_gate(YaoWireVector &&in_a, YaoWireVector &&in_b) {
  YaoWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  std::unique_ptr<detail::BasicYaoBinaryGate> gate;
  if (role_ == Role::garbler) {
    gate = std::make_unique<YaoANDGateGarbler>(gate_id, *this, std::move(in_a), std::move(in_b));
  } else {
    gate = std::make_unique<YaoANDGateEvaluator>(gate_id, *this, std::move(in_a), std::move(in_b));
  }
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return output;
}

static flatbuffers::FlatBufferBuilder build_yao_setup_message(
    const ENCRYPTO::block128_t &aes_key, const ENCRYPTO::block128_t &hash_key) {
  flatbuffers::FlatBufferBuilder builder;
  auto aes_vector =
      builder.CreateVector(reinterpret_cast<const std::uint8_t *>(aes_key.data()), aes_key.size());
  auto hash_vector = builder.CreateVector(reinterpret_cast<const std::uint8_t *>(hash_key.data()),
                                          hash_key.size());
  auto root = Communication::CreateYaoSetupMessage(builder, aes_vector, hash_vector);
  builder.Finish(root);
  return Communication::BuildMessage(Communication::MessageType::YaoSetup,
                                     builder.GetBufferPointer(), builder.GetSize());
}

ENCRYPTO::block128_t YaoProvider::get_global_offset() const {
  if (role_ == Role::evaluator) {
    throw std::logic_error("global offset is unknown to evaluator");
  }
  if (!setup_ran_) {
    throw std::logic_error("setup phase not executed, global offset is not set yet");
  }
  assert(hg_garbler_);
  return hg_garbler_->get_offset();
}

void YaoProvider::setup() {
  if (setup_ran_) {
    throw std::logic_error("YaoProvider::setup already ran");
  }
  if (role_ == Role::garbler) {
    hg_garbler_ = std::make_unique<Crypto::garbling::HalfGateGarbler>();
    auto public_data = hg_garbler_->get_public_data();
    communication_layer_.broadcast_message(
        build_yao_setup_message(public_data.aes_key, public_data.hash_key));
  } else {
    auto public_data = message_handler_->hg_public_data_future_.get();
    hg_evaluator_ = std::make_unique<Crypto::garbling::HalfGateEvaluator>(public_data);
  }
  setup_ran_ = true;
}

static flatbuffers::FlatBufferBuilder build_yao_gate_message(std::size_t gate_id,
                                                             const std::uint8_t *message,
                                                             std::size_t size) {
  flatbuffers::FlatBufferBuilder builder;
  auto vector = builder.CreateVector(message, size);
  auto root = Communication::CreateYaoGateMessage(builder, gate_id, vector);
  builder.Finish(root);
  return Communication::BuildMessage(Communication::MessageType::YaoGate,
                                     builder.GetBufferPointer(), builder.GetSize());
}

static flatbuffers::FlatBufferBuilder build_yao_gate_message(
    std::size_t gate_id, const ENCRYPTO::block128_vector &message) {
  return build_yao_gate_message(
      gate_id, reinterpret_cast<const std::uint8_t *>(message.data()->data()), message.byte_size());
}

static flatbuffers::FlatBufferBuilder build_yao_gate_message(std::size_t gate_id,
                                                             const ENCRYPTO::BitVector<> &message) {
  const auto &vector = message.GetData();
  return build_yao_gate_message(gate_id, reinterpret_cast<const std::uint8_t *>(vector.data()),
                                vector.size());
}
void YaoProvider::send_blocks_message(std::size_t gate_id,
                                      ENCRYPTO::block128_vector &&message) const {
  communication_layer_.broadcast_message(build_yao_gate_message(gate_id, message));
}

void YaoProvider::send_bits_message(std::size_t gate_id, ENCRYPTO::BitVector<> &&message) const {
  send_bits_message(gate_id, message);
}

void YaoProvider::send_bits_message(std::size_t gate_id,
                                    const ENCRYPTO::BitVector<> &message) const {
  communication_layer_.broadcast_message(build_yao_gate_message(gate_id, message));
}

ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector> YaoProvider::register_for_blocks_message(
    std::size_t gate_id, std::size_t num_blocks) {
  auto &mh = *message_handler_;
  {
    auto it = mh.bits_promises_.find(gate_id);
    if (it != mh.bits_promises_.end()) {
      throw std::logic_error(
          fmt::format("tried to register twice for yao message for gate {}", gate_id));
    }
  }
  ENCRYPTO::ReusableFiberPromise<ENCRYPTO::block128_vector> promise;
  auto future = promise.get_future();
  auto [_, success] =
      mh.blocks_promises_.insert({gate_id, std::make_pair(num_blocks, std::move(promise))});
  if (!success) {
    throw std::logic_error(
        fmt::format("tried to register twice for yao message for gate {}", gate_id));
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for blocks message of size {}", gate_id, num_blocks));
    }
  }
  return future;
}

ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> YaoProvider::register_for_bits_message(
    std::size_t gate_id, std::size_t num_bits) {
  auto &mh = *message_handler_;
  {
    auto it = mh.blocks_promises_.find(gate_id);
    if (it != mh.blocks_promises_.end()) {
      throw std::logic_error(
          fmt::format("tried to register twice for yao message for gate {}", gate_id));
    }
  }
  ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>> promise;
  auto future = promise.get_future();
  auto [_, success] =
      mh.bits_promises_.insert({gate_id, std::make_pair(num_bits, std::move(promise))});
  if (!success) {
    throw std::logic_error(
        fmt::format("tried to register twice for yao message for gate {}", gate_id));
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for bits message of size {}", gate_id, num_bits));
    }
  }
  return future;
}

void YaoProvider::create_garbled_tables(std::size_t gate_id,
                                        const ENCRYPTO::block128_vector &keys_a,
                                        const ENCRYPTO::block128_vector &keys_b,
                                        ENCRYPTO::block128_t *tables,
                                        ENCRYPTO::block128_vector &keys_out) const noexcept {
  assert(hg_garbler_);
  hg_garbler_->batch_garble_and(keys_out, tables, gate_id, keys_a, keys_b);
}

void YaoProvider::evaluate_garbled_tables(std::size_t gate_id,
                                          const ENCRYPTO::block128_vector &keys_a,
                                          const ENCRYPTO::block128_vector &keys_b,
                                          const ENCRYPTO::block128_t *tables,
                                          ENCRYPTO::block128_vector &keys_out) const noexcept {
  assert(hg_evaluator_);
  hg_evaluator_->batch_evaluate_and(keys_out, tables, gate_id, keys_a, keys_b);
}

static std::vector<std::shared_ptr<NewWire>> cast_wires(gmw::BooleanGMWWireVector &&wires) {
  return std::vector<std::shared_ptr<NewWire>>(std::begin(wires), std::end(wires));
}

WireVector YaoProvider::make_convert_to_boolean_gmw_gate(YaoWireVector &&in_a) {
  auto gate_id = gate_register_.get_next_gate_id();
  gmw::BooleanGMWWireVector output;
  if (role_ == Role::garbler) {
    auto gate = std::make_unique<YaoToBooleanGMWGateGarbler>(gate_id, *this, std::move(in_a));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  } else {
    auto gate = std::make_unique<YaoToBooleanGMWGateEvaluator>(gate_id, *this, std::move(in_a));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  }
  return cast_wires(std::move(output));
}

template <typename T>
WireVector YaoProvider::basic_make_convert_to_arithmetic_gmw_gate(YaoWireVector&& in_a) {
  auto num_wires = in_a.size();
  assert(num_wires == ENCRYPTO::bit_size_v<T>);
  auto conv_gate_id = gate_register_.get_next_gate_id();
  auto input_gate_id = gate_register_.get_next_gate_id();
  auto num_simd = in_a[0]->get_num_simd();
  YaoWireVector in_b;
  YaoToArithmeticGMWGateEvaluator<T>* y2a_gate_evaluator = nullptr;
  gmw::ArithmeticGMWWireP<T> output;
  if (role_ == Role::garbler) {
    auto conv_gate =
        std::make_unique<YaoToArithmeticGMWGateGarbler<T>>(conv_gate_id, *this, num_simd);
    output = conv_gate->get_output_wire();
    auto input_gate = std::make_unique<SetupGate<YaoInputGateGarbler>>(
        input_gate_id, *this, num_wires, num_simd, conv_gate->get_mask_future());
    in_b = input_gate->get_output_wires();
    gate_register_.register_gate(std::move(conv_gate));
    gate_register_.register_gate(std::move(input_gate));
  } else {
    auto conv_gate =
        std::make_unique<YaoToArithmeticGMWGateEvaluator<T>>(conv_gate_id, *this, num_simd);
    output = conv_gate->get_output_wire();
    y2a_gate_evaluator = conv_gate.get();
    auto input_gate = std::make_unique<SetupGate<YaoInputGateEvaluator>>(input_gate_id, *this,
                                                                         num_wires, num_simd);
    in_b = input_gate->get_output_wires();
    gate_register_.register_gate(std::move(conv_gate));
    gate_register_.register_gate(std::move(input_gate));
  }
  // TODO: build addition circuit: out <- in_a + in_b
  if (role_ == Role::garbler) {
    // TODO:  y2a_gate_evaluator->set_masked_value_future(/* output gate's future */);
  }
  return {std::dynamic_pointer_cast<NewWire>(output)};
}

WireVector YaoProvider::make_convert_to_arithmetic_gmw_gate(YaoWireVector &&in_a) {
  auto bit_size = in_a.size();
  switch (bit_size) {
    case 8:
      return basic_make_convert_to_arithmetic_gmw_gate<std::uint8_t>(std::move(in_a));
    case 16:
      return basic_make_convert_to_arithmetic_gmw_gate<std::uint16_t>(std::move(in_a));
    case 32:
      return basic_make_convert_to_arithmetic_gmw_gate<std::uint32_t>(std::move(in_a));
    case 64:
      return basic_make_convert_to_arithmetic_gmw_gate<std::uint64_t>(std::move(in_a));
    default:
      throw std::logic_error(fmt::format("unsupported bit size {} for Yao to Arithmetic GMW conversion\n", bit_size));
  }
}

WireVector YaoProvider::convert(MPCProtocol proto, const WireVector &in) {
  auto input = cast_wires(in);

  switch (proto) {
    case MPCProtocol::ArithmeticGMW:
      return make_convert_to_arithmetic_gmw_gate(std::move(input));
    case MPCProtocol::BooleanGMW:
      return make_convert_to_boolean_gmw_gate(std::move(input));
    default:
      throw std::logic_error(fmt::format("Yao does not support conversion to {}", ToString(proto)));
  }
}

}  // namespace MOTION::proto::yao
