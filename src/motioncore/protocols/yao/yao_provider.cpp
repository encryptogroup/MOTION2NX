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
#include <type_traits>

#include "algorithm/circuit_loader.h"
#include "algorithm/make_circuit.h"
#include "base/gate_register.h"
#include "communication/communication_layer.h"
#include "communication/fbs_headers/yao_message_generated.h"
#include "communication/message.h"
#include "communication/message_handler.h"
#include "conversion.h"
#include "crypto/garbling/half_gates.h"
#include "gate.h"
#include "gate/input_gate_adapter.h"
#include "protocols/beavy/gate.h"
#include "protocols/beavy/wire.h"
#include "protocols/gmw/gate.h"
#include "protocols/gmw/tensor.h"
#include "protocols/gmw/wire.h"
#include "tensor_op.h"
#include "utility/constants.h"
#include "utility/logger.h"
#include "utility/typedefs.h"

namespace MOTION::proto::yao {

struct YaoMessageHandler : public Communication::MessageHandler {
  YaoMessageHandler(std::shared_ptr<Logger> logger);
  void received_message(std::size_t, std::vector<std::uint8_t>&& raw_message) override;

  ENCRYPTO::ReusableFiberPromise<Crypto::garbling::HalfGatePublicData> hg_public_data_promise_;
  ENCRYPTO::ReusableFiberFuture<Crypto::garbling::HalfGatePublicData> hg_public_data_future_;
  ENCRYPTO::ReusableFiberPromise<ENCRYPTO::block128_t> shared_zero_promise_;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_t> shared_zero_future_;
  std::shared_ptr<Logger> logger_;
};

YaoMessageHandler::YaoMessageHandler(std::shared_ptr<Logger> logger)
    : hg_public_data_future_(hg_public_data_promise_.get_future()),
      shared_zero_future_(shared_zero_promise_.get_future()),
      logger_(logger) {}

void YaoMessageHandler::received_message([[maybe_unused]] std::size_t party_id,
                                         std::vector<std::uint8_t>&& raw_message) {
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
          reinterpret_cast<const std::byte*>(setup_message->aes_key()->data()));
      public_data.hash_key.load_from_memory(
          reinterpret_cast<const std::byte*>(setup_message->hash_key()->data()));
      ENCRYPTO::block128_t shared_zero;
      shared_zero.load_from_memory(
          reinterpret_cast<const std::byte*>(setup_message->shared_zero()->data()));
      try {
        hg_public_data_promise_.set_value(std::move(public_data));
        shared_zero_promise_.set_value(std::move(shared_zero));
      } catch (std::future_error& e) {
        // TODO: log and drop instead
        throw std::runtime_error(
            fmt::format("received second YaoSetupMessage from party {}: {}", party_id, e.what()));
      }
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
}

YaoProvider::YaoProvider(Communication::CommunicationLayer& communication_layer,
                         GateRegister& gate_register, CircuitLoader& circuit_loader,
                         Crypto::MotionBaseProvider& motion_base_provider,
                         ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider,
                         std::shared_ptr<Logger> logger)
    : CommMixin(communication_layer, Communication::MessageType::YaoGate, logger),
      communication_layer_(communication_layer),
      gate_register_(gate_register),
      circuit_loader_(circuit_loader),
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
  if (role_ == Role::evaluator) {
    communication_layer_.register_message_handler([this](auto) { return message_handler_; },
                                                  {Communication::MessageType::YaoSetup});
  }
}

YaoProvider::~YaoProvider() {
  if (role_ == Role::evaluator) {
    communication_layer_.deregister_message_handler({Communication::MessageType::YaoSetup});
  }
}

static YaoWireVector cast_wires(std::vector<std::shared_ptr<NewWire>> wires) {
  YaoWireVector result(wires.size());
  std::transform(std::begin(wires), std::end(wires), std::begin(result),
                 [](auto& w) { return std::dynamic_pointer_cast<YaoWire>(w); });
  return result;
}

static std::vector<std::shared_ptr<NewWire>> cast_wires(const YaoWireVector& wires) {
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
    std::size_t output_owner, const WireVector& in) {
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

void YaoProvider::make_boolean_output_gate_other(std::size_t output_owner, const WireVector& in) {
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
    ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>& in_a) {
  auto input_a = cast_wires(in_a);
  YaoWireVector output;

  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::INV:
      output = make_inv_gate(std::move(input_a));
      break;
    default:
      throw std::logic_error(
          fmt::format("Yao does not support the unary operation {}", ToString(op)));
  }

  return cast_wires(std::move(output));
}

std::vector<std::shared_ptr<NewWire>> YaoProvider::make_binary_gate(
    ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>& in_a,
    const std::vector<std::shared_ptr<NewWire>>& in_b) {
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
      throw std::logic_error(
          fmt::format("Yao does not support the binary operation {}", ToString(op)));
  }

  return cast_wires(std::move(output));
}

YaoWireVector YaoProvider::make_inv_gate(YaoWireVector&& in_a) {
  YaoWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  std::unique_ptr<detail::BasicYaoUnaryGate> gate;
  if (role_ == Role::garbler) {
    gate = std::make_unique<YaoINVGateGarbler>(gate_id, *this, std::move(in_a));
  } else {
    gate = std::make_unique<YaoINVGateEvaluator>(gate_id, *this, std::move(in_a));
  }
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return output;
}

YaoWireVector YaoProvider::make_xor_gate(YaoWireVector&& in_a, YaoWireVector&& in_b) {
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

YaoWireVector YaoProvider::make_and_gate(YaoWireVector&& in_a, YaoWireVector&& in_b) {
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
    const ENCRYPTO::block128_t& aes_key, const ENCRYPTO::block128_t& hash_key,
    const ENCRYPTO::block128_t& shared_zero) {
  flatbuffers::FlatBufferBuilder builder;
  auto aes_vector =
      builder.CreateVector(reinterpret_cast<const std::uint8_t*>(aes_key.data()), aes_key.size());
  auto hash_vector =
      builder.CreateVector(reinterpret_cast<const std::uint8_t*>(hash_key.data()), hash_key.size());
  auto zero_vector = builder.CreateVector(reinterpret_cast<const std::uint8_t*>(shared_zero.data()),
                                          shared_zero.size());
  auto root = Communication::CreateYaoSetupMessage(builder, aes_vector, hash_vector, zero_vector);
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

ENCRYPTO::block128_t YaoProvider::get_shared_zero() const noexcept {
  assert(setup_ran_);
  return shared_zero_;
}

void YaoProvider::setup() {
  if (setup_ran_) {
    throw std::logic_error("YaoProvider::setup already ran");
  }
  if (role_ == Role::garbler) {
    shared_zero_.set_to_random();
    hg_garbler_ = std::make_unique<Crypto::garbling::HalfGateGarbler>();
    auto public_data = hg_garbler_->get_public_data();
    communication_layer_.broadcast_message(
        build_yao_setup_message(public_data.aes_key, public_data.hash_key, shared_zero_));
  } else {
    auto public_data = message_handler_->hg_public_data_future_.get();
    hg_evaluator_ = std::make_unique<Crypto::garbling::HalfGateEvaluator>(public_data);
    shared_zero_ = message_handler_->shared_zero_future_.get();
  }
  setup_ran_ = true;
}

void YaoProvider::send_blocks_message(std::size_t gate_id,
                                      ENCRYPTO::block128_vector&& message) const {
  CommMixin::send_blocks_message(1 - my_id_, gate_id, std::move(message));
}

void YaoProvider::send_bits_message(std::size_t gate_id, ENCRYPTO::BitVector<>&& message) const {
  CommMixin::send_bits_message(1 - my_id_, gate_id, std::move(message));
}

void YaoProvider::send_bits_message(std::size_t gate_id,
                                    const ENCRYPTO::BitVector<>& message) const {
  CommMixin::send_bits_message(1 - my_id_, gate_id, message);
}

ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector> YaoProvider::register_for_blocks_message(
    std::size_t gate_id, std::size_t num_blocks) {
  return CommMixin::register_for_blocks_message(1 - my_id_, gate_id, num_blocks);
}

ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> YaoProvider::register_for_bits_message(
    std::size_t gate_id, std::size_t num_bits) {
  return CommMixin::register_for_bits_message(1 - my_id_, gate_id, num_bits);
}

void YaoProvider::create_garbled_tables(std::size_t gate_id,
                                        const ENCRYPTO::block128_vector& keys_a,
                                        const ENCRYPTO::block128_vector& keys_b,
                                        ENCRYPTO::block128_t* tables,
                                        ENCRYPTO::block128_vector& keys_out) const noexcept {
  assert(hg_garbler_);
  hg_garbler_->batch_garble_and(keys_out, tables, gate_id, keys_a, keys_b);
}

void YaoProvider::evaluate_garbled_tables(std::size_t gate_id,
                                          const ENCRYPTO::block128_vector& keys_a,
                                          const ENCRYPTO::block128_vector& keys_b,
                                          const ENCRYPTO::block128_t* tables,
                                          ENCRYPTO::block128_vector& keys_out) const noexcept {
  assert(hg_evaluator_);
  hg_evaluator_->batch_evaluate_and(keys_out, tables, gate_id, keys_a, keys_b);
}

void YaoProvider::create_garbled_circuit(std::size_t gate_id, std::size_t num_simd,
                                         const ENCRYPTO::AlgorithmDescription& algo,
                                         const ENCRYPTO::block128_vector& input_keys_a,
                                         const ENCRYPTO::block128_vector& input_keys_b,
                                         ENCRYPTO::block128_vector& tables,
                                         ENCRYPTO::block128_vector& output_keys,
                                         bool parallel) const {
  assert(hg_garbler_);
  hg_garbler_->garble_circuit(output_keys, tables, gate_id, input_keys_a, input_keys_b, num_simd,
                              algo, parallel);
}

void YaoProvider::evaluate_garbled_circuit(std::size_t gate_id, std::size_t num_simd,
                                           const ENCRYPTO::AlgorithmDescription& algo,
                                           const ENCRYPTO::block128_vector& input_keys_a,
                                           const ENCRYPTO::block128_vector& input_keys_b,
                                           const ENCRYPTO::block128_vector& tables,
                                           ENCRYPTO::block128_vector& output_keys,
                                           bool parallel) const {
  assert(hg_evaluator_);
  hg_evaluator_->evaluate_circuit(output_keys, tables, gate_id, input_keys_a, input_keys_b,
                                  num_simd, algo, parallel);
}

static std::vector<std::shared_ptr<NewWire>> cast_wires(gmw::BooleanGMWWireVector&& wires) {
  return std::vector<std::shared_ptr<NewWire>>(std::begin(wires), std::end(wires));
}

WireVector YaoProvider::make_convert_to_boolean_gmw_gate(YaoWireVector&& in_a) {
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

YaoWireVector YaoProvider::make_convert_from_boolean_gmw_gate(const WireVector& in) {
  auto gate_id = gate_register_.get_next_gate_id();
  YaoWireVector output;
  gmw::BooleanGMWWireVector input;
  input.reserve(in.size());
  std::transform(std::begin(in), std::end(in), std::back_inserter(input),
                 [](auto& w) { return std::dynamic_pointer_cast<gmw::BooleanGMWWire>(w); });
  if (role_ == Role::garbler) {
    auto gate = std::make_unique<BooleanGMWToYaoGateGarbler>(gate_id, *this, std::move(input));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  } else {
    auto gate = std::make_unique<BooleanGMWToYaoGateEvaluator>(gate_id, *this, std::move(input));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  }
  return output;
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

  // sum up the additive shares in a Boolean circuit
  const auto circuit_name = fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>);
  const auto& algo = circuit_loader_.load_circuit(circuit_name, CircuitFormat::Bristol);
  auto sum_wires = make_circuit(*this, algo, cast_wires(in_a), cast_wires(in_b));

  // the output is given to the evaluator
  if (role_ == Role::garbler) {
    make_boolean_output_gate_other(1, sum_wires);
  } else {
    auto future = make_boolean_output_gate_my(1, sum_wires);
    y2a_gate_evaluator->set_masked_value_future(std::move(future));
  }
  return {std::dynamic_pointer_cast<NewWire>(output)};
}

WireVector YaoProvider::make_convert_to_arithmetic_gmw_gate(YaoWireVector&& in_a) {
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
      throw std::logic_error(
          fmt::format("unsupported bit size {} for Yao to Arithmetic GMW conversion\n", bit_size));
  }
}

template <typename T>
WireVector YaoProvider::basic_make_convert_from_arithmetic_gmw_gate(const WireVector& in) {
  assert(in.size() == 1);
  auto num_wires = ENCRYPTO::bit_size_v<T>;
  auto arith_wire = std::dynamic_pointer_cast<gmw::ArithmeticGMWWire<T>>(in.at(0));
  auto num_simd = arith_wire->get_num_simd();
  auto share_output_gate_id = gate_register_.get_next_gate_id();
  auto share_output_gate = std::make_unique<gmw::ArithmeticGMWOutputShareGate<T>>(
      share_output_gate_id, std::move(arith_wire));
  auto share_future = share_output_gate->get_output_future();
  gate_register_.register_gate(std::move(share_output_gate));
  auto input_gate_0_id = gate_register_.get_next_gate_id();
  auto input_gate_1_id = gate_register_.get_next_gate_id();
  YaoWireVector wires_0;
  YaoWireVector wires_1;
  if (role_ == Role::garbler) {
    auto input_gate_0 = std::make_unique<ArithmeticInputAdapterGate<YaoInputGateGarbler, T>>(
        std::move(share_future), [](auto x) { return x; }, logger_, input_gate_0_id, *this,
        num_wires, num_simd);
    auto input_gate_1 =
        std::make_unique<YaoInputGateGarbler>(input_gate_1_id, *this, num_wires, num_simd);
    wires_0 = input_gate_0->get_output_wires();
    wires_1 = input_gate_1->get_output_wires();
    gate_register_.register_gate(std::move(input_gate_0));
    gate_register_.register_gate(std::move(input_gate_1));
  } else {
    auto input_gate_0 =
        std::make_unique<YaoInputGateEvaluator>(input_gate_0_id, *this, num_wires, num_simd);
    auto input_gate_1 = std::make_unique<ArithmeticInputAdapterGate<YaoInputGateEvaluator, T>>(
        std::move(share_future), [](auto x) { return x; }, logger_, input_gate_1_id, *this,
        num_wires, num_simd);
    wires_0 = input_gate_0->get_output_wires();
    wires_1 = input_gate_1->get_output_wires();
    gate_register_.register_gate(std::move(input_gate_0));
    gate_register_.register_gate(std::move(input_gate_1));
  }

  // sum up the additive shares in a Boolean circuit
  const auto circuit_name = fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>);
  const auto& algo = circuit_loader_.load_circuit(circuit_name, CircuitFormat::Bristol);
  auto sum_wires = make_circuit(*this, algo, cast_wires(wires_0), cast_wires(wires_1));
  return sum_wires;
}

WireVector YaoProvider::make_convert_from_arithmetic_gmw_gate(const WireVector& in) {
  assert(in.size() == 1);
  auto bit_size = in.at(0)->get_bit_size();
  switch (bit_size) {
    case 8:
      return basic_make_convert_from_arithmetic_gmw_gate<std::uint8_t>(in);
    case 16:
      return basic_make_convert_from_arithmetic_gmw_gate<std::uint16_t>(in);
    case 32:
      return basic_make_convert_from_arithmetic_gmw_gate<std::uint32_t>(in);
    case 64:
      return basic_make_convert_from_arithmetic_gmw_gate<std::uint64_t>(in);
    default:
      throw std::logic_error(fmt::format(
          "unsupported bit size {} for Yao from Arithmetic GMW conversion\n", bit_size));
  }
}

static std::vector<std::shared_ptr<NewWire>> cast_wires(beavy::BooleanBEAVYWireVector&& wires) {
  return std::vector<std::shared_ptr<NewWire>>(std::begin(wires), std::end(wires));
}

WireVector YaoProvider::make_convert_to_boolean_beavy_gate(YaoWireVector&& in_a) {
  auto gate_id = gate_register_.get_next_gate_id();
  beavy::BooleanBEAVYWireVector output;
  if (role_ == Role::garbler) {
    auto gate = std::make_unique<YaoToBooleanBEAVYGateGarbler>(gate_id, *this, std::move(in_a));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  } else {
    auto gate = std::make_unique<YaoToBooleanBEAVYGateEvaluator>(gate_id, *this, std::move(in_a));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  }
  return cast_wires(std::move(output));
}

YaoWireVector YaoProvider::make_convert_from_boolean_beavy_gate(const WireVector& in) {
  auto gate_id = gate_register_.get_next_gate_id();
  YaoWireVector output;
  beavy::BooleanBEAVYWireVector input;
  input.reserve(in.size());
  std::transform(std::begin(in), std::end(in), std::back_inserter(input),
                 [](auto& w) { return std::dynamic_pointer_cast<beavy::BooleanBEAVYWire>(w); });
  if (role_ == Role::garbler) {
    auto gate = std::make_unique<BooleanBEAVYToYaoGateGarbler>(gate_id, *this, std::move(input));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  } else {
    auto gate = std::make_unique<BooleanBEAVYToYaoGateEvaluator>(gate_id, *this, std::move(input));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  }
  return output;
}

template <typename T>
WireVector YaoProvider::basic_make_convert_to_arithmetic_beavy_gate(YaoWireVector&& in_a) {
  auto num_wires = in_a.size();
  assert(num_wires == ENCRYPTO::bit_size_v<T>);
  auto conv_gate_id = gate_register_.get_next_gate_id();
  auto input_gate_id = gate_register_.get_next_gate_id();
  auto num_simd = in_a[0]->get_num_simd();
  YaoWireVector in_b;
  YaoToArithmeticBEAVYGateEvaluator<T>* y2a_gate_evaluator = nullptr;
  beavy::ArithmeticBEAVYWireP<T> output;
  if (role_ == Role::garbler) {
    auto conv_gate =
        std::make_unique<YaoToArithmeticBEAVYGateGarbler<T>>(conv_gate_id, *this, num_simd);
    output = conv_gate->get_output_wire();
    auto input_gate = std::make_unique<SetupGate<YaoInputGateGarbler>>(
        input_gate_id, *this, num_wires, num_simd, conv_gate->get_mask_future());
    in_b = input_gate->get_output_wires();
    gate_register_.register_gate(std::move(conv_gate));
    gate_register_.register_gate(std::move(input_gate));
  } else {
    auto conv_gate =
        std::make_unique<YaoToArithmeticBEAVYGateEvaluator<T>>(conv_gate_id, *this, num_simd);
    output = conv_gate->get_output_wire();
    y2a_gate_evaluator = conv_gate.get();
    auto input_gate = std::make_unique<SetupGate<YaoInputGateEvaluator>>(input_gate_id, *this,
                                                                         num_wires, num_simd);
    in_b = input_gate->get_output_wires();
    gate_register_.register_gate(std::move(conv_gate));
    gate_register_.register_gate(std::move(input_gate));
  }

  // sum up the additive shares in a Boolean circuit
  const auto circuit_name = fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>);
  const auto& algo = circuit_loader_.load_circuit(circuit_name, CircuitFormat::Bristol);
  auto sum_wires = make_circuit(*this, algo, cast_wires(in_a), cast_wires(in_b));

  // the output is given to the evaluator
  if (role_ == Role::garbler) {
    make_boolean_output_gate_other(1, sum_wires);
  } else {
    auto future = make_boolean_output_gate_my(1, sum_wires);
    y2a_gate_evaluator->set_masked_value_future(std::move(future));
  }
  return {std::dynamic_pointer_cast<NewWire>(output)};
}

WireVector YaoProvider::make_convert_to_arithmetic_beavy_gate(YaoWireVector&& in_a) {
  auto bit_size = in_a.size();
  switch (bit_size) {
    case 8:
      return basic_make_convert_to_arithmetic_beavy_gate<std::uint8_t>(std::move(in_a));
    case 16:
      return basic_make_convert_to_arithmetic_beavy_gate<std::uint16_t>(std::move(in_a));
    case 32:
      return basic_make_convert_to_arithmetic_beavy_gate<std::uint32_t>(std::move(in_a));
    case 64:
      return basic_make_convert_to_arithmetic_beavy_gate<std::uint64_t>(std::move(in_a));
    default:
      throw std::logic_error(fmt::format(
          "unsupported bit size {} for Yao to Arithmetic BEAVY conversion\n", bit_size));
  }
}

template <typename T>
WireVector YaoProvider::basic_make_convert_from_arithmetic_beavy_gate(const WireVector& in) {
  assert(in.size() == 1);
  auto num_wires = ENCRYPTO::bit_size_v<T>;
  auto arith_wire = std::dynamic_pointer_cast<beavy::ArithmeticBEAVYWire<T>>(in.at(0));
  assert(arith_wire != nullptr);
  auto num_simd = arith_wire->get_num_simd();
  auto share_output_gate_id = gate_register_.get_next_gate_id();
  auto share_output_gate = std::make_unique<beavy::ArithmeticBEAVYOutputShareGate<T>>(
      share_output_gate_id, std::move(arith_wire));
  auto secret_share_future = share_output_gate->get_secret_share_future();
  auto public_share_future = share_output_gate->get_public_share_future();
  gate_register_.register_gate(std::move(share_output_gate));  // XXX: move
  auto input_gate_0_id = gate_register_.get_next_gate_id();
  auto input_gate_1_id = gate_register_.get_next_gate_id();
  YaoWireVector wires_0;
  YaoWireVector wires_1;
  if (role_ == Role::garbler) {
    auto input_gate_0 = std::make_unique<ArithmeticInputAdapterGate<YaoInputGateGarbler, T>>(
        std::move(public_share_future), std::move(secret_share_future),
        [](auto x, auto y) { return x - y; }, logger_, input_gate_0_id, *this, num_wires, num_simd);
    auto input_gate_1 = std::make_unique<SetupGate<YaoInputGateGarbler>>(input_gate_1_id, *this,
                                                                         num_wires, num_simd, true);
    wires_0 = input_gate_0->get_output_wires();
    wires_1 = input_gate_1->get_output_wires();
    gate_register_.register_gate(std::move(input_gate_0));
    gate_register_.register_gate(std::move(input_gate_1));
  } else {
    auto input_gate_0 =
        std::make_unique<YaoInputGateEvaluator>(input_gate_0_id, *this, num_wires, num_simd);
    auto input_gate_1 =
        std::make_unique<SetupGate<ArithmeticInputAdapterGate<YaoInputGateEvaluator, T>>>(
            std::move(secret_share_future), std::negate{}, true, logger_, input_gate_1_id, *this,
            num_wires, num_simd, true);
    wires_0 = input_gate_0->get_output_wires();
    wires_1 = input_gate_1->get_output_wires();
    gate_register_.register_gate(std::move(input_gate_0));
    gate_register_.register_gate(std::move(input_gate_1));
  }

  // sum up the additive shares in a Boolean circuit
  const auto circuit_name = fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>);
  const auto& algo = circuit_loader_.load_circuit(circuit_name, CircuitFormat::Bristol);
  auto sum_wires = make_circuit(*this, algo, cast_wires(wires_0), cast_wires(wires_1));
  return sum_wires;
}

WireVector YaoProvider::make_convert_from_arithmetic_beavy_gate(const WireVector& in) {
  assert(in.size() == 1);
  auto bit_size = in.at(0)->get_bit_size();
  switch (bit_size) {
    case 8:
      return basic_make_convert_from_arithmetic_beavy_gate<std::uint8_t>(in);
    case 16:
      return basic_make_convert_from_arithmetic_beavy_gate<std::uint16_t>(in);
    case 32:
      return basic_make_convert_from_arithmetic_beavy_gate<std::uint32_t>(in);
    case 64:
      return basic_make_convert_from_arithmetic_beavy_gate<std::uint64_t>(in);
    default:
      throw std::logic_error(fmt::format(
          "unsupported bit size {} for Yao from Arithmetic BEAVY conversion\n", bit_size));
  }
}

WireVector YaoProvider::convert_from_yao(MPCProtocol dst_proto, const WireVector& in) {
  auto input = cast_wires(in);

  switch (dst_proto) {
    case MPCProtocol::ArithmeticBEAVY:
      return make_convert_to_arithmetic_beavy_gate(std::move(input));
    case MPCProtocol::ArithmeticGMW:
      return make_convert_to_arithmetic_gmw_gate(std::move(input));
    case MPCProtocol::BooleanBEAVY:
      return make_convert_to_boolean_beavy_gate(std::move(input));
    case MPCProtocol::BooleanGMW:
      return make_convert_to_boolean_gmw_gate(std::move(input));
    default:
      throw std::logic_error(
          fmt::format("Yao does not support conversion from Yao to {}", ToString(dst_proto)));
  }
}

WireVector YaoProvider::convert_from_other_to_yao(MPCProtocol src_proto, const WireVector& in) {
  switch (src_proto) {
    case MPCProtocol::ArithmeticBEAVY:
      return make_convert_from_arithmetic_beavy_gate(in);
    case MPCProtocol::ArithmeticGMW:
      return make_convert_from_arithmetic_gmw_gate(in);
    case MPCProtocol::BooleanBEAVY:
      return cast_wires(make_convert_from_boolean_beavy_gate(in));
    case MPCProtocol::BooleanGMW:
      return cast_wires(make_convert_from_boolean_gmw_gate(in));
    default:
      throw std::logic_error(
          fmt::format("Yao does not support conversion from {} to Yao", ToString(src_proto)));
  }
}

WireVector YaoProvider::convert(MPCProtocol dst_proto, const WireVector& in) {
  if (in.empty()) {
    throw std::logic_error("empty WireVector");
  }
  const auto src_proto = in[0]->get_protocol();
  if (src_proto == MPCProtocol::Yao) {
    return convert_from_yao(dst_proto, in);
  } else {
    return convert_from_other_to_yao(src_proto, in);
  }
}

template <typename T>
tensor::TensorCP YaoProvider::basic_make_convert_from_arithmetic_gmw_tensor(
    const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const gmw::ArithmeticGMWTensor<T>>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  if (role_ == Role::garbler) {
    auto tensor_op = std::make_unique<ArithmeticGMWToYaoTensorConversionGarbler<T>>(gate_id, *this,
                                                                                    input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  } else {
    auto tensor_op = std::make_unique<ArithmeticGMWToYaoTensorConversionEvaluator<T>>(
        gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  }
  return output;
}

template tensor::TensorCP YaoProvider::basic_make_convert_from_arithmetic_gmw_tensor<std::uint64_t>(
    const tensor::TensorCP);

tensor::TensorCP YaoProvider::make_convert_from_arithmetic_gmw_tensor(const tensor::TensorCP in) {
  switch (in->get_bit_size()) {
    case 32: {
      return basic_make_convert_from_arithmetic_gmw_tensor<std::uint32_t>(std::move(in));
      break;
    }
    case 64: {
      return basic_make_convert_from_arithmetic_gmw_tensor<std::uint64_t>(std::move(in));
      break;
    }
    default: {
      throw std::logic_error("unsupprted bit size");
    }
  }
}

template <typename T>
tensor::TensorCP YaoProvider::basic_make_convert_to_arithmetic_gmw_tensor(
    const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const YaoTensor>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  if (role_ == Role::garbler) {
    auto tensor_op = std::make_unique<YaoToArithmeticGMWTensorConversionGarbler<T>>(gate_id, *this,
                                                                                    input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  } else {
    auto tensor_op = std::make_unique<YaoToArithmeticGMWTensorConversionEvaluator<T>>(
        gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  }
  return output;
}

template tensor::TensorCP YaoProvider::basic_make_convert_to_arithmetic_gmw_tensor<std::uint64_t>(
    const tensor::TensorCP);

tensor::TensorCP YaoProvider::make_convert_to_arithmetic_gmw_tensor(const tensor::TensorCP in) {
  switch (in->get_bit_size()) {
    case 32: {
      return basic_make_convert_to_arithmetic_gmw_tensor<std::uint32_t>(std::move(in));
      break;
    }
    case 64: {
      return basic_make_convert_to_arithmetic_gmw_tensor<std::uint64_t>(std::move(in));
      break;
    }
    default: {
      throw std::logic_error("unsupprted bit size");
    }
  }
}

tensor::TensorCP YaoProvider::make_convert_to_boolean_gmw_tensor(const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const YaoTensor>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  if (role_ == Role::garbler) {
    auto tensor_op =
        std::make_unique<YaoToBooleanGMWTensorConversionGarbler>(gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  } else {
    auto tensor_op =
        std::make_unique<YaoToBooleanGMWTensorConversionEvaluator>(gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  }
  return output;
}

template <typename T>
tensor::TensorCP YaoProvider::basic_make_convert_from_arithmetic_beavy_tensor(
    const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const beavy::ArithmeticBEAVYTensor<T>>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  if (role_ == Role::garbler) {
    auto tensor_op = std::make_unique<ArithmeticBEAVYToYaoTensorConversionGarbler<T>>(
        gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  } else {
    auto tensor_op = std::make_unique<ArithmeticBEAVYToYaoTensorConversionEvaluator<T>>(
        gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  }
  return output;
}

template tensor::TensorCP
YaoProvider::basic_make_convert_from_arithmetic_beavy_tensor<std::uint64_t>(const tensor::TensorCP);

tensor::TensorCP YaoProvider::make_convert_from_arithmetic_beavy_tensor(const tensor::TensorCP in) {
  switch (in->get_bit_size()) {
    case 32: {
      return basic_make_convert_from_arithmetic_beavy_tensor<std::uint32_t>(std::move(in));
      break;
    }
    case 64: {
      return basic_make_convert_from_arithmetic_beavy_tensor<std::uint64_t>(std::move(in));
      break;
    }
    default: {
      throw std::logic_error("unsupprted bit size");
    }
  }
}

template <typename T>
tensor::TensorCP YaoProvider::basic_make_convert_to_arithmetic_beavy_tensor(
    const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const YaoTensor>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  if (role_ == Role::garbler) {
    auto tensor_op = std::make_unique<YaoToArithmeticBEAVYTensorConversionGarbler<T>>(
        gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  } else {
    auto tensor_op = std::make_unique<YaoToArithmeticBEAVYTensorConversionEvaluator<T>>(
        gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  }
  return output;
}

template tensor::TensorCP YaoProvider::basic_make_convert_to_arithmetic_beavy_tensor<std::uint64_t>(
    const tensor::TensorCP);

tensor::TensorCP YaoProvider::make_convert_to_arithmetic_beavy_tensor(const tensor::TensorCP in) {
  switch (in->get_bit_size()) {
    case 32: {
      return basic_make_convert_to_arithmetic_beavy_tensor<std::uint32_t>(std::move(in));
      break;
    }
    case 64: {
      return basic_make_convert_to_arithmetic_beavy_tensor<std::uint64_t>(std::move(in));
      break;
    }
    default: {
      throw std::logic_error("unsupprted bit size");
    }
  }
}

tensor::TensorCP YaoProvider::make_convert_to_boolean_beavy_tensor(const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const YaoTensor>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  if (role_ == Role::garbler) {
    auto tensor_op =
        std::make_unique<YaoToBooleanBEAVYTensorConversionGarbler>(gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  } else {
    auto tensor_op =
        std::make_unique<YaoToBooleanBEAVYTensorConversionEvaluator>(gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  }
  return output;
}

tensor::TensorCP YaoProvider::make_tensor_conversion(MPCProtocol proto_to,
                                                     const tensor::TensorCP in) {
  auto proto_from = in->get_protocol();
  if (proto_from == MPCProtocol::Yao) {
    if (proto_to == MPCProtocol::ArithmeticBEAVY) {
      return make_convert_to_arithmetic_beavy_tensor(in);
    } else if (proto_to == MPCProtocol::ArithmeticGMW) {
      return make_convert_to_arithmetic_gmw_tensor(in);
    } else if (proto_to == MPCProtocol::BooleanBEAVY) {
      return make_convert_to_boolean_beavy_tensor(in);
    } else if (proto_to == MPCProtocol::BooleanGMW) {
      return make_convert_to_boolean_gmw_tensor(in);
    }
  } else if (proto_to == MPCProtocol::Yao) {
    if (proto_from == MPCProtocol::ArithmeticBEAVY) {
      return make_convert_from_arithmetic_beavy_tensor(in);
    } else if (proto_from == MPCProtocol::ArithmeticGMW) {
      return make_convert_from_arithmetic_gmw_tensor(in);
    }
  }
  throw std::logic_error(
      fmt::format("YaoProvider does not support tensor conversions from {} to {}",
                  ToString(proto_from), ToString(proto_to)));
}

tensor::TensorCP YaoProvider::make_tensor_relu_op(const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const YaoTensor>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  if (role_ == Role::garbler) {
    auto tensor_op = std::make_unique<YaoTensorReluGarbler>(gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  } else {
    auto tensor_op = std::make_unique<YaoTensorReluEvaluator>(gate_id, *this, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  }
  return output;
}

tensor::TensorCP YaoProvider::make_tensor_maxpool_op(const tensor::MaxPoolOp& maxpool_op,
                                                     const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const YaoTensor>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  if (role_ == Role::garbler) {
    auto tensor_op =
        std::make_unique<YaoTensorMaxPoolGarbler>(gate_id, *this, maxpool_op, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  } else {
    auto tensor_op =
        std::make_unique<YaoTensorMaxPoolEvaluator>(gate_id, *this, maxpool_op, input_tensor);
    output = tensor_op->get_output_tensor();
    gate_register_.register_gate(std::move(tensor_op));
  }
  return output;
}

}  // namespace MOTION::proto::yao
