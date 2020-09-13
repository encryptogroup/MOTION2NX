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

#include "beavy_provider.h"

#include <cstdint>
#include <unordered_map>

#include "base/gate_register.h"
#include "communication/communication_layer.h"
#include "communication/fbs_headers/gmw_message_generated.h"
#include "communication/message.h"
#include "communication/message_handler.h"
#include "conversion.h"
#include "crypto/motion_base_provider.h"
#include "gate.h"
#include "plain.h"
#include "protocols/gmw/wire.h"
#include "protocols/plain/wire.h"
#include "tensor_op.h"
#include "utility/constants.h"
#include "utility/logger.h"
#include "utility/meta.hpp"
#include "wire.h"

namespace MOTION::proto::beavy {

BEAVYProvider::BEAVYProvider(Communication::CommunicationLayer& communication_layer,
                             GateRegister& gate_register, CircuitLoader& circuit_loader,
                             Crypto::MotionBaseProvider& motion_base_provider,
                             ENCRYPTO::ObliviousTransfer::OTProviderManager& ot_manager,
                             ArithmeticProviderManager& arith_manager,
                             std::shared_ptr<Logger> logger, bool fake_setup)
    : CommMixin(communication_layer, Communication::MessageType::BEAVYGate, logger),
      communication_layer_(communication_layer),
      gate_register_(gate_register),
      circuit_loader_(circuit_loader),
      motion_base_provider_(motion_base_provider),
      ot_manager_(ot_manager),
      arith_manager_(arith_manager),
      my_id_(communication_layer_.get_my_id()),
      num_parties_(communication_layer_.get_num_parties()),
      next_input_id_(0),
      logger_(std::move(logger)),
      fake_setup_(fake_setup) {
  if (communication_layer.get_num_parties() != 2) {
    throw std::logic_error("currently only two parties are supported");
  }
}

BEAVYProvider::~BEAVYProvider() = default;

void BEAVYProvider::setup() {
  motion_base_provider_.wait_setup();
  // TODO wait for ot setup
  set_setup_ready();
}

bool BEAVYProvider::is_my_job(std::size_t gate_id) const noexcept {
  return my_id_ == (gate_id % num_parties_);
}

std::size_t BEAVYProvider::get_next_input_id(std::size_t num_inputs) noexcept {
  auto next_id = next_input_id_;
  next_input_id_ += num_inputs;
  return next_id;
}

static BooleanBEAVYWireVector cast_wires(std::vector<std::shared_ptr<NewWire>> wires) {
  BooleanBEAVYWireVector result(wires.size());
  std::transform(std::begin(wires), std::end(wires), std::begin(result),
                 [](auto& w) { return std::dynamic_pointer_cast<BooleanBEAVYWire>(w); });
  return result;
}

static plain::BooleanPlainWireVector cast_to_plain_wires(
    std::vector<std::shared_ptr<NewWire>> wires) {
  plain::BooleanPlainWireVector result(wires.size());
  std::transform(std::begin(wires), std::end(wires), std::begin(result), [](auto& w) {
    return std::dynamic_pointer_cast<proto::plain::BooleanPlainWire>(w);
  });
  return result;
}

static std::vector<std::shared_ptr<NewWire>> cast_wires(BooleanBEAVYWireVector&& wires) {
  return std::vector<std::shared_ptr<NewWire>>(std::begin(wires), std::end(wires));
}

template <typename T>
static ArithmeticBEAVYWireP<T> cast_arith_wire(std::shared_ptr<NewWire> wire) {
  auto ptr = std::dynamic_pointer_cast<ArithmeticBEAVYWire<T>>(wire);
  assert(ptr);
  return ptr;
}

template <typename T>
static plain::ArithmeticPlainWireP<T> cast_arith_plain_wire(std::shared_ptr<NewWire> wire) {
  auto ptr = std::dynamic_pointer_cast<proto::plain::ArithmeticPlainWire<T>>(wire);
  assert(ptr);
  return ptr;
}

template <typename T>
static std::shared_ptr<NewWire> cast_arith_wire(ArithmeticBEAVYWireP<T> wire) {
  return std::shared_ptr<NewWire>(wire);
}

// Boolean inputs/outputs

std::pair<ENCRYPTO::ReusableFiberPromise<BitValues>, WireVector>
BEAVYProvider::make_boolean_input_gate_my(std::size_t input_owner, std::size_t num_wires,
                                          std::size_t num_simd) {
  if (input_owner != my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  BooleanBEAVYWireVector output;
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> promise;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanBEAVYInputGateSender>(gate_id, *this, num_wires, num_simd,
                                                            promise.get_future());
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return {std::move(promise), cast_wires(std::move(output))};
}

WireVector BEAVYProvider::make_boolean_input_gate_other(std::size_t input_owner,
                                                        std::size_t num_wires,
                                                        std::size_t num_simd) {
  if (input_owner == my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  BooleanBEAVYWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanBEAVYInputGateReceiver>(gate_id, *this, num_wires, num_simd,
                                                              input_owner);
  output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return cast_wires(std::move(output));
}

ENCRYPTO::ReusableFiberFuture<BitValues> BEAVYProvider::make_boolean_output_gate_my(
    std::size_t output_owner, const WireVector& in) {
  if (output_owner != ALL_PARTIES && output_owner != my_id_) {
    throw std::logic_error("trying to create output gate for wrong party");
  }
  auto gate_id = gate_register_.get_next_gate_id();
  auto input = cast_wires(in);
  auto gate =
      std::make_unique<BooleanBEAVYOutputGate>(gate_id, *this, std::move(input), output_owner);
  auto future = gate->get_output_future();
  gate_register_.register_gate(std::move(gate));
  return future;
}

void BEAVYProvider::make_boolean_output_gate_other(std::size_t output_owner, const WireVector& in) {
  if (output_owner == ALL_PARTIES || output_owner == my_id_) {
    throw std::logic_error("trying to create output gate for wrong party");
  }
  auto gate_id = gate_register_.get_next_gate_id();
  auto input = cast_wires(in);
  auto gate =
      std::make_unique<BooleanBEAVYOutputGate>(gate_id, *this, std::move(input), output_owner);
  gate_register_.register_gate(std::move(gate));
}

// arithmetic inputs/outputs

template <typename T>
std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<T>>, WireVector>
BEAVYProvider::basic_make_arithmetic_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  if (input_owner != my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  ENCRYPTO::ReusableFiberPromise<std::vector<T>> promise;
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<ArithmeticBEAVYInputGateSender<T>>(gate_id, *this, num_simd,
                                                                  promise.get_future());
  auto output = gate->get_output_wire();
  gate_register_.register_gate(std::move(gate));
  return {std::move(promise), {cast_arith_wire(std::move(output))}};
}

std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint8_t>>, WireVector>
BEAVYProvider::make_arithmetic_8_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_my<std::uint8_t>(input_owner, num_simd);
}
std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint16_t>>, WireVector>
BEAVYProvider::make_arithmetic_16_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_my<std::uint16_t>(input_owner, num_simd);
}
std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint32_t>>, WireVector>
BEAVYProvider::make_arithmetic_32_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_my<std::uint32_t>(input_owner, num_simd);
}
std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint64_t>>, WireVector>
BEAVYProvider::make_arithmetic_64_input_gate_my(std::size_t input_owner, std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_my<std::uint64_t>(input_owner, num_simd);
}

template <typename T>
WireVector BEAVYProvider::basic_make_arithmetic_input_gate_other(std::size_t input_owner,
                                                                 std::size_t num_simd) {
  if (input_owner == my_id_) {
    throw std::logic_error("trying to create input gate for wrong party");
  }
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate =
      std::make_unique<ArithmeticBEAVYInputGateReceiver<T>>(gate_id, *this, num_simd, input_owner);
  auto output = gate->get_output_wire();
  gate_register_.register_gate(std::move(gate));
  return {cast_arith_wire(std::move(output))};
}

WireVector BEAVYProvider::make_arithmetic_8_input_gate_other(std::size_t input_owner,
                                                             std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_other<std::uint8_t>(input_owner, num_simd);
}
WireVector BEAVYProvider::make_arithmetic_16_input_gate_other(std::size_t input_owner,
                                                              std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_other<std::uint16_t>(input_owner, num_simd);
}
WireVector BEAVYProvider::make_arithmetic_32_input_gate_other(std::size_t input_owner,
                                                              std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_other<std::uint32_t>(input_owner, num_simd);
}
WireVector BEAVYProvider::make_arithmetic_64_input_gate_other(std::size_t input_owner,
                                                              std::size_t num_simd) {
  return basic_make_arithmetic_input_gate_other<std::uint64_t>(input_owner, num_simd);
}

template <typename T>
ENCRYPTO::ReusableFiberFuture<IntegerValues<T>> BEAVYProvider::basic_make_arithmetic_output_gate_my(
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
  auto gate = std::make_unique<ArithmeticBEAVYOutputGate<T>>(gate_id, *this, std::move(input),
                                                             output_owner);
  auto future = gate->get_output_future();
  gate_register_.register_gate(std::move(gate));
  return future;
}

ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint8_t>>
BEAVYProvider::make_arithmetic_8_output_gate_my(std::size_t output_owner, const WireVector& in) {
  return basic_make_arithmetic_output_gate_my<std::uint8_t>(output_owner, in);
}
ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint16_t>>
BEAVYProvider::make_arithmetic_16_output_gate_my(std::size_t output_owner, const WireVector& in) {
  return basic_make_arithmetic_output_gate_my<std::uint16_t>(output_owner, in);
}
ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint32_t>>
BEAVYProvider::make_arithmetic_32_output_gate_my(std::size_t output_owner, const WireVector& in) {
  return basic_make_arithmetic_output_gate_my<std::uint32_t>(output_owner, in);
}
ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint64_t>>
BEAVYProvider::make_arithmetic_64_output_gate_my(std::size_t output_owner, const WireVector& in) {
  return basic_make_arithmetic_output_gate_my<std::uint64_t>(output_owner, in);
}

void BEAVYProvider::make_arithmetic_output_gate_other(std::size_t output_owner,
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
      gate = std::make_unique<ArithmeticBEAVYOutputGate<std::uint8_t>>(
          gate_id, *this, cast_arith_wire<std::uint8_t>(in[0]), output_owner);
      break;
    }
    case 16: {
      gate = std::make_unique<ArithmeticBEAVYOutputGate<std::uint16_t>>(
          gate_id, *this, cast_arith_wire<std::uint16_t>(in[0]), output_owner);
      break;
    }
    case 32: {
      gate = std::make_unique<ArithmeticBEAVYOutputGate<std::uint32_t>>(
          gate_id, *this, cast_arith_wire<std::uint32_t>(in[0]), output_owner);
      break;
    }
    case 64: {
      gate = std::make_unique<ArithmeticBEAVYOutputGate<std::uint64_t>>(
          gate_id, *this, cast_arith_wire<std::uint64_t>(in[0]), output_owner);
      break;
    }
    default: {
      throw std::logic_error("unsupprted bit size");
    }
  }
  gate_register_.register_gate(std::move(gate));
}

std::pair<NewGateP, WireVector> BEAVYProvider::construct_unary_gate(
    ENCRYPTO::PrimitiveOperationType op, const WireVector& in_a) {
  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::INV:
      return construct_inv_gate(in_a);
    default:
      throw std::logic_error(fmt::format("BEAVY does not support the unary operation {}", op));
  }
}

std::vector<std::shared_ptr<NewWire>> BEAVYProvider::make_unary_gate(
    ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>& in_a) {
  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::INV:
      return make_inv_gate(in_a);
    case ENCRYPTO::PrimitiveOperationType::NEG:
      return make_neg_gate(in_a);
    case ENCRYPTO::PrimitiveOperationType::SQR:
      return make_sqr_gate(in_a);
    default:
      throw std::logic_error(fmt::format("BEAVY does not support the unary operation {}", op));
  }
}

std::pair<NewGateP, WireVector> BEAVYProvider::construct_binary_gate(
    ENCRYPTO::PrimitiveOperationType op, const WireVector& in_a, const WireVector& in_b) {
  switch (op) {
    case ENCRYPTO::PrimitiveOperationType::XOR:
      return construct_xor_gate(in_a, in_b);
    case ENCRYPTO::PrimitiveOperationType::AND:
      return construct_and_gate(in_a, in_b);
    default:
      throw std::logic_error(fmt::format("BEAVY does not support the binary operation {}", op));
  }
}

std::vector<std::shared_ptr<NewWire>> BEAVYProvider::make_binary_gate(
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
      throw std::logic_error(fmt::format("BEAVY does not support the binary operation {}", op));
  }
}

std::pair<std::unique_ptr<NewGate>, WireVector> BEAVYProvider::construct_inv_gate(
    const WireVector& in_a) {
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanBEAVYINVGate>(gate_id, *this, cast_wires(in_a));
  auto output = gate->get_output_wires();
  return {std::move(gate), cast_wires(std::move(output))};
}

WireVector BEAVYProvider::make_inv_gate(const WireVector& in_a) {
  auto [gate, output] = construct_inv_gate(in_a);
  gate_register_.register_gate(std::move(gate));
  return output;
}

template <typename BinaryGate, bool plain>
std::pair<NewGateP, WireVector> BEAVYProvider::construct_boolean_binary_gate(
    const WireVector& in_a, const WireVector& in_b) {
  BooleanBEAVYWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  if constexpr (plain) {
    auto gate =
        std::make_unique<BinaryGate>(gate_id, *this, cast_wires(in_a), cast_to_plain_wires(in_b));
    output = gate->get_output_wires();
    return {std::move(gate), cast_wires(std::move(output))};
  } else {
    auto gate = std::make_unique<BinaryGate>(gate_id, *this, cast_wires(in_a), cast_wires(in_b));
    output = gate->get_output_wires();
    return {std::move(gate), cast_wires(std::move(output))};
  }
}

template <typename BinaryGate, bool plain>
WireVector BEAVYProvider::make_boolean_binary_gate(const WireVector& in_a, const WireVector& in_b) {
  auto [gate, out] = construct_boolean_binary_gate<BinaryGate, plain>(in_a, in_b);
  gate_register_.register_gate(std::move(gate));
  return out;
}

std::pair<NewGateP, WireVector> BEAVYProvider::construct_xor_gate(const WireVector& in_a,
                                                                  const WireVector& in_b) {
  // assume, at most one of the inputs is a plain wire
  if (in_a.at(0)->get_protocol() == MPCProtocol::BooleanPlain) {
    return construct_xor_gate(in_b, in_a);
  }
  assert(in_a.at(0)->get_protocol() == MPCProtocol::BooleanBEAVY);
  if (in_b.at(0)->get_protocol() == MPCProtocol::BooleanPlain) {
    return construct_boolean_binary_gate<BooleanBEAVYXORPlainGate, true>(in_a, in_b);
  } else {
    return construct_boolean_binary_gate<BooleanBEAVYXORGate>(in_a, in_b);
  }
}

std::pair<NewGateP, WireVector> BEAVYProvider::construct_and_gate(const WireVector& in_a,
                                                                  const WireVector& in_b) {
  // assume, at most one of the inputs is a plain wire
  if (in_a.at(0)->get_protocol() == MPCProtocol::BooleanPlain) {
    return construct_xor_gate(in_b, in_a);
  }
  assert(in_a.at(0)->get_protocol() == MPCProtocol::BooleanBEAVY);
  if (in_b.at(0)->get_protocol() == MPCProtocol::BooleanPlain) {
    return construct_boolean_binary_gate<BooleanBEAVYANDPlainGate, true>(in_a, in_b);
  } else {
    return construct_boolean_binary_gate<BooleanBEAVYANDGate>(in_a, in_b);
  }
}

WireVector BEAVYProvider::make_xor_gate(const WireVector& in_a, const WireVector& in_b) {
  // assume, at most one of the inputs is a plain wire
  if (in_a.at(0)->get_protocol() == MPCProtocol::BooleanPlain) {
    return make_xor_gate(in_b, in_a);
  }
  assert(in_a.at(0)->get_protocol() == MPCProtocol::BooleanBEAVY);
  if (in_b.at(0)->get_protocol() == MPCProtocol::BooleanPlain) {
    return make_boolean_binary_gate<BooleanBEAVYXORPlainGate, true>(in_a, in_b);
  } else {
    return make_boolean_binary_gate<BooleanBEAVYXORGate>(in_a, in_b);
  }
}

WireVector BEAVYProvider::make_and_gate(const WireVector& in_a, const WireVector& in_b) {
  // assume, at most one of the inputs is a plain wire
  if (in_a.at(0)->get_protocol() == MPCProtocol::BooleanPlain) {
    return make_xor_gate(in_b, in_a);
  }
  assert(in_a.at(0)->get_protocol() == MPCProtocol::BooleanBEAVY);
  if (in_b.at(0)->get_protocol() == MPCProtocol::BooleanPlain) {
    return make_boolean_binary_gate<BooleanBEAVYANDPlainGate, true>(in_a, in_b);
  } else {
    return make_boolean_binary_gate<BooleanBEAVYANDGate>(in_a, in_b);
  }
}

static std::size_t check_arithmetic_wire(const WireVector& in) {
  if (in.size() != 1) {
    throw std::logic_error("arithmetic operations support single wires only");
  }
  return in[0]->get_bit_size();
}

template <template <typename> class UnaryGate, typename T>
WireVector BEAVYProvider::make_arithmetic_unary_gate(const NewWireP& in_a) {
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<UnaryGate<T>>(gate_id, *this, cast_arith_wire<T>(in_a));
  auto output = {cast_arith_wire(gate->get_output_wire())};
  gate_register_.register_gate(std::move(gate));
  return output;
}

template <template <typename> class UnaryGate>
WireVector BEAVYProvider::make_arithmetic_unary_gate(const WireVector& in_a) {
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

template <template <typename> class BinaryGate, typename T, bool plain>
WireVector BEAVYProvider::make_arithmetic_binary_gate(const NewWireP& in_a, const NewWireP& in_b) {
  auto gate_id = gate_register_.get_next_gate_id();
  WireVector output;
  if constexpr (plain) {
    auto gate = std::make_unique<BinaryGate<T>>(gate_id, *this, cast_arith_wire<T>(in_a),
                                                cast_arith_plain_wire<T>(in_b));
    output = {cast_arith_wire(gate->get_output_wire())};
    gate_register_.register_gate(std::move(gate));
  } else {
    auto gate = std::make_unique<BinaryGate<T>>(gate_id, *this, cast_arith_wire<T>(in_a),
                                                cast_arith_wire<T>(in_b));
    output = {cast_arith_wire(gate->get_output_wire())};
    gate_register_.register_gate(std::move(gate));
  }
  return output;
}

template <template <typename> class BinaryGate, bool plain>
WireVector BEAVYProvider::make_arithmetic_binary_gate(const WireVector& in_a,
                                                      const WireVector& in_b) {
  auto bit_size = check_arithmetic_wires(in_a, in_b);
  switch (bit_size) {
    case 8:
      return make_arithmetic_binary_gate<BinaryGate, std::uint8_t, plain>(in_a[0], in_b[0]);
    case 16:
      return make_arithmetic_binary_gate<BinaryGate, std::uint16_t, plain>(in_a[0], in_b[0]);
    case 32:
      return make_arithmetic_binary_gate<BinaryGate, std::uint32_t, plain>(in_a[0], in_b[0]);
    case 64:
      return make_arithmetic_binary_gate<BinaryGate, std::uint64_t, plain>(in_a[0], in_b[0]);
    default:
      throw std::logic_error(fmt::format("unexpected bit size {}", bit_size));
  }
}

WireVector BEAVYProvider::make_neg_gate(const WireVector& in) {
  return make_arithmetic_unary_gate<ArithmeticBEAVYNEGGate>(in);
}

WireVector BEAVYProvider::make_add_gate(const WireVector& in_a, const WireVector& in_b) {
  // assume, at most one of the inputs is a plain wire
  if (in_a.at(0)->get_protocol() == MPCProtocol::ArithmeticPlain) {
    return make_add_gate(in_b, in_a);
  }
  assert(in_a.at(0)->get_protocol() == MPCProtocol::ArithmeticBEAVY);
  if (in_b.at(0)->get_protocol() == MPCProtocol::ArithmeticPlain) {
    return make_arithmetic_binary_gate<ArithmeticBEAVYADDPlainGate, true>(in_a, in_b);
  } else {
    return make_arithmetic_binary_gate<ArithmeticBEAVYADDGate>(in_a, in_b);
  }
}

WireVector BEAVYProvider::make_mul_gate(const WireVector& in_a, const WireVector& in_b) {
  // assume, at most one of the inputs is a plain wire
  if (in_a.at(0)->get_protocol() == MPCProtocol::ArithmeticPlain) {
    return make_mul_gate(in_b, in_a);
  }
  assert(in_a.at(0)->get_protocol() == MPCProtocol::ArithmeticBEAVY);
  if (in_b.at(0)->get_protocol() == MPCProtocol::ArithmeticPlain) {
    return make_arithmetic_binary_gate<ArithmeticBEAVYMULPlainGate, true>(in_a, in_b);
  } else {
    return make_arithmetic_binary_gate<ArithmeticBEAVYMULGate>(in_a, in_b);
  }
}

WireVector BEAVYProvider::make_sqr_gate(const WireVector& in) {
  return make_arithmetic_unary_gate<ArithmeticBEAVYSQRGate>(in);
}

template <typename T>
WireVector BEAVYProvider::basic_make_convert_to_arithmetic_beavy_gate(
    BooleanBEAVYWireVector&& in_a) {
  [[maybe_unused]] auto num_wires = in_a.size();
  assert(num_wires == ENCRYPTO::bit_size_v<T>);
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanToArithmeticBEAVYGate<T>>(gate_id, *this, std::move(in_a));
  auto output = gate->get_output_wire();
  gate_register_.register_gate(std::move(gate));
  return {std::dynamic_pointer_cast<NewWire>(output)};
}

WireVector BEAVYProvider::make_convert_to_arithmetic_beavy_gate(BooleanBEAVYWireVector&& in_a) {
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
          "unsupported bit size {} for Boolean to Arithmetic BEAVY conversion\n", bit_size));
  }
}

template <typename T>
WireVector BEAVYProvider::basic_make_convert_bit_to_arithmetic_beavy_gate(BooleanBEAVYWireP in_a) {
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanBitToArithmeticBEAVYGate<T>>(gate_id, *this, std::move(in_a));
  auto output = gate->get_output_wire();
  gate_register_.register_gate(std::move(gate));
  return {std::dynamic_pointer_cast<NewWire>(output)};
}

template WireVector BEAVYProvider::basic_make_convert_bit_to_arithmetic_beavy_gate<std::uint8_t>(
    BooleanBEAVYWireP);
template WireVector BEAVYProvider::basic_make_convert_bit_to_arithmetic_beavy_gate<std::uint16_t>(
    BooleanBEAVYWireP);
template WireVector BEAVYProvider::basic_make_convert_bit_to_arithmetic_beavy_gate<std::uint32_t>(
    BooleanBEAVYWireP);
template WireVector BEAVYProvider::basic_make_convert_bit_to_arithmetic_beavy_gate<std::uint64_t>(
    BooleanBEAVYWireP);

WireVector BEAVYProvider::make_convert_to_boolean_gmw_gate(BooleanBEAVYWireVector&& in_a) {
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<BooleanBEAVYToGMWGate>(gate_id, *this, std::move(in_a));
  auto output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return std::vector<std::shared_ptr<NewWire>>(std::begin(output), std::end(output));
}

template <typename T>
WireVector BEAVYProvider::basic_make_convert_to_arithmetic_gmw_gate(const NewWireP& in_a) {
  auto input = std::dynamic_pointer_cast<ArithmeticBEAVYWire<T>>(in_a);
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<ArithmeticBEAVYToGMWGate<T>>(gate_id, *this, input);
  auto output = gate->get_output_wire();
  gate_register_.register_gate(std::move(gate));
  return {std::dynamic_pointer_cast<NewWire>(output)};
}

WireVector BEAVYProvider::make_convert_to_arithmetic_gmw_gate(const WireVector& in_a) {
  assert(in_a.size() == 1);
  const auto& wire = in_a.at(0);
  auto bit_size = wire->get_bit_size();
  switch (bit_size) {
    case 8:
      return basic_make_convert_to_arithmetic_gmw_gate<std::uint8_t>(wire);
    case 16:
      return basic_make_convert_to_arithmetic_gmw_gate<std::uint16_t>(wire);
    case 32:
      return basic_make_convert_to_arithmetic_gmw_gate<std::uint32_t>(wire);
    case 64:
      return basic_make_convert_to_arithmetic_gmw_gate<std::uint64_t>(wire);
    default:
      throw std::logic_error(fmt::format(
          "unsupported bit size {} for Arithmetic BEAVY to GMW conversion\n", bit_size));
  }
}

WireVector BEAVYProvider::convert_from_boolean_beavy(MPCProtocol proto, const WireVector& in) {
  auto input = cast_wires(in);

  switch (proto) {
    case MPCProtocol::ArithmeticBEAVY:
      return make_convert_to_arithmetic_beavy_gate(std::move(input));
    case MPCProtocol::BooleanGMW:
      return make_convert_to_boolean_gmw_gate(std::move(input));
    default:
      throw std::logic_error(
          fmt::format("BooleanBEAVY does not support conversion to {}", ToString(proto)));
  }
}

WireVector BEAVYProvider::convert_from_arithmetic_beavy(MPCProtocol proto, const WireVector& in) {
  switch (proto) {
    case MPCProtocol::ArithmeticGMW:
      return make_convert_to_arithmetic_gmw_gate(in);
    default:
      throw std::logic_error(
          fmt::format("ArithmeticBEAVY does not support conversion to {}", ToString(proto)));
  }
}

BooleanBEAVYWireVector BEAVYProvider::make_convert_from_boolean_gmw_gate(const WireVector& in) {
  auto gate_id = gate_register_.get_next_gate_id();
  gmw::BooleanGMWWireVector input;
  input.reserve(in.size());
  std::transform(std::begin(in), std::end(in), std::back_inserter(input),
                 [](auto& w) { return std::dynamic_pointer_cast<gmw::BooleanGMWWire>(w); });
  auto gate = std::make_unique<BooleanGMWToBEAVYGate>(gate_id, *this, std::move(input));
  auto output = gate->get_output_wires();
  gate_register_.register_gate(std::move(gate));
  return output;
}

WireVector BEAVYProvider::convert_from_other_to_boolean_beavy(const WireVector& in) {
  assert(in.size() > 0);
  auto src_proto = in.at(0)->get_protocol();

  switch (src_proto) {
    case MPCProtocol::BooleanGMW:
      return cast_wires(make_convert_from_boolean_gmw_gate(in));
    default:
      throw std::logic_error(
          fmt::format("BooleanBEAVY does not support conversion from {}", ToString(src_proto)));
  }
}

template <typename T>
WireVector BEAVYProvider::basic_make_convert_from_arithmetic_gmw_gate(const NewWireP& in_a) {
  auto input = std::dynamic_pointer_cast<gmw::ArithmeticGMWWire<T>>(in_a);
  auto gate_id = gate_register_.get_next_gate_id();
  auto gate = std::make_unique<ArithmeticGMWToBEAVYGate<T>>(gate_id, *this, input);
  auto output = gate->get_output_wire();
  gate_register_.register_gate(std::move(gate));
  return {std::dynamic_pointer_cast<NewWire>(output)};
}

WireVector BEAVYProvider::make_convert_from_arithmetic_gmw_gate(const WireVector& in_a) {
  assert(in_a.size() == 1);
  const auto& wire = in_a.at(0);
  auto bit_size = wire->get_bit_size();
  switch (bit_size) {
    case 8:
      return basic_make_convert_from_arithmetic_gmw_gate<std::uint8_t>(wire);
    case 16:
      return basic_make_convert_from_arithmetic_gmw_gate<std::uint16_t>(wire);
    case 32:
      return basic_make_convert_from_arithmetic_gmw_gate<std::uint32_t>(wire);
    case 64:
      return basic_make_convert_from_arithmetic_gmw_gate<std::uint64_t>(wire);
    default:
      throw std::logic_error(fmt::format(
          "unsupported bit size {} for Arithmetic BEAVY from GMW conversion\n", bit_size));
  }
}

WireVector BEAVYProvider::convert_from_other_to_arithmetic_beavy(const WireVector& in) {
  assert(in.size() > 0);
  const auto src_proto = in.at(0)->get_protocol();
  switch (src_proto) {
    case MPCProtocol::ArithmeticGMW:
      return make_convert_from_arithmetic_gmw_gate(in);
    default:
      throw std::logic_error(
          fmt::format("ArithmeticBEAVY does not support conversion from {}", ToString(src_proto)));
  }
}

WireVector BEAVYProvider::convert(MPCProtocol dst_proto, const WireVector& in) {
  if (in.empty()) {
    throw std::logic_error("empty WireVector");
  }
  const auto src_proto = in[0]->get_protocol();
  if (src_proto == MPCProtocol::ArithmeticBEAVY) {
    return convert_from_arithmetic_beavy(dst_proto, in);
  } else if (src_proto == MPCProtocol::BooleanBEAVY) {
    return convert_from_boolean_beavy(dst_proto, in);
  } else if (dst_proto == MPCProtocol::ArithmeticBEAVY) {
    return convert_from_other_to_arithmetic_beavy(in);
  } else if (dst_proto == MPCProtocol::BooleanBEAVY) {
    return convert_from_other_to_boolean_beavy(in);
  }
  throw std::logic_error("expected conversion to or from BEAVY protocol");
}

// implementation of TensorOpFactory

template <typename T>
std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<T>>, tensor::TensorCP>
BEAVYProvider::basic_make_arithmetic_tensor_input_my(const tensor::TensorDimensions& dims) {
  ENCRYPTO::ReusableFiberPromise<std::vector<T>> promise;
  auto gate_id = gate_register_.get_next_gate_id();
  auto tensor_op = std::make_unique<ArithmeticBEAVYTensorInputSender<T>>(gate_id, *this, dims,
                                                                         promise.get_future());
  auto output = tensor_op->get_output_tensor();
  gate_register_.register_gate(std::move(tensor_op));
  return {std::move(promise), std::dynamic_pointer_cast<const tensor::Tensor>(output)};
}

template std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint64_t>>, tensor::TensorCP>
BEAVYProvider::basic_make_arithmetic_tensor_input_my(const tensor::TensorDimensions&);

template <typename T>
tensor::TensorCP BEAVYProvider::basic_make_arithmetic_tensor_input_other(
    const tensor::TensorDimensions& dims) {
  auto gate_id = gate_register_.get_next_gate_id();
  auto tensor_op = std::make_unique<ArithmeticBEAVYTensorInputReceiver<T>>(gate_id, *this, dims);
  auto output = tensor_op->get_output_tensor();
  gate_register_.register_gate(std::move(tensor_op));
  return std::dynamic_pointer_cast<const tensor::Tensor>(output);
}

template tensor::TensorCP BEAVYProvider::basic_make_arithmetic_tensor_input_other<std::uint64_t>(
    const tensor::TensorDimensions&);

std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint32_t>>, tensor::TensorCP>
BEAVYProvider::make_arithmetic_32_tensor_input_my(const tensor::TensorDimensions& dims) {
  return basic_make_arithmetic_tensor_input_my<std::uint32_t>(dims);
}

std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint64_t>>, tensor::TensorCP>
BEAVYProvider::make_arithmetic_64_tensor_input_my(const tensor::TensorDimensions& dims) {
  return basic_make_arithmetic_tensor_input_my<std::uint64_t>(dims);
}

tensor::TensorCP BEAVYProvider::make_arithmetic_32_tensor_input_other(
    const tensor::TensorDimensions& dims) {
  return basic_make_arithmetic_tensor_input_other<std::uint32_t>(dims);
}

tensor::TensorCP BEAVYProvider::make_arithmetic_64_tensor_input_other(
    const tensor::TensorDimensions& dims) {
  return basic_make_arithmetic_tensor_input_other<std::uint64_t>(dims);
}

template <typename T>
ENCRYPTO::ReusableFiberFuture<IntegerValues<T>>
BEAVYProvider::basic_make_arithmetic_tensor_output_my(const tensor::TensorCP& in) {
  auto input = std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(in);
  if (input == nullptr) {
    throw std::logic_error("wrong tensor type");
  }
  auto gate_id = gate_register_.get_next_gate_id();
  auto tensor_op =
      std::make_unique<ArithmeticBEAVYTensorOutput<T>>(gate_id, *this, std::move(input), my_id_);
  auto future = tensor_op->get_output_future();
  gate_register_.register_gate(std::move(tensor_op));
  return future;
}

template ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint64_t>>
BEAVYProvider::basic_make_arithmetic_tensor_output_my(const tensor::TensorCP&);

ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint32_t>>
BEAVYProvider::make_arithmetic_32_tensor_output_my(const tensor::TensorCP& in) {
  return basic_make_arithmetic_tensor_output_my<std::uint32_t>(in);
}

ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint64_t>>
BEAVYProvider::make_arithmetic_64_tensor_output_my(const tensor::TensorCP& in) {
  return basic_make_arithmetic_tensor_output_my<std::uint64_t>(in);
}

void BEAVYProvider::make_arithmetic_tensor_output_other(const tensor::TensorCP& in) {
  std::unique_ptr<NewGate> gate;
  auto gate_id = gate_register_.get_next_gate_id();
  switch (in->get_bit_size()) {
    case 32: {
      gate = std::make_unique<ArithmeticBEAVYTensorOutput<std::uint32_t>>(
          gate_id, *this, std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<std::uint32_t>>(in),
          1 - my_id_);
      break;
    }
    case 64: {
      gate = std::make_unique<ArithmeticBEAVYTensorOutput<std::uint64_t>>(
          gate_id, *this, std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<std::uint64_t>>(in),
          1 - my_id_);
      break;
    }
    default: {
      throw std::logic_error("unsupprted bit size");
    }
  }
  gate_register_.register_gate(std::move(gate));
}

tensor::TensorCP BEAVYProvider::make_tensor_flatten_op(const tensor::TensorCP input,
                                                       std::size_t axis) {
  if (axis > 4) {
    throw std::invalid_argument("invalid axis argument > 4");
  }
  auto bit_size = input->get_bit_size();
  std::unique_ptr<NewGate> gate;
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  const auto make_op = [this, input, axis, gate_id, &output](auto dummy_arg) {
    using T = decltype(dummy_arg);
    auto tensor_op = std::make_unique<ArithmeticBEAVYTensorFlatten<T>>(
        gate_id, *this, axis, std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(input));
    output = tensor_op->get_output_tensor();
    return tensor_op;
  };
  switch (bit_size) {
    case 32:
      gate = make_op(std::uint32_t{});
      break;
    case 64:
      gate = make_op(std::uint64_t{});
      break;
    default:
      throw std::logic_error(fmt::format("unexpected bit size {}", bit_size));
  }
  gate_register_.register_gate(std::move(gate));
  return output;
}

tensor::TensorCP BEAVYProvider::make_tensor_conversion(MPCProtocol dst_proto,
                                                       const tensor::TensorCP input) {
  auto src_proto = input->get_protocol();
  if (src_proto == MPCProtocol::BooleanBEAVY && dst_proto == MPCProtocol::ArithmeticBEAVY) {
    return make_convert_boolean_to_arithmetic_beavy_tensor(input);
  }
  throw std::invalid_argument(fmt::format("BEAVYProvider: cannot convert tensor from {} to {}",
                                          ToString(src_proto), ToString(dst_proto)));
}

tensor::TensorCP BEAVYProvider::make_tensor_conv2d_op(const tensor::Conv2DOp& conv_op,
                                                      const tensor::TensorCP input,
                                                      const tensor::TensorCP kernel,
                                                      const tensor::TensorCP bias,
                                                      std::size_t fractional_bits) {
  if (!conv_op.verify()) {
    throw std::invalid_argument("invalid Conv2dOp");
  }
  if (input->get_dimensions() != conv_op.get_input_tensor_dims()) {
    throw std::invalid_argument("invalid input dimensions");
  }
  if (kernel->get_dimensions() != conv_op.get_kernel_tensor_dims()) {
    throw std::invalid_argument("invalid kernel dimensions");
  }
  auto bit_size = input->get_bit_size();
  if (bit_size != kernel->get_bit_size()) {
    throw std::invalid_argument("bit size mismatch");
  }
  if (bias != nullptr) {
    if (bias->get_dimensions().get_data_size() != conv_op.compute_bias_size()) {
      throw std::invalid_argument("invalid bias size");
    }
    if (bit_size != bias->get_bit_size()) {
      throw std::invalid_argument("bit size mismatch");
    }
  }
  std::unique_ptr<NewGate> gate;
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  const auto make_op = [this, input, conv_op, kernel, bias, fractional_bits, gate_id,
                        &output](auto dummy_arg) {
    using T = decltype(dummy_arg);
    auto input_ptr = std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(input);
    auto kernel_ptr = std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(kernel);
    std::shared_ptr<const ArithmeticBEAVYTensor<T>> bias_ptr = nullptr;
    if (bias != nullptr) {
      bias_ptr = std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(bias);
      assert(bias_ptr);
    }
    auto tensor_op = std::make_unique<ArithmeticBEAVYTensorConv2D<T>>(
        gate_id, *this, conv_op, input_ptr, kernel_ptr, bias_ptr, fractional_bits);
    output = tensor_op->get_output_tensor();
    return tensor_op;
  };
  switch (bit_size) {
    case 32:
      gate = make_op(std::uint32_t{});
      break;
    case 64:
      gate = make_op(std::uint64_t{});
      break;
    default:
      throw std::logic_error(fmt::format("unexpected bit size {}", bit_size));
  }
  gate_register_.register_gate(std::move(gate));
  return output;
}

tensor::TensorCP BEAVYProvider::make_tensor_gemm_op(const tensor::GemmOp& gemm_op,
                                                    const tensor::TensorCP input_A,
                                                    const tensor::TensorCP input_B,
                                                    std::size_t fractional_bits) {
  if (!gemm_op.verify()) {
    throw std::invalid_argument("invalid GemmOp");
  }
  if (input_A->get_dimensions() != gemm_op.get_input_A_tensor_dims()) {
    throw std::invalid_argument("invalid input_A dimensions");
  }
  if (input_B->get_dimensions() != gemm_op.get_input_B_tensor_dims()) {
    throw std::invalid_argument("invalid input_B dimensions");
  }
  auto bit_size = input_A->get_bit_size();
  if (bit_size != input_B->get_bit_size()) {
    throw std::invalid_argument("bit size mismatch");
  }
  std::unique_ptr<NewGate> gate;
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  const auto make_op = [this, input_A, gemm_op, input_B, fractional_bits, gate_id,
                        &output](auto dummy_arg) {
    using T = decltype(dummy_arg);
    auto tensor_op = std::make_unique<ArithmeticBEAVYTensorGemm<T>>(
        gate_id, *this, gemm_op, std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(input_A),
        std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(input_B), fractional_bits);
    output = tensor_op->get_output_tensor();
    return tensor_op;
  };
  switch (bit_size) {
    case 32:
      gate = make_op(std::uint32_t{});
      break;
    case 64:
      gate = make_op(std::uint64_t{});
      break;
    default:
      throw std::logic_error(fmt::format("unexpected bit size {}", bit_size));
  }
  gate_register_.register_gate(std::move(gate));
  return output;
}

tensor::TensorCP BEAVYProvider::make_tensor_sqr_op(const tensor::TensorCP input,
                                                   std::size_t fractional_bits) {
  auto bit_size = input->get_bit_size();
  std::unique_ptr<NewGate> gate;
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  const auto make_op = [this, input, fractional_bits, gate_id, &output](auto dummy_arg) {
    using T = decltype(dummy_arg);
    auto tensor_op = std::make_unique<ArithmeticBEAVYTensorMul<T>>(
        gate_id, *this, std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(input),
        std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(input), fractional_bits);
    output = tensor_op->get_output_tensor();
    return tensor_op;
  };
  switch (bit_size) {
    case 32:
      gate = make_op(std::uint32_t{});
      break;
    case 64:
      gate = make_op(std::uint64_t{});
      break;
    default:
      throw std::logic_error(fmt::format("unexpected bit size {}", bit_size));
  }
  gate_register_.register_gate(std::move(gate));
  return output;
}

tensor::TensorCP BEAVYProvider::make_tensor_avgpool_op(const tensor::AveragePoolOp& avgpool_op,
                                                       const tensor::TensorCP input,
                                                       std::size_t fractional_bits) {
  auto bit_size = input->get_bit_size();
  std::unique_ptr<NewGate> gate;
  auto gate_id = gate_register_.get_next_gate_id();
  tensor::TensorCP output;
  const auto make_op = [this, input, &avgpool_op, fractional_bits, gate_id,
                        &output](auto dummy_arg) {
    using T = decltype(dummy_arg);
    auto tensor_op = std::make_unique<ArithmeticBEAVYTensorAveragePool<T>>(
        gate_id, *this, avgpool_op,
        std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(input), fractional_bits);
    output = tensor_op->get_output_tensor();
    return tensor_op;
  };
  switch (bit_size) {
    case 32:
      gate = make_op(std::uint32_t{});
      break;
    case 64:
      gate = make_op(std::uint64_t{});
      break;
    default:
      throw std::logic_error(fmt::format("unexpected bit size {}", bit_size));
  }
  gate_register_.register_gate(std::move(gate));
  return output;
}

tensor::TensorCP BEAVYProvider::make_tensor_relu_op(const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const BooleanBEAVYTensor>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  auto tensor_op = std::make_unique<BooleanBEAVYTensorRelu>(gate_id, *this, input_tensor);
  auto output = tensor_op->get_output_tensor();
  gate_register_.register_gate(std::move(tensor_op));
  return output;
}

template <typename T>
tensor::TensorCP BEAVYProvider::basic_make_tensor_relu_op(const tensor::TensorCP in_bool,
                                                          const tensor::TensorCP in_arith) {
  const auto input_bool_tensor = std::dynamic_pointer_cast<const BooleanBEAVYTensor>(in_bool);
  assert(input_bool_tensor != nullptr);
  const auto input_arith_tensor =
      std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<T>>(in_arith);
  assert(input_arith_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  auto tensor_op = std::make_unique<BooleanXArithmeticBEAVYTensorRelu<T>>(
      gate_id, *this, input_bool_tensor, input_arith_tensor);
  auto output = tensor_op->get_output_tensor();
  gate_register_.register_gate(std::move(tensor_op));
  return output;
}

tensor::TensorCP BEAVYProvider::make_tensor_relu_op(const tensor::TensorCP in_bool,
                                                    const tensor::TensorCP in_arith) {
  if (in_bool->get_protocol() != MPCProtocol::BooleanBEAVY ||
      in_arith->get_protocol() != MPCProtocol::ArithmeticBEAVY) {
    throw std::invalid_argument("expected Boolean and arithmetic BEAVY, respectively");
  }
  const auto bit_size = in_bool->get_bit_size();
  if (bit_size != in_arith->get_bit_size()) {
    throw std::invalid_argument("bit size mismatch");
  }
  switch (bit_size) {
    case 32:
      return basic_make_tensor_relu_op<std::uint32_t>(in_bool, in_arith);
      break;
    case 64:
      return basic_make_tensor_relu_op<std::uint64_t>(in_bool, in_arith);
      break;
    default:
      throw std::invalid_argument(fmt::format("unexpected bit size {}", bit_size));
  }
}

tensor::TensorCP BEAVYProvider::make_tensor_maxpool_op(const tensor::MaxPoolOp& maxpool_op,
                                                       const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const BooleanBEAVYTensor>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  auto tensor_op =
      std::make_unique<BooleanBEAVYTensorMaxPool>(gate_id, *this, maxpool_op, input_tensor);
  auto output = tensor_op->get_output_tensor();
  gate_register_.register_gate(std::move(tensor_op));
  return output;
}

template <typename T>
tensor::TensorCP BEAVYProvider::basic_make_convert_boolean_to_arithmetic_beavy_tensor(
    const tensor::TensorCP in) {
  const auto input_tensor = std::dynamic_pointer_cast<const BooleanBEAVYTensor>(in);
  assert(input_tensor != nullptr);
  auto gate_id = gate_register_.get_next_gate_id();
  auto tensor_op =
      std::make_unique<BooleanToArithmeticBEAVYTensorConversion<T>>(gate_id, *this, input_tensor);
  auto output = tensor_op->get_output_tensor();
  gate_register_.register_gate(std::move(tensor_op));
  return output;
}

template tensor::TensorCP BEAVYProvider::basic_make_convert_boolean_to_arithmetic_beavy_tensor<
    std::uint64_t>(const tensor::TensorCP);

tensor::TensorCP BEAVYProvider::make_convert_boolean_to_arithmetic_beavy_tensor(
    const tensor::TensorCP in) {
  switch (in->get_bit_size()) {
    case 32: {
      return basic_make_convert_boolean_to_arithmetic_beavy_tensor<std::uint32_t>(std::move(in));
      break;
    }
    case 64: {
      return basic_make_convert_boolean_to_arithmetic_beavy_tensor<std::uint64_t>(std::move(in));
      break;
    }
    default: {
      throw std::logic_error("unsupported bit size");
    }
  }
}

}  // namespace MOTION::proto::beavy
