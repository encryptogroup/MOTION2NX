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

#include "base/gate_register.h"
#include "communication/communication_layer.h"
#include "gate.h"
#include "utility/typedefs.h"

namespace MOTION::proto::yao {

YaoProvider::YaoProvider(Communication::CommunicationLayer &communication_layer,
                         GateRegister &gate_register,
                         ENCRYPTO::ObliviousTransfer::OTProvider &ot_provider,
                         std::shared_ptr<Logger> logger)
    : communication_layer_(communication_layer),
      gate_register_(gate_register),
      ot_provider_(ot_provider),
      logger_(logger) {
  if (communication_layer.get_num_parties() != 2) {
    throw std::logic_error("Yao is a two party protocol");
  }
  auto my_id = communication_layer_.get_my_id();
  role_ = (my_id == 0) ? Role::garbler : Role::evaluator;
}

static YaoWireVector cast_wires(std::vector<std::shared_ptr<NewWire>> wires) {
  YaoWireVector result(wires.size());
  std::transform(std::begin(wires), std::end(wires), std::begin(result),
                 [](auto &w) { return std::dynamic_pointer_cast<YaoWire>(w); });
  return result;
}

static std::vector<std::shared_ptr<NewWire>> cast_wires(YaoWireVector &&wires) {
  return std::vector<std::shared_ptr<NewWire>>(std::begin(wires), std::end(wires));
}

std::vector<std::shared_ptr<NewWire>> YaoProvider::make_unary_gate(
    ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>> &in_a) const {
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
    const std::vector<std::shared_ptr<NewWire>> &in_b) const {
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

YaoWireVector YaoProvider::make_inv_gate(YaoWireVector &&in_a) const {
  YaoWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  if (role_ == Role::garbler) {
    auto gate = std::make_unique<YaoINVGateGarbler>(gate_id, *this, std::move(in_a));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  } else {
    auto gate = std::make_unique<YaoINVGateEvaluator>(gate_id, *this, std::move(in_a));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  }
  return output;
}

YaoWireVector YaoProvider::make_xor_gate(YaoWireVector &&in_a, YaoWireVector &&in_b) const {
  YaoWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  if (role_ == Role::garbler) {
    auto gate =
        std::make_unique<YaoXORGateGarbler>(gate_id, *this, std::move(in_a), std::move(in_b));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  } else {
    auto gate =
        std::make_unique<YaoXORGateEvaluator>(gate_id, *this, std::move(in_a), std::move(in_b));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  }
  return output;
}

YaoWireVector YaoProvider::make_and_gate(YaoWireVector &&in_a, YaoWireVector &&in_b) const {
  YaoWireVector output;
  auto gate_id = gate_register_.get_next_gate_id();
  if (role_ == Role::garbler) {
    auto gate =
        std::make_unique<YaoANDGateGarbler>(gate_id, *this, std::move(in_a), std::move(in_b));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  } else {
    auto gate =
        std::make_unique<YaoANDGateEvaluator>(gate_id, *this, std::move(in_a), std::move(in_b));
    output = gate->get_output_wires();
    gate_register_.register_gate(std::move(gate));
  }
  return output;
}

void YaoProvider::setup() noexcept {
  global_offset_.set_to_random();
  // The first bit of the key is the permutation bit which needs to be
  // different for the zero and one keys.  So we have to set the corresponding
  // bit of the global offset to 1.
  *global_offset_.data() |= std::byte(0x01);
}

void YaoProvider::send_keys_message(std::size_t gate_id, ENCRYPTO::block128_vector&& message) const {
  // TODO
}
void YaoProvider::send_bits_message(std::size_t gate_id, ENCRYPTO::BitVector<>&& message) const {
  // TODO
}
void YaoProvider::send_bits_message(std::size_t gate_id, const ENCRYPTO::BitVector<>& message) const {
  // TODO
}

void YaoProvider::create_garbled_tables(const ENCRYPTO::block128_vector &keys_a,
                                        const ENCRYPTO::block128_vector &keys_b,
                                        ENCRYPTO::block128_vector &tables,
                                        ENCRYPTO::block128_vector &keys_out) const noexcept {
  // TODO
}

void YaoProvider::evaluate_garbled_tables(const ENCRYPTO::block128_vector &keys_a,
                                          const ENCRYPTO::block128_vector &keys_b,
                                          const ENCRYPTO::block128_vector &tables,
                                          ENCRYPTO::block128_vector &keys_out) const noexcept {
  // TODO
}

}  // namespace MOTION::proto::yao
