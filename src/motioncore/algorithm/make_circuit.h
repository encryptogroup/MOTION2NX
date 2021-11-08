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

#pragma once

#include <fmt/format.h>

#include "algorithm_description.h"
#include "gate/new_gate.h"
#include "wire/new_wire.h"

namespace MOTION {

using WireVector = std::vector<std::shared_ptr<NewWire>>;

template <typename Builder>
WireVector make_circuit(Builder& builder, const ENCRYPTO::AlgorithmDescription& algo,
                        const WireVector& wires_in_a, const WireVector& wires_in_b) {
  if (!algo.n_input_wires_parent_b_.has_value()) {
    throw std::invalid_argument("AlgorithmDescription expects 1 input but 2 are provided");
  }
  if (algo.n_input_wires_parent_a_ != wires_in_a.size()) {
    throw std::invalid_argument(
        fmt::format("AlgorithmDescription expects {} wires for input a, but {} are provided",
                    algo.n_input_wires_parent_a_, wires_in_a.size()));
  }
  if (*algo.n_input_wires_parent_b_ != wires_in_b.size()) {
    throw std::invalid_argument(
        fmt::format("AlgorithmDescription expects {} wires for input b, but {} are provided",
                    *algo.n_input_wires_parent_b_, wires_in_b.size()));
  }
  WireVector circuit_wires(algo.n_wires_);
  // load the input wires into vector
  auto it = std::copy(std::begin(wires_in_a), std::end(wires_in_a), std::begin(circuit_wires));
  std::copy(std::begin(wires_in_b), std::end(wires_in_b), it);
  // create all the gates gate
  for (const auto& prim_op : algo.gates_) {
    auto op_type = prim_op.type_;
    const auto& gate_input_wire_a = circuit_wires.at(prim_op.parent_a_);
    WireVector gate_output_wires;
    if (prim_op.parent_b_.has_value()) {
      const auto& gate_input_wire_b = circuit_wires.at(*prim_op.parent_b_);
      gate_output_wires =
          builder.make_binary_gate(op_type, {gate_input_wire_a}, {gate_input_wire_b});
    } else {
      gate_output_wires = builder.make_unary_gate(op_type, {gate_input_wire_a});
    }
    circuit_wires.at(prim_op.output_wire_) = std::move(gate_output_wires.at(0));
  }
  WireVector output_wires(std::end(circuit_wires) - algo.n_output_wires_, std::end(circuit_wires));
  return output_wires;
}

template <typename Builder>
std::pair<std::vector<std::unique_ptr<NewGate>>, WireVector> construct_circuit(
    Builder& builder, const ENCRYPTO::AlgorithmDescription& algo, const WireVector& wires_in_a) {
  if (algo.n_input_wires_parent_b_.has_value()) {
    throw std::invalid_argument("AlgorithmDescription expects 2 input but only 1 is provided");
  }
  if (algo.n_input_wires_parent_a_ != wires_in_a.size()) {
    throw std::invalid_argument(
        fmt::format("AlgorithmDescription expects {} wires for input a, but {} are provided",
                    algo.n_input_wires_parent_a_, wires_in_a.size()));
  }
  std::vector<std::unique_ptr<NewGate>> gates;
  gates.reserve(algo.n_gates_);
  WireVector circuit_wires(algo.n_wires_);
  // load the input wires into vector
  std::copy(std::begin(wires_in_a), std::end(wires_in_a), std::begin(circuit_wires));
  // create all the gates gate
  for (const auto& prim_op : algo.gates_) {
    auto op_type = prim_op.type_;
    const auto& gate_input_wire_a = circuit_wires.at(prim_op.parent_a_);
    WireVector gate_output_wires;
    if (prim_op.parent_b_.has_value()) {
      const auto& gate_input_wire_b = circuit_wires.at(*prim_op.parent_b_);
      auto ret = builder.construct_binary_gate(op_type, {gate_input_wire_a}, {gate_input_wire_b});
      gates.emplace_back(std::move(ret.first));
      gate_output_wires = std::move(ret.second);
    } else {
      auto ret = builder.construct_unary_gate(op_type, {gate_input_wire_a});
      gates.emplace_back(std::move(ret.first));
      gate_output_wires = std::move(ret.second);
    }
    circuit_wires.at(prim_op.output_wire_) = std::move(gate_output_wires.at(0));
  }
  WireVector output_wires(std::end(circuit_wires) - algo.n_output_wires_, std::end(circuit_wires));
  return {std::move(gates), std::move(output_wires)};
}

}  // namespace MOTION
