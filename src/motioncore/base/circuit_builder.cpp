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

#include "circuit_builder.h"

#include <stdexcept>

#include <fmt/format.h>

#include "algorithm/algorithm_description.h"
#include "algorithm/make_circuit.h"
#include "gate_factory.h"
#include "wire/new_wire.h"

namespace MOTION {

WireVector CircuitBuilder::make_unary_gate(ENCRYPTO::PrimitiveOperationType op,
                                           const WireVector& wires) {
  if (wires.empty()) {
    throw std::logic_error("empty WireVector");
  }
  auto proto = wires[0]->get_protocol();
  auto& gate_factory = get_gate_factory(proto);
  return gate_factory.make_unary_gate(op, wires);
}

WireVector CircuitBuilder::make_binary_gate(ENCRYPTO::PrimitiveOperationType op,
                                            const WireVector& wires_a, const WireVector& wires_b) {
  if (wires_a.empty() || wires_b.empty()) {
    throw std::logic_error("empty WireVector");
  }
  auto proto_a = wires_a[0]->get_protocol();
  auto proto_b = wires_b[0]->get_protocol();
  if (proto_a == proto_b) {
    auto& gate_factory = get_gate_factory(proto_a);
    return gate_factory.make_binary_gate(op, wires_a, wires_b);
  } else if (proto_a == MPCProtocol::ArithmeticPlain || proto_a == MPCProtocol::BooleanPlain) {
    auto& gate_factory = get_gate_factory(proto_b);
    return gate_factory.make_binary_gate(op, wires_a, wires_b);
  } else if (proto_b == MPCProtocol::ArithmeticPlain || proto_b == MPCProtocol::BooleanPlain) {
    auto& gate_factory = get_gate_factory(proto_a);
    return gate_factory.make_binary_gate(op, wires_a, wires_b);
  } else {
    throw std::logic_error("multi-protocol gates are not yet supported");
  }
}

WireVector CircuitBuilder::convert(MPCProtocol dst_proto, const WireVector& wires) {
  if (wires.empty()) {
    throw std::logic_error("empty WireVector");
  }
  const auto src_proto = wires[0]->get_protocol();
  const auto convert_f = [dst_proto, wires](auto& factory) -> std::pair<WireVector, bool> {
    try {
      auto output_wires = factory.convert(dst_proto, wires);
      return {std::move(output_wires), true};
    } catch (std::exception& e) {
      return {{}, false};
    }
  };
  auto via_protocol = convert_via(src_proto, dst_proto);
  if (via_protocol.has_value()) {
    // implicit conversion via third protocol
    auto tmp = convert(*via_protocol, wires);
    return convert(dst_proto, tmp);
  } else {
    // direct conversion
    {
      auto& factory = get_gate_factory(src_proto);
      auto [output_wires, success] = convert_f(factory);
      if (success) {
        return output_wires;
      }
    }
    {
      auto& factory = get_gate_factory(dst_proto);
      auto [output_wires, success] = convert_f(factory);
      if (success) {
        return output_wires;
      }
    }
  }
  throw std::runtime_error(fmt::format("no conversion from {} to {} supported", ToString(src_proto),
                                       ToString(dst_proto)));
}

std::optional<MPCProtocol> CircuitBuilder::convert_via(MPCProtocol, MPCProtocol) {
  return std::nullopt;
}

WireVector CircuitBuilder::make_circuit(const ENCRYPTO::AlgorithmDescription& algo,
                                        const WireVector& wires_in_a,
                                        const WireVector& wires_in_b) {
  return ::MOTION::make_circuit(*this, algo, wires_in_a, wires_in_b);
}

}  // namespace MOTION
