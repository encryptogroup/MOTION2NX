// MIT License
//
// Copyright (c) 2020-2021 Lennart Braun
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

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ENCRYPTO {
struct AlgorithmDescription;
enum class PrimitiveOperationType : std::uint8_t;
}  // namespace ENCRYPTO

namespace MOTION {

class GateFactory;
enum class MPCProtocol : unsigned int;
class NewWire;
using WireVector = std::vector<std::shared_ptr<NewWire>>;

class CircuitBuilder {
 public:
  WireVector make_unary_gate(ENCRYPTO::PrimitiveOperationType, const WireVector&);
  WireVector make_binary_gate(ENCRYPTO::PrimitiveOperationType, const WireVector&,
                              const WireVector&);
  WireVector make_ternary_gate(ENCRYPTO::PrimitiveOperationType, const WireVector&,
                               const WireVector&, const WireVector&);
  WireVector make_quarternary_gate(ENCRYPTO::PrimitiveOperationType, const WireVector&,
                                   const WireVector&, const WireVector&, const WireVector&);
  WireVector make_circuit(const ENCRYPTO::AlgorithmDescription&, const WireVector&,
                          const WireVector&);
  WireVector convert(MPCProtocol, const WireVector&);
  virtual std::optional<MPCProtocol> convert_via(MPCProtocol src, MPCProtocol dst);
  virtual GateFactory& get_gate_factory(MPCProtocol) = 0;
};

}  // namespace MOTION
