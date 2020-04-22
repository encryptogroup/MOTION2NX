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

#include <limits>
#include <memory>
#include <vector>

#include "utility/bit_vector.h"
#include "utility/reusable_future.h"

namespace ENCRYPTO {
enum class PrimitiveOperationType : std::uint8_t;
}

namespace MOTION {

class NewWire;
constexpr std::size_t ALL_PARTIES = std::numeric_limits<std::size_t>::max();
using WireVector = std::vector<std::shared_ptr<NewWire>>;
using BitValues = std::vector<ENCRYPTO::BitVector<>>;

class GateFactory {
 public:
  virtual std::pair<ENCRYPTO::ReusableFiberPromise<BitValues>, WireVector>
  make_boolean_input_gate_my(std::size_t input_owner, std::size_t num_wires,
                             std::size_t num_simd) = 0;

  virtual WireVector make_boolean_input_gate_other(std::size_t input_owner, std::size_t num_wires,
                                                   std::size_t num_simd) = 0;

  virtual ENCRYPTO::ReusableFiberFuture<BitValues> make_boolean_output_gate_my(
      std::size_t output_owner, const WireVector&) = 0;

  virtual void make_boolean_output_gate_other(std::size_t output_owner, const WireVector&) = 0;

  virtual WireVector make_unary_gate(ENCRYPTO::PrimitiveOperationType op, const WireVector&) = 0;

  virtual WireVector make_binary_gate(ENCRYPTO::PrimitiveOperationType op, const WireVector&,
                                      const WireVector&) = 0;
};

}  // namespace MOTION
