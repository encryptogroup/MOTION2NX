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
template <typename T>
using IntegerValues = std::vector<T>;

class GateFactory {
 public:
  virtual ~GateFactory() = 0;

  virtual std::string get_provider_name() const noexcept = 0;

  // Boolean inputs
  virtual std::pair<ENCRYPTO::ReusableFiberPromise<BitValues>, WireVector>
  make_boolean_input_gate_my(std::size_t input_owner, std::size_t num_wires, std::size_t num_simd);

  virtual WireVector make_boolean_input_gate_other(std::size_t input_owner, std::size_t num_wires,
                                                   std::size_t num_simd);

  // arithmetic inputs
  virtual std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint8_t>>, WireVector>
  make_arithmetic_8_input_gate_my(std::size_t input_owner, std::size_t num_simd);
  virtual std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint16_t>>, WireVector>
  make_arithmetic_16_input_gate_my(std::size_t input_owner, std::size_t num_simd);
  virtual std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint32_t>>, WireVector>
  make_arithmetic_32_input_gate_my(std::size_t input_owner, std::size_t num_simd);
  virtual std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint64_t>>, WireVector>
  make_arithmetic_64_input_gate_my(std::size_t input_owner, std::size_t num_simd);

  virtual WireVector make_arithmetic_8_input_gate_other(std::size_t input_owner,
                                                        std::size_t num_simd);
  virtual WireVector make_arithmetic_16_input_gate_other(std::size_t input_owner,
                                                         std::size_t num_simd);
  virtual WireVector make_arithmetic_32_input_gate_other(std::size_t input_owner,
                                                         std::size_t num_simd);
  virtual WireVector make_arithmetic_64_input_gate_other(std::size_t input_owner,
                                                         std::size_t num_simd);

  // Boolean outputs
  virtual ENCRYPTO::ReusableFiberFuture<BitValues> make_boolean_output_gate_my(
      std::size_t output_owner, const WireVector&);

  virtual void make_boolean_output_gate_other(std::size_t output_owner, const WireVector&);

  // arithmetic outputs
  virtual ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint8_t>>
  make_arithmetic_8_output_gate_my(std::size_t output_owner, const WireVector&);
  virtual ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint16_t>>
  make_arithmetic_16_output_gate_my(std::size_t output_owner, const WireVector&);
  virtual ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint32_t>>
  make_arithmetic_32_output_gate_my(std::size_t output_owner, const WireVector&);
  virtual ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint64_t>>
  make_arithmetic_64_output_gate_my(std::size_t output_owner, const WireVector&);

  virtual void make_arithmetic_output_gate_other(std::size_t output_owner, const WireVector&);

  // function gates

  virtual WireVector make_unary_gate(ENCRYPTO::PrimitiveOperationType op, const WireVector&) = 0;
  virtual WireVector make_binary_gate(ENCRYPTO::PrimitiveOperationType op, const WireVector&,
                                      const WireVector&) = 0;
  virtual WireVector make_ternary_gate(ENCRYPTO::PrimitiveOperationType op, const WireVector&,
                                       const WireVector&, const WireVector&);
  virtual WireVector make_quarternary_gate(ENCRYPTO::PrimitiveOperationType op, const WireVector&,
                                           const WireVector&, const WireVector&, const WireVector&);

  // conversions
  virtual WireVector convert(MPCProtocol dst_protocol, const WireVector&) = 0;
};

}  // namespace MOTION
