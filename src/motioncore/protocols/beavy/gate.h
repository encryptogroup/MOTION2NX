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

#include <cstddef>

#include "gate/new_gate.h"
#include "utility/bit_vector.h"
#include "utility/reusable_future.h"
#include "utility/type_traits.hpp"
#include "wire.h"

namespace ENCRYPTO::ObliviousTransfer {
class XCOTBitSender;
class XCOTBitReceiver;
}

namespace MOTION::proto::beavy {

namespace detail {

class BasicBooleanBEAVYBinaryGate : public NewGate {
 public:
  BasicBooleanBEAVYBinaryGate(std::size_t gate_id, BooleanBEAVYWireVector&&, BooleanBEAVYWireVector&&);
  BooleanBEAVYWireVector& get_output_wires() noexcept { return outputs_; }

 protected:
  std::size_t num_wires_;
  const BooleanBEAVYWireVector inputs_a_;
  const BooleanBEAVYWireVector inputs_b_;
  BooleanBEAVYWireVector outputs_;
};

class BasicBooleanBEAVYUnaryGate : public NewGate {
 public:
  BasicBooleanBEAVYUnaryGate(std::size_t gate_id, BooleanBEAVYWireVector&&, bool forward);
  BooleanBEAVYWireVector& get_output_wires() noexcept { return outputs_; }

 protected:
  std::size_t num_wires_;
  const BooleanBEAVYWireVector inputs_;
  BooleanBEAVYWireVector outputs_;
};

}  // namespace detail

class BEAVYProvider;
class BooleanBEAVYWire;
using BooleanBEAVYWireVector = std::vector<std::shared_ptr<BooleanBEAVYWire>>;

class BooleanBEAVYInputGateSender : public NewGate {
 public:
  BooleanBEAVYInputGateSender(std::size_t gate_id, BEAVYProvider&, std::size_t num_wires,
                            std::size_t num_simd,
                            ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  BooleanBEAVYWireVector& get_output_wires() noexcept { return outputs_; }

 private:
  BEAVYProvider& beavy_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  std::size_t input_id_;
  ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>> input_future_;
  BooleanBEAVYWireVector outputs_;
};

class BooleanBEAVYInputGateReceiver : public NewGate {
 public:
  BooleanBEAVYInputGateReceiver(std::size_t gate_id, BEAVYProvider&, std::size_t num_wires,
                              std::size_t num_simd, std::size_t input_owner);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  BooleanBEAVYWireVector& get_output_wires() noexcept { return outputs_; }

 private:
  BEAVYProvider& beavy_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  std::size_t input_owner_;
  std::size_t input_id_;
  BooleanBEAVYWireVector outputs_;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> public_share_future_;
};

class BooleanBEAVYOutputGate : public NewGate {
 public:
  BooleanBEAVYOutputGate(std::size_t gate_id, BEAVYProvider&, BooleanBEAVYWireVector&&,
                       std::size_t output_owner);
  ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>> get_output_future();
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;

 private:
  BEAVYProvider& beavy_provider_;
  std::size_t num_wires_;
  std::size_t output_owner_;
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> output_promise_;
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>> share_futures_;
  const BooleanBEAVYWireVector inputs_;
};

class BooleanBEAVYINVGate : public detail::BasicBooleanBEAVYUnaryGate {
 public:
  BooleanBEAVYINVGate(std::size_t gate_id, const BEAVYProvider&, BooleanBEAVYWireVector&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;

 private:
  bool is_my_job_;
};

class BooleanBEAVYXORGate : public detail::BasicBooleanBEAVYBinaryGate {
 public:
  BooleanBEAVYXORGate(std::size_t gate_id, BEAVYProvider&, BooleanBEAVYWireVector&&,
                    BooleanBEAVYWireVector&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
};

class BooleanBEAVYANDGate : public detail::BasicBooleanBEAVYBinaryGate {
 public:
  BooleanBEAVYANDGate(std::size_t gate_id, BEAVYProvider&, BooleanBEAVYWireVector&&,
                    BooleanBEAVYWireVector&&);
  ~BooleanBEAVYANDGate();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;

 private:
  BEAVYProvider& beavy_provider_;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> share_future_;
  ENCRYPTO::BitVector<> delta_a_share_;
  ENCRYPTO::BitVector<> delta_b_share_;
  ENCRYPTO::BitVector<> Delta_y_share_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitSender> ot_sender_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitReceiver> ot_receiver_;
};

template <typename T>
class ArithmeticBEAVYInputGateSender : public NewGate {
 public:
  ArithmeticBEAVYInputGateSender(std::size_t gate_id, BEAVYProvider&, std::size_t num_simd,
                               ENCRYPTO::ReusableFiberFuture<std::vector<T>>&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  ArithmeticBEAVYWireP<T>& get_output_wire() noexcept { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  std::size_t input_id_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> input_future_;
  ArithmeticBEAVYWireP<T> output_;
};

template <typename T>
class ArithmeticBEAVYInputGateReceiver : public NewGate {
 public:
  ArithmeticBEAVYInputGateReceiver(std::size_t gate_id, BEAVYProvider&, std::size_t num_simd,
                                 std::size_t input_owner);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  ArithmeticBEAVYWireP<T>& get_output_wire() noexcept { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  std::size_t input_owner_;
  std::size_t input_id_;
  ArithmeticBEAVYWireP<T> output_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> public_share_future_;
};

}  // namespace MOTION::proto::beavy
