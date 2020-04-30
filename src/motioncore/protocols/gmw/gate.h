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

namespace MOTION::proto::gmw {

namespace detail {

template <typename WireType>
class BasicGMWBinaryGate : public NewGate {
 public:
  using GMWWireVector = std::vector<std::shared_ptr<WireType>>;
  BasicGMWBinaryGate(std::size_t gate_id, GMWWireVector&&, GMWWireVector&&);
  GMWWireVector& get_output_wires() noexcept { return outputs_; };

 protected:
  std::size_t num_wires_;
  const GMWWireVector inputs_a_;
  const GMWWireVector inputs_b_;
  GMWWireVector outputs_;
};

template <typename WireType>
class BasicGMWUnaryGate : public NewGate {
 public:
  using GMWWireVector = std::vector<std::shared_ptr<WireType>>;
  BasicGMWUnaryGate(std::size_t gate_id, GMWWireVector&&, bool forward);
  GMWWireVector& get_output_wires() noexcept { return outputs_; };

 protected:
  std::size_t num_wires_;
  const GMWWireVector inputs_;
  GMWWireVector outputs_;
};

}

class GMWProvider;
class BooleanGMWWire;
using BooleanGMWWireVector = std::vector<std::shared_ptr<BooleanGMWWire>>;

class BooleanGMWInputGateSender : public NewGate {
 public:
  BooleanGMWInputGateSender(std::size_t gate_id, GMWProvider&, std::size_t num_wires,
                            std::size_t num_simd,
                            ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;

 private:
  GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>> input_future_;
  BooleanGMWWireVector outputs_;
};

class BooleanGMWInputGateReceiver : public NewGate {
 public:
  BooleanGMWInputGateReceiver(std::size_t gate_id, GMWProvider&, std::size_t num_wires,
                              std::size_t num_simd, std::size_t input_owner);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return false; }
  void evaluate_setup() override;
  void evaluate_online() override;

 private:
  GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  std::size_t input_owner_;
  BooleanGMWWireVector outputs_;
};

class BooleanGMWOutputGate : public NewGate {
 public:
  BooleanGMWOutputGate(std::size_t gate_id, GMWProvider&, BooleanGMWWireVector&&, std::size_t output_owner);
  ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>> get_output_future();
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;

 private:
  GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  std::size_t output_owner_;
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> output_promise_;
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>> share_futures_;
  const BooleanGMWWireVector inputs_;
};

class BooleanGMWINVGate : public detail::BasicGMWUnaryGate<BooleanGMWWire> {
 public:
  BooleanGMWINVGate(std::size_t gate_id, const GMWProvider&, GMWWireVector&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
 private:
  bool is_my_job_;
};

class BooleanGMWXORGate : public detail::BasicGMWBinaryGate<BooleanGMWWire> {
 public:
  using detail::BasicGMWBinaryGate<BooleanGMWWire>::BasicGMWBinaryGate;
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
};

class BooleanGMWANDGate : public detail::BasicGMWBinaryGate<BooleanGMWWire> {
 public:
  BooleanGMWANDGate(std::size_t gate_id, GMWProvider&, GMWWireVector&&, GMWWireVector&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
 private:
  GMWProvider& gmw_provider_;
  // TODO: MT ids
};


template <typename T>
class ArithmeticGMWNEGGate : public detail::BasicGMWUnaryGate<ArithmeticGMWWire<T>> {
 public:
  using GMWWireVector = typename detail::BasicGMWUnaryGate<ArithmeticGMWWire<T>>::GMWWireVector;
  ArithmeticGMWNEGGate(std::size_t gate_id, const GMWProvider&, GMWWireVector&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
};

template <typename T>
class ArithmeticGMWADDGate : public detail::BasicGMWBinaryGate<ArithmeticGMWWire<T>> {
 public:
  using detail::BasicGMWBinaryGate<ArithmeticGMWWire<T>>::BasicGMWBinaryGate;
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
};

template <typename T>
class ArithmeticGMWMULGate : public detail::BasicGMWBinaryGate<ArithmeticGMWWire<T>> {
 public:
  using GMWWireVector = typename detail::BasicGMWUnaryGate<ArithmeticGMWWire<T>>::GMWWireVector;
  ArithmeticGMWMULGate(std::size_t gate_id, GMWProvider&, GMWWireVector&&, GMWWireVector&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
 private:
  GMWProvider& gmw_provider_;
  // TODO: MT ids
};

template <typename T>
class ArithmeticGMWSQRGate : public detail::BasicGMWBinaryGate<ArithmeticGMWWire<T>> {
 public:
  using GMWWireVector = typename detail::BasicGMWUnaryGate<ArithmeticGMWWire<T>>::GMWWireVector;
  ArithmeticGMWSQRGate(std::size_t gate_id, GMWProvider&, GMWWireVector&&, GMWWireVector&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
 private:
  GMWProvider& gmw_provider_;
  // TODO: SP ids
};

}  // namespace MOTION::proto::gmw
