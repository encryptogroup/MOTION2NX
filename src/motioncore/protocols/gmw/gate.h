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

class BasicBooleanGMWBinaryGate : public NewGate {
 public:
  BasicBooleanGMWBinaryGate(std::size_t gate_id, BooleanGMWWireVector&&, BooleanGMWWireVector&&);
  BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; }

 protected:
  std::size_t num_wires_;
  const BooleanGMWWireVector inputs_a_;
  const BooleanGMWWireVector inputs_b_;
  BooleanGMWWireVector outputs_;
};

class BasicBooleanGMWUnaryGate : public NewGate {
 public:
  BasicBooleanGMWUnaryGate(std::size_t gate_id, BooleanGMWWireVector&&, bool forward);
  BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; }

 protected:
  std::size_t num_wires_;
  const BooleanGMWWireVector inputs_;
  BooleanGMWWireVector outputs_;
};

}  // namespace detail

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
  BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; }

 private:
  GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  std::size_t input_id_;
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
  void evaluate_online() override {}
  BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; }

 private:
  GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  std::size_t input_owner_;
  std::size_t input_id_;
  BooleanGMWWireVector outputs_;
};

class BooleanGMWOutputGate : public NewGate {
 public:
  BooleanGMWOutputGate(std::size_t gate_id, GMWProvider&, BooleanGMWWireVector&&,
                       std::size_t output_owner);
  ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>> get_output_future();
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;

 private:
  GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  std::size_t output_owner_;
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> output_promise_;
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>> share_futures_;
  const BooleanGMWWireVector inputs_;
};

class BooleanGMWINVGate : public detail::BasicBooleanGMWUnaryGate {
 public:
  BooleanGMWINVGate(std::size_t gate_id, const GMWProvider&, BooleanGMWWireVector&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;

 private:
  bool is_my_job_;
};

class BooleanGMWXORGate : public detail::BasicBooleanGMWBinaryGate {
 public:
  using BasicBooleanGMWBinaryGate::BasicBooleanGMWBinaryGate;
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
};

class BooleanGMWANDGate : public detail::BasicBooleanGMWBinaryGate {
 public:
  BooleanGMWANDGate(std::size_t gate_id, GMWProvider&, BooleanGMWWireVector&&,
                    BooleanGMWWireVector&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;

 private:
  GMWProvider& gmw_provider_;
  std::size_t mt_offset_;
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>> share_futures_;
};

namespace detail {

template <typename T>
class BasicArithmeticGMWBinaryGate : public NewGate {
 public:
  BasicArithmeticGMWBinaryGate(std::size_t gate_id, GMWProvider&, ArithmeticGMWWireP<T>&&,
                               ArithmeticGMWWireP<T>&&);
  ArithmeticGMWWireP<T>& get_output_wire() noexcept { return output_; }

 protected:
  std::size_t num_wires_;
  const ArithmeticGMWWireP<T> input_a_;
  const ArithmeticGMWWireP<T> input_b_;
  ArithmeticGMWWireP<T> output_;
};

template <typename T>
class BasicArithmeticGMWUnaryGate : public NewGate {
 public:
  BasicArithmeticGMWUnaryGate(std::size_t gate_id, GMWProvider&, ArithmeticGMWWireP<T>&&);
  ArithmeticGMWWireP<T>& get_output_wire() noexcept { return output_; }

 protected:
  std::size_t num_wires_;
  const ArithmeticGMWWireP<T> input_;
  ArithmeticGMWWireP<T> output_;
};

}  // namespace detail

template <typename T>
class ArithmeticGMWInputGateSender : public NewGate {
 public:
  ArithmeticGMWInputGateSender(std::size_t gate_id, GMWProvider&, std::size_t num_simd,
                               ENCRYPTO::ReusableFiberFuture<std::vector<T>>&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  ArithmeticGMWWireP<T>& get_output_wire() noexcept { return output_; }

 private:
  GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  std::size_t input_id_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> input_future_;
  ArithmeticGMWWireP<T> output_;
};

template <typename T>
class ArithmeticGMWInputGateReceiver : public NewGate {
 public:
  ArithmeticGMWInputGateReceiver(std::size_t gate_id, GMWProvider&, std::size_t num_simd,
                                 std::size_t input_owner);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return false; }
  void evaluate_setup() override;
  void evaluate_online() override {}
  ArithmeticGMWWireP<T>& get_output_wire() noexcept { return output_; }

 private:
  GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  std::size_t num_simd_;
  std::size_t input_owner_;
  std::size_t input_id_;
  ArithmeticGMWWireP<T> output_;
};

template <typename T>
class ArithmeticGMWOutputGate : public NewGate {
 public:
  ArithmeticGMWOutputGate(std::size_t gate_id, GMWProvider&, ArithmeticGMWWireP<T>&&,
                          std::size_t output_owner);
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> get_output_future();
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;

 private:
  GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  std::size_t output_owner_;
  ENCRYPTO::ReusableFiberPromise<std::vector<T>> output_promise_;
  std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<T>>> share_futures_;
  const ArithmeticGMWWireP<T> input_;
};

template <typename T>
class ArithmeticGMWOutputShareGate : public NewGate {
 public:
  ArithmeticGMWOutputShareGate(std::size_t gate_id, ArithmeticGMWWireP<T>&&);
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> get_output_future();
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;

 private:
  ENCRYPTO::ReusableFiberPromise<std::vector<T>> output_promise_;
  const ArithmeticGMWWireP<T> input_;
};

template <typename T>
class ArithmeticGMWNEGGate : public detail::BasicArithmeticGMWUnaryGate<T> {
 public:
  ArithmeticGMWNEGGate(std::size_t gate_id, GMWProvider&, ArithmeticGMWWireP<T>&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
};

template <typename T>
class ArithmeticGMWADDGate : public detail::BasicArithmeticGMWBinaryGate<T> {
 public:
  ArithmeticGMWADDGate(std::size_t gate_id, GMWProvider&, ArithmeticGMWWireP<T>&&,
                       ArithmeticGMWWireP<T>&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
};

template <typename T>
class ArithmeticGMWMULGate : public detail::BasicArithmeticGMWBinaryGate<T> {
 public:
  ArithmeticGMWMULGate(std::size_t gate_id, GMWProvider&, ArithmeticGMWWireP<T>&&,
                       ArithmeticGMWWireP<T>&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;

 private:
  GMWProvider& gmw_provider_;
  std::size_t mt_offset_;
  std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<T>>> share_futures_;
};

template <typename T>
class ArithmeticGMWSQRGate : public detail::BasicArithmeticGMWUnaryGate<T> {
 public:
  ArithmeticGMWSQRGate(std::size_t gate_id, GMWProvider&,
                       ArithmeticGMWWireP<T>&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;

 private:
  GMWProvider& gmw_provider_;
  std::size_t sp_offset_;
  std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<T>>> share_futures_;
};

}  // namespace MOTION::proto::gmw
