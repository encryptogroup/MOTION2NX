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

#include "gate/new_gate.h"

namespace MOTION::proto::plain {
class BooleanPlainWire;
using BooleanPlainWireVector = std::vector<std::shared_ptr<BooleanPlainWire>>;
template <typename T>
class ArithmeticPlainWire;
template <typename T>
using ArithmeticPlainWireP = std::shared_ptr<ArithmeticPlainWire<T>>;
}  // namespace MOTION::proto::plain

namespace MOTION::proto::gmw {

class GMWProvider;
class BooleanGMWWire;
using BooleanGMWWireVector = std::vector<std::shared_ptr<BooleanGMWWire>>;
template <typename T>
class ArithmeticGMWWire;
template <typename T>
using ArithmeticGMWWireP = std::shared_ptr<ArithmeticGMWWire<T>>;

namespace detail {

class BasicBooleanGMWPlainBinaryGate : public NewGate {
 public:
  BasicBooleanGMWPlainBinaryGate(std::size_t gate_id, GMWProvider&, BooleanGMWWireVector&&,
                                 plain::BooleanPlainWireVector&&);
  BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; };

 protected:
  const GMWProvider& gmw_provider_;
  std::size_t num_wires_;
  const BooleanGMWWireVector inputs_gmw_;
  const plain::BooleanPlainWireVector inputs_plain_;
  BooleanGMWWireVector outputs_;
};

template <typename T>
class BasicArithmeticGMWPlainBinaryGate : public NewGate {
 public:
  BasicArithmeticGMWPlainBinaryGate(std::size_t gate_id, GMWProvider&, ArithmeticGMWWireP<T>&&,
                                    plain::ArithmeticPlainWireP<T>&&);
  ArithmeticGMWWireP<T>& get_output_wire() noexcept { return output_; };

 protected:
  const GMWProvider& gmw_provider_;
  const ArithmeticGMWWireP<T> input_gmw_;
  const plain::ArithmeticPlainWireP<T> input_plain_;
  ArithmeticGMWWireP<T> output_;
};

}  // namespace detail

class BooleanGMWXORPlainGate : public detail::BasicBooleanGMWPlainBinaryGate {
 public:
  using detail::BasicBooleanGMWPlainBinaryGate::BasicBooleanGMWPlainBinaryGate;
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
};

class BooleanGMWANDPlainGate : public detail::BasicBooleanGMWPlainBinaryGate {
 public:
  using detail::BasicBooleanGMWPlainBinaryGate::BasicBooleanGMWPlainBinaryGate;
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
};

template <typename T>
class ArithmeticGMWADDPlainGate : public detail::BasicArithmeticGMWPlainBinaryGate<T> {
 public:
  using detail::BasicArithmeticGMWPlainBinaryGate<T>::BasicArithmeticGMWPlainBinaryGate;
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
};

template <typename T>
class ArithmeticGMWMULPlainGate : public detail::BasicArithmeticGMWPlainBinaryGate<T> {
 public:
  using detail::BasicArithmeticGMWPlainBinaryGate<T>::BasicArithmeticGMWPlainBinaryGate;
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
};

}  // namespace MOTION::proto::gmw
