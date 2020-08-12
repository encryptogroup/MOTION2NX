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

namespace MOTION::proto::beavy {

class BEAVYProvider;
class BooleanBEAVYWire;
using BooleanBEAVYWireVector = std::vector<std::shared_ptr<BooleanBEAVYWire>>;
template <typename T>
class ArithmeticBEAVYWire;
template <typename T>
using ArithmeticBEAVYWireP = std::shared_ptr<ArithmeticBEAVYWire<T>>;

namespace detail {

class BasicBooleanBEAVYPlainBinaryGate : public NewGate {
 public:
  BasicBooleanBEAVYPlainBinaryGate(std::size_t gate_id, BEAVYProvider&, BooleanBEAVYWireVector&&,
                                   plain::BooleanPlainWireVector&&);
  BooleanBEAVYWireVector& get_output_wires() noexcept { return outputs_; };

 protected:
  const BEAVYProvider& beavy_provider_;
  std::size_t num_wires_;
  const BooleanBEAVYWireVector inputs_beavy_;
  const plain::BooleanPlainWireVector inputs_plain_;
  BooleanBEAVYWireVector outputs_;
};

template <typename T>
class BasicArithmeticBEAVYPlainBinaryGate : public NewGate {
 public:
  BasicArithmeticBEAVYPlainBinaryGate(std::size_t gate_id, BEAVYProvider&,
                                      ArithmeticBEAVYWireP<T>&&, plain::ArithmeticPlainWireP<T>&&);
  ArithmeticBEAVYWireP<T>& get_output_wire() noexcept { return output_; };

 protected:
  const BEAVYProvider& beavy_provider_;
  const ArithmeticBEAVYWireP<T> input_beavy_;
  const plain::ArithmeticPlainWireP<T> input_plain_;
  ArithmeticBEAVYWireP<T> output_;
};

}  // namespace detail

class BooleanBEAVYXORPlainGate : public detail::BasicBooleanBEAVYPlainBinaryGate {
 public:
  using detail::BasicBooleanBEAVYPlainBinaryGate::BasicBooleanBEAVYPlainBinaryGate;
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
};

class BooleanBEAVYANDPlainGate : public detail::BasicBooleanBEAVYPlainBinaryGate {
 public:
  using detail::BasicBooleanBEAVYPlainBinaryGate::BasicBooleanBEAVYPlainBinaryGate;
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
};

template <typename T>
class ArithmeticBEAVYADDPlainGate : public detail::BasicArithmeticBEAVYPlainBinaryGate<T> {
 public:
  using detail::BasicArithmeticBEAVYPlainBinaryGate<T>::BasicArithmeticBEAVYPlainBinaryGate;
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
};

template <typename T>
class ArithmeticBEAVYMULPlainGate : public detail::BasicArithmeticBEAVYPlainBinaryGate<T> {
 public:
  using detail::BasicArithmeticBEAVYPlainBinaryGate<T>::BasicArithmeticBEAVYPlainBinaryGate;
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
};

}  // namespace MOTION::proto::beavy
