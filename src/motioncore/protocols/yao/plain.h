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
}  // namespace MOTION::proto::plain

namespace MOTION::proto::yao {

class YaoProvider;
class YaoWire;
using YaoWireVector = std::vector<std::shared_ptr<YaoWire>>;

namespace detail {

class BasicYaoPlainBinaryGate : public NewGate {
 public:
  BasicYaoPlainBinaryGate(std::size_t gate_id, YaoProvider&, YaoWireVector&&,
                          plain::BooleanPlainWireVector&&);
  YaoWireVector& get_output_wires() noexcept { return outputs_; };

 protected:
  YaoProvider& yao_provider_;
  std::size_t num_wires_;
  const YaoWireVector inputs_yao_;
  const plain::BooleanPlainWireVector inputs_plain_;
  YaoWireVector outputs_;
};

}  // namespace detail

class YaoXORPlainGateGarbler : public detail::BasicYaoPlainBinaryGate {
 public:
  using detail::BasicYaoPlainBinaryGate::BasicYaoPlainBinaryGate;
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return false; }
  void evaluate_setup() override;
  void evaluate_online() override {}
};

class YaoXORPlainGateEvaluator : public detail::BasicYaoPlainBinaryGate {
 public:
  using detail::BasicYaoPlainBinaryGate::BasicYaoPlainBinaryGate;
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
};

class YaoANDPlainGateGarbler : public detail::BasicYaoPlainBinaryGate {
 public:
  using detail::BasicYaoPlainBinaryGate::BasicYaoPlainBinaryGate;
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return false; }
  void evaluate_setup() override;
  void evaluate_online() override {}
};

class YaoANDPlainGateEvaluator : public detail::BasicYaoPlainBinaryGate {
 public:
  using detail::BasicYaoPlainBinaryGate::BasicYaoPlainBinaryGate;
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
};

}  // namespace MOTION::proto::yao
