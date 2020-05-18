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
#include <memory>
#include <vector>

#include "gate/new_gate.h"
#include "utility/bit_vector.h"
#include "utility/reusable_future.h"

namespace MOTION::proto {

namespace gmw {
class BooleanGMWWire;
using BooleanGMWWireVector = std::vector<std::shared_ptr<BooleanGMWWire>>;
}  // namespace gmw

namespace yao {

class YaoProvider;
class YaoWire;
using YaoWireVector = std::vector<std::shared_ptr<YaoWire>>;

class YaoToBooleanGMWGateGarbler : public NewGate {
 public:
  YaoToBooleanGMWGateGarbler(std::size_t gate_id, YaoProvider&, YaoWireVector&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return false; }
  void evaluate_setup() override;
  void evaluate_online() override;
  gmw::BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; };

 private:
  const YaoWireVector inputs_;
  gmw::BooleanGMWWireVector outputs_;
};

class YaoToBooleanGMWGateEvaluator : public NewGate {
 public:
  YaoToBooleanGMWGateEvaluator(std::size_t gate_id, YaoProvider&, YaoWireVector&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  gmw::BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; };

 private:
  const YaoWireVector inputs_;
  gmw::BooleanGMWWireVector outputs_;
};

}  // namespace yao
}  // namespace MOTION::proto
