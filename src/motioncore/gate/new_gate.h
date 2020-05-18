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
#include <cstdint>

#include "utility/enable_wait.h"
#include "utility/fiber_condition.h"

namespace MOTION {

enum class MPCProtocol : unsigned int;

class NewGate : public ENCRYPTO::enable_wait_setup, public ENCRYPTO::enable_wait_online {
 public:
  virtual ~NewGate() = default;
  virtual bool need_setup() const noexcept = 0;
  virtual bool need_online() const noexcept = 0;
  virtual void evaluate_setup() = 0;
  virtual void evaluate_online() = 0;
  std::size_t get_gate_id() const noexcept { return gate_id_; }

 protected:
  NewGate(std::size_t gate_id) noexcept : gate_id_(gate_id) {}
  std::size_t gate_id_;
};

// Convert a GateClass to one that is evaluated purely during the setup phase.
template <typename GateClass>
class SetupGate : public GateClass {
  using is_enabled_ = std::enable_if_t<std::is_base_of_v<NewGate, GateClass>>;

 public:
  using GateClass::GateClass;
  bool need_setup() const noexcept override {
    return GateClass::need_setup() || GateClass::need_online();
  }
  bool need_online() const noexcept override { return false; }
  virtual void evaluate_setup() override {
    GateClass::evaluate_setup();
    GateClass::evaluate_online();
  }
  virtual void evaluate_online() override {}
};

}  // namespace MOTION
