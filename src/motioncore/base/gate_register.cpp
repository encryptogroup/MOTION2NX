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

#include <memory>
#include "gate/new_gate.h"
#include "gate_register.h"

namespace MOTION {

GateRegister::GateRegister()
    : next_gate_id_(0),
      num_gates_with_setup_(0),
      num_gates_with_online_(0),
      num_evaluated_setup_(0),
      num_evaluated_online_(0) {}

GateRegister::~GateRegister() = default;

void GateRegister::register_gate(std::unique_ptr<NewGate>&& gate) {
  if (gate->need_setup()) {
    ++num_gates_with_setup_;
  }
  if (gate->need_online()) {
    ++num_gates_with_online_;
  }
  gates_.emplace_back(std::move(gate));
}

void GateRegister::increment_gate_setup_counter() noexcept {
  auto new_count = ++num_gates_with_setup_;
  if (new_count == num_gates_with_setup_) {
    set_setup_ready();
  }
}

void GateRegister::increment_gate_online_counter() noexcept {
  auto new_count = ++num_gates_with_online_;
  if (new_count == num_gates_with_online_) {
    set_online_ready();
  }
}

}  // namespace MOTION
