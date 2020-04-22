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

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "utility/enable_wait.h"

namespace MOTION {

class NewGate;

class GateRegister : public ENCRYPTO::enable_wait_setup, public ENCRYPTO::enable_wait_online {
 public:
  GateRegister();
  ~GateRegister();
  std::size_t get_next_gate_id() noexcept { return next_gate_id_++; }
  void register_gate(std::unique_ptr<NewGate>&& gate);
  void increment_gate_setup_counter() noexcept;
  void increment_gate_online_counter() noexcept;

  std::vector<std::unique_ptr<NewGate>>& get_gates() noexcept { return gates_; }
  const std::vector<std::unique_ptr<NewGate>>& get_gates() const noexcept { return gates_; }

 private:
  std::size_t next_gate_id_;
  std::size_t num_gates_with_setup_;
  std::size_t num_gates_with_online_;
  std::atomic<std::size_t> num_evaluated_setup_;
  std::atomic<std::size_t> num_evaluated_online_;
  std::vector<std::unique_ptr<NewGate>> gates_;
};

}  // namespace MOTION
