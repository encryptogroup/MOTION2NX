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

#include <type_traits>
#include <vector>

#include "gate/new_gate.h"
#include "utility/bit_vector.h"
#include "utility/reusable_future.h"
#include "utility/type_traits.hpp"

#include <csignal>

namespace MOTION {

template <typename InputGateClass, typename T>
class ArithmeticInputAdapterGate : public InputGateClass {
  using is_enabled_0_ = std::enable_if_t<std::is_base_of_v<NewGate, InputGateClass>>;
  using is_enabled_1_ = ENCRYPTO::is_unsigned_int_t<T>;

 public:
  ArithmeticInputAdapterGate(const ArithmeticInputAdapterGate&) = delete;
  template <typename... Args>
  ArithmeticInputAdapterGate(ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& int_vector_future,
                             Args&&... args)
      : InputGateClass(std::forward<Args>(args)...,
                       // ugly hack!
                       [this] {
                         ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> promise;
                         auto future = promise.get_future();
                         tmp_promise_ = std::move(promise);
                         return future;
                       }()),
        // another ugly hack!
        bit_vectors_promise_(std::move(tmp_promise_)),
        int_vector_future_(std::move(int_vector_future)) {}

  bool need_setup() const noexcept override { return InputGateClass::need_setup(); }
  bool need_online() const noexcept override { return InputGateClass::need_online(); }
  void evaluate_setup() override { InputGateClass::evaluate_setup(); }
  void evaluate_online() override {
    auto int_vector = int_vector_future_.get();
    bit_vectors_promise_.set_value(ENCRYPTO::ToInput(int_vector));
    InputGateClass::evaluate_online();
  }

 private:
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> bit_vectors_promise_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> int_vector_future_;
  // another ugly hack!
  static thread_local ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>>
      tmp_promise_;
};

template <typename InputGateClass, typename T>
thread_local ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>>
    ArithmeticInputAdapterGate<InputGateClass, T>::tmp_promise_;

}  // namespace MOTION
