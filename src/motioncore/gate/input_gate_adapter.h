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

#include <functional>
#include <type_traits>
#include <vector>

#include "gate/new_gate.h"
#include "utility/bit_vector.h"
#include "utility/constants.h"
#include "utility/logger.h"
#include "utility/reusable_future.h"
#include "utility/type_traits.hpp"

#include <csignal>

namespace MOTION {

template <typename InputGateClass, typename T>
class ArithmeticInputAdapterGate : public InputGateClass {
  using is_enabled_0_ = std::enable_if_t<std::is_base_of_v<NewGate, InputGateClass>>;
  using is_enabled_1_ = ENCRYPTO::is_unsigned_int_t<T>;
  using base_constructor_tag = struct {};

 public:
  template <typename... Args>
  ArithmeticInputAdapterGate(base_constructor_tag,
                             ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& int_vector_future_1,
                             ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& int_vector_future_2,
                             std::function<T(T)> unary_op, std::function<T(T, T)> binary_op,
                             std::shared_ptr<Logger> logger, Args&&... args)
      : InputGateClass(std::forward<Args>(args)...,
                       // ugly hack!
                       [this] {
                         ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> promise;
                         auto future = promise.get_future();
                         tmp_promise_ = std::move(promise);
                         return future;
                       }()),
        bit_vectors_promise_(std::move(tmp_promise_)),
        int_vector_future_1_(std::move(int_vector_future_1)),
        int_vector_future_2_(std::move(int_vector_future_2)),
        unary_op_(unary_op),
        binary_op_(binary_op),
        logger_(logger) {
    if constexpr (MOTION_VERBOSE_DEBUG) {
      if (logger_) {
        logger_->LogTrace(
            fmt::format("Gate {}: ArithmeticInputAdapterGate created", InputGateClass::gate_id_));
      }
    }
  }

  template <typename... Args>
  ArithmeticInputAdapterGate(ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& int_vector_future,
                             std::function<T(T)> unary_op, std::shared_ptr<Logger> logger,
                             Args&&... args)
      : ArithmeticInputAdapterGate(base_constructor_tag{}, std::move(int_vector_future),
                                   ENCRYPTO::ReusableFiberFuture<std::vector<T>>{}, unary_op,
                                   std::function<T(T, T)>{}, logger, std::forward<Args>(args)...) {}

  template <typename... Args>
  ArithmeticInputAdapterGate(ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& int_vector_future_1,
                             ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& int_vector_future_2,
                             std::function<T(T, T)> binary_op, std::shared_ptr<Logger> logger,
                             Args&&... args)
      : ArithmeticInputAdapterGate(base_constructor_tag{}, std::move(int_vector_future_1),
                                   std::move(int_vector_future_2), std::function<T(T)>{}, binary_op,
                                   logger, std::forward<Args>(args)...) {}

  bool need_setup() const noexcept override { return InputGateClass::need_setup(); }
  bool need_online() const noexcept override { return InputGateClass::need_online(); }

  void evaluate_setup() override {
    if constexpr (MOTION_VERBOSE_DEBUG) {
      if (logger_) {
        logger_->LogTrace(fmt::format("Gate {}: ArithmeticInputAdapterGate::evaluate_setup start",
                                      InputGateClass::gate_id_));
      }
    }

    InputGateClass::evaluate_setup();

    if constexpr (MOTION_VERBOSE_DEBUG) {
      if (logger_) {
        logger_->LogTrace(fmt::format("Gate {}: ArithmeticInputAdapterGate::evaluate_setup end",
                                      InputGateClass::gate_id_));
      }
    }
  }

  void evaluate_online() override {
    if constexpr (MOTION_VERBOSE_DEBUG) {
      if (logger_) {
        logger_->LogTrace(fmt::format("Gate {}: ArithmeticInputAdapterGate::evaluate_online start",
                                      InputGateClass::gate_id_));
      }
    }

    auto int_vector = int_vector_future_1_.get();
    if (int_vector_future_2_.valid()) {
      auto int_vector_2 = int_vector_future_2_.get();
      if (int_vector.size() != int_vector_2.size()) {
        throw std::logic_error("ArithmeticInputAdapterGate: expected int vectors of same size");
      }
      std::transform(std::begin(int_vector), std::end(int_vector), std::begin(int_vector_2),
                     std::begin(int_vector), binary_op_);
    } else {
      std::transform(std::begin(int_vector), std::end(int_vector), std::begin(int_vector),
                     unary_op_);
    }
    bit_vectors_promise_.set_value(ENCRYPTO::ToInput(int_vector));
    InputGateClass::evaluate_online();

    if constexpr (MOTION_VERBOSE_DEBUG) {
      if (logger_) {
        logger_->LogTrace(fmt::format("Gate {}: ArithmeticInputAdapterGate::evaluate_online end",
                                      InputGateClass::gate_id_));
      }
    }
  }

 private:
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> bit_vectors_promise_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> int_vector_future_1_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> int_vector_future_2_;
  std::function<T(T)> unary_op_;
  std::function<T(T, T)> binary_op_;
  std::shared_ptr<Logger> logger_;
  // another ugly hack!
  static thread_local ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>>
      tmp_promise_;
};

template <typename InputGateClass, typename T>
thread_local ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>>
    ArithmeticInputAdapterGate<InputGateClass, T>::tmp_promise_;

}  // namespace MOTION
