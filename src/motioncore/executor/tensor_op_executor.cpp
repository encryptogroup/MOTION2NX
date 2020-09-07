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

#include "tensor_op_executor.h"

#include <fmt/format.h>
#include <omp.h>
#include <algorithm>
#include <iostream>

#include "base/gate_register.h"
#include "executor/execution_context.h"
#include "gate/new_gate.h"
#include "statistics/run_time_stats.h"
#include "utility/fiber_thread_pool/fiber_thread_pool.hpp"
#include "utility/logger.h"

namespace MOTION {

TensorOpExecutor::TensorOpExecutor(GateRegister& reg, std::function<void(void)> preprocessing_fctn,
                                   std::size_t num_threads, std::shared_ptr<Logger> logger)
    : register_(reg),
      preprocessing_fctn_(std::move(preprocessing_fctn)),
      num_threads_(num_threads),
      logger_(std::move(logger)) {
  if (num_threads_ == 0) {
    num_threads_ = std::thread::hardware_concurrency();
  }
}

void TensorOpExecutor::evaluate_setup_online(Statistics::RunTimeStats& stats) {
  if (num_threads_ > 0) {
    if (logger_) {
      logger_->LogInfo(fmt::format("Set OpenMP threads to {}", num_threads_));
    }
    omp_set_num_threads(num_threads_);
  }

  ExecutionContext exec_ctx{.num_threads_ = num_threads_,
                            .fpool_ = std::make_unique<ENCRYPTO::FiberThreadPool>(
                                std::max(std::size_t{2}, num_threads_))};

  stats.record_start<Statistics::RunTimeStats::StatID::evaluate>();

  preprocessing_fctn_();

  if (logger_) {
    logger_->LogInfo(
        "Start evaluating the circuit gates sequentially (online after all finished setup)");
  }

  // ------------------------------ setup phase ------------------------------
  stats.record_start<Statistics::RunTimeStats::StatID::gates_setup>();

  // evaluate the setup phase of all the gates
  for (auto& gate : register_.get_gates()) {
    if (gate->need_setup()) {
      gate->evaluate_setup_with_context(exec_ctx);
      register_.increment_gate_setup_counter();
    }
  }
  register_.wait_setup();

  stats.record_end<Statistics::RunTimeStats::StatID::gates_setup>();

  if (logger_) {
    logger_->LogInfo("Start with the online phase of the circuit gates");
  }

  // ------------------------------ online phase ------------------------------
  stats.record_start<Statistics::RunTimeStats::StatID::gates_online>();

  // evaluate the online phase of all the gates
  for (auto& gate : register_.get_gates()) {
    if (gate->need_online()) {
      gate->evaluate_online_with_context(exec_ctx);
      register_.increment_gate_online_counter();
    }
  }
  register_.wait_online();

  stats.record_end<Statistics::RunTimeStats::StatID::gates_online>();

  // --------------------------------------------------------------------------

  if (logger_) {
    logger_->LogInfo("Finished with the online phase of the circuit gates");
  }

  stats.record_end<Statistics::RunTimeStats::StatID::evaluate>();
  exec_ctx.fpool_->join();
}

void TensorOpExecutor::evaluate(Statistics::RunTimeStats& stats) {
  throw std::logic_error("not implemented");
}

}  // namespace MOTION
