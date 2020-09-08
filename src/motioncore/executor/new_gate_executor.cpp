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

#include "new_gate_executor.h"

#include <boost/fiber/future/async.hpp>
#include <boost/fiber/policy.hpp>
#include <iostream>

#include "base/gate_register.h"
#include "gate/new_gate.h"
#include "statistics/run_time_stats.h"
#include "utility/fiber_thread_pool/fiber_thread_pool.hpp"
#include "utility/synchronized_queue.h"
#include "utility/logger.h"

namespace MOTION {

NewGateExecutor::NewGateExecutor(GateRegister& reg, std::function<void(void)> preprocessing_fctn,
                                 bool sync_between_setup_and_online,
                                 std::function<void(void)> sync_fctn, std::size_t num_threads,
                                 std::shared_ptr<Logger> logger)
    : register_(reg),
      preprocessing_fctn_(std::move(preprocessing_fctn)),
      sync_fctn_(std::move(sync_fctn)),
      num_threads_(num_threads),
      sync_between_setup_and_online_(sync_between_setup_and_online),
      logger_(std::move(logger)) {}

NewGateExecutor::NewGateExecutor(GateRegister& reg, std::function<void(void)> preprocessing_fctn,
                                 std::size_t num_threads, std::shared_ptr<Logger> logger)
    : NewGateExecutor(
          reg, std::move(preprocessing_fctn), false, [] {}, num_threads, std::move(logger)) {}


void NewGateExecutor::evaluate_setup_online(Statistics::RunTimeStats& stats) {
  if (num_threads_ == 1) {
    evaluate_setup_online_single_threaded(stats);
  } else {
    evaluate_setup_online_multi_threaded(stats);
  }
}

void NewGateExecutor::evaluate_setup_online_multi_threaded(Statistics::RunTimeStats& stats) {
  stats.record_start<Statistics::RunTimeStats::StatID::evaluate>();

  preprocessing_fctn_();

  if (logger_) {
    logger_->LogInfo(
        "Start evaluating the circuit gates sequentially (online after all finished setup)");
  }

  // create a pool to execute fibers
  ENCRYPTO::FiberThreadPool fpool(num_threads_, 2 * register_.get_num_gates());

  // ------------------------------ setup phase ------------------------------
  stats.record_start<Statistics::RunTimeStats::StatID::gates_setup>();

  if (register_.get_num_gates_with_setup()) {
    // evaluate the setup phase of all the gates
    for (auto& gate : register_.get_gates()) {
      if (gate->need_setup()) {
        fpool.post([&] {
          gate->evaluate_setup();
          register_.increment_gate_setup_counter();
        });
      }
    }
    register_.wait_setup();
  }

  stats.record_end<Statistics::RunTimeStats::StatID::gates_setup>();

  if (sync_between_setup_and_online_) {
    sync_fctn_();
  }

  if (logger_) {
    logger_->LogInfo("Start with the online phase of the circuit gates");
  }

  // ------------------------------ online phase ------------------------------
  stats.record_start<Statistics::RunTimeStats::StatID::gates_online>();

  if (register_.get_num_gates_with_online()) {
    // evaluate the online phase of all the gates
    for (auto& gate : register_.get_gates()) {
      if (gate->need_online()) {
        fpool.post([&] {
          gate->evaluate_online();
          register_.increment_gate_online_counter();
        });
      }
    }
    register_.wait_online();
  }

  stats.record_end<Statistics::RunTimeStats::StatID::gates_online>();

  // --------------------------------------------------------------------------

  if (logger_) {
    logger_->LogInfo("Finished with the online phase of the circuit gates");
  }

  fpool.join();

  stats.record_end<Statistics::RunTimeStats::StatID::evaluate>();
}

void NewGateExecutor::evaluate_setup_online_single_threaded(Statistics::RunTimeStats& stats) {
  stats.record_start<Statistics::RunTimeStats::StatID::evaluate>();

  preprocessing_fctn_();

  ENCRYPTO::SynchronizedFiberQueue<boost::fibers::fiber> cleanup_channel;
  auto cleanup_fut = boost::fibers::async([&cleanup_channel] {
    while (auto f = cleanup_channel.dequeue()) {
      f->join();
    }
  });

  if (logger_) {
    logger_->LogInfo(
        "Start evaluating the circuit gates sequentially (online after all finished setup) "
        "(single-threaded)");
  }

  // ------------------------------ setup phase ------------------------------
  stats.record_start<Statistics::RunTimeStats::StatID::gates_setup>();

  // evaluate the setup phase of all the gates
  for (auto& gate : register_.get_gates()) {
    if (gate->need_setup()) {
      cleanup_channel.enqueue(boost::fibers::fiber(boost::fibers::launch::dispatch, [&] {
        gate->evaluate_setup();
        register_.increment_gate_setup_counter();
      }));
    }
  }
  register_.wait_setup();

  stats.record_end<Statistics::RunTimeStats::StatID::gates_setup>();

  if (sync_between_setup_and_online_) {
    sync_fctn_();
  }

  if (logger_) {
    logger_->LogInfo("Start with the online phase of the circuit gates (single-threaded)");
  }

  // ------------------------------ online phase ------------------------------
  stats.record_start<Statistics::RunTimeStats::StatID::gates_online>();

  // evaluate the online phase of all the gates
  for (auto& gate : register_.get_gates()) {
    if (gate->need_online()) {
      cleanup_channel.enqueue(boost::fibers::fiber(boost::fibers::launch::dispatch, [&] {
        gate->evaluate_online();
        register_.increment_gate_online_counter();
      }));
    }
  }
  register_.wait_online();

  stats.record_end<Statistics::RunTimeStats::StatID::gates_online>();

  // --------------------------------------------------------------------------

  stats.record_end<Statistics::RunTimeStats::StatID::evaluate>();

  if (logger_) {
    logger_->LogInfo("Finished with the online phase of the circuit gates (single-threaded)");
  }

  cleanup_channel.close();
  cleanup_fut.get();
}

void NewGateExecutor::evaluate(Statistics::RunTimeStats&) {
  throw std::logic_error("not implemented");
}

}  // namespace MOTION
