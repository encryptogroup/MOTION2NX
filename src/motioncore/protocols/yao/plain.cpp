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

#include "plain.h"

#include "protocols/plain/wire.h"
#include "utility/constants.h"
#include "utility/logger.h"
#include "wire.h"
#include "yao_provider.h"

namespace MOTION::proto::yao {

namespace detail {

BasicYaoPlainBinaryGate::BasicYaoPlainBinaryGate(std::size_t gate_id, YaoProvider& yao_provider,
                                                 YaoWireVector&& in_yao,
                                                 plain::BooleanPlainWireVector&& in_plain)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      num_wires_(in_yao.size()),
      inputs_yao_(std::move(in_yao)),
      inputs_plain_(std::move(in_plain)) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  if (num_wires_ != inputs_plain_.size()) {
    throw std::logic_error("number of wires need to be the same for both inputs");
  }
  const auto num_simd = inputs_yao_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    if (inputs_yao_[wire_i]->get_num_simd() != num_simd ||
        inputs_plain_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<YaoWire>(num_simd); });
}

}  // namespace detail

void YaoXORPlainGateGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoXORPlainGateGarbler::evaluate_setup start", gate_id_));
    }
  }

  const auto num_simd = inputs_yao_[0]->get_num_simd();
  const auto global_offset = yao_provider_.get_global_offset();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_yao = inputs_yao_[wire_i];
    const auto& wire_plain = inputs_plain_[wire_i];
    const auto& wire_out = outputs_[wire_i];
    wire_yao->wait_setup();
    wire_plain->wait_online();
    const auto& plain_data = wire_plain->get_data();
    const auto& in_keys = wire_yao->get_keys();
    auto& out_keys = wire_out->get_keys();
    assert(plain_data.GetSize() == num_simd);
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      if (plain_data.Get(simd_j)) {
        out_keys[simd_j] = in_keys[simd_j] ^ global_offset;
      } else {
        out_keys[simd_j] = in_keys[simd_j];
      }
    }
    wire_out->set_setup_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoXORPlainGateGarbler::evaluate_setup end", gate_id_));
    }
  }
}

void YaoXORPlainGateEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoXORPlainGateEvaluator::evaluate_online start", gate_id_));
    }
  }

  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_yao = inputs_yao_[wire_i];
    const auto& wire_out = outputs_[wire_i];
    wire_yao->wait_online();
    wire_out->get_keys() = wire_yao->get_keys();
    wire_out->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoXORPlainGateEvaluator::evaluate_online end", gate_id_));
    }
  }
}

void YaoANDPlainGateGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoANDPlainGateGarbler::evaluate_setup start", gate_id_));
    }
  }

  const auto num_simd = inputs_yao_[0]->get_num_simd();
  const auto shared_zero = yao_provider_.get_shared_zero();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_yao = inputs_yao_[wire_i];
    const auto& wire_plain = inputs_plain_[wire_i];
    const auto& wire_out = outputs_[wire_i];
    wire_yao->wait_setup();
    wire_plain->wait_online();
    const auto& plain_data = wire_plain->get_data();
    const auto& in_keys = wire_yao->get_keys();
    auto& out_keys = wire_out->get_keys();
    assert(plain_data.GetSize() == num_simd);
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      if (plain_data.Get(simd_j)) {
        out_keys[simd_j] = in_keys[simd_j];
      } else {
        out_keys[simd_j] = shared_zero;
      }
    }
    wire_out->set_setup_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoANDPlainGateGarbler::evaluate_setup end", gate_id_));
    }
  }
}

void YaoANDPlainGateEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoANDPlainGateEvaluator::evaluate_online start", gate_id_));
    }
  }

  const auto num_simd = inputs_yao_[0]->get_num_simd();
  const auto shared_zero = yao_provider_.get_shared_zero();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_yao = inputs_yao_[wire_i];
    const auto& wire_plain = inputs_plain_[wire_i];
    const auto& wire_out = outputs_[wire_i];
    wire_yao->wait_online();
    wire_plain->wait_online();
    const auto& plain_data = wire_plain->get_data();
    const auto& in_keys = wire_yao->get_keys();
    auto& out_keys = wire_out->get_keys();
    assert(plain_data.GetSize() == num_simd);
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      if (plain_data.Get(simd_j)) {
        out_keys[simd_j] = in_keys[simd_j];
      } else {
        out_keys[simd_j] = shared_zero;
      }
    }
    wire_out->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoANDPlainGateEvaluator::evaluate_online end", gate_id_));
    }
  }
}

}  // namespace MOTION::proto::yao
