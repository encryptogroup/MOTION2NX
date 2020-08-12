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

#include <fmt/format.h>
#include <iterator>
#include <memory>

#include "gmw_provider.h"
#include "protocols/plain/wire.h"
#include "utility/constants.h"
#include "utility/logger.h"
#include "wire.h"

namespace MOTION::proto::gmw {

namespace detail {

BasicBooleanGMWPlainBinaryGate::BasicBooleanGMWPlainBinaryGate(
    std::size_t gate_id, GMWProvider& gmw_provider, BooleanGMWWireVector&& in_gmw,
    plain::BooleanPlainWireVector&& in_plain)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_wires_(in_gmw.size()),
      inputs_gmw_(std::move(in_gmw)),
      inputs_plain_(std::move(in_plain)) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  if (num_wires_ != inputs_plain_.size()) {
    throw std::logic_error("number of wires need to be the same for both inputs");
  }
  auto num_simd = inputs_gmw_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    if (inputs_gmw_[wire_i]->get_num_simd() != num_simd ||
        inputs_plain_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<BooleanGMWWire>(num_simd); });
}

template <typename T>
BasicArithmeticGMWPlainBinaryGate<T>::BasicArithmeticGMWPlainBinaryGate(
    std::size_t gate_id, GMWProvider& gmw_provider, ArithmeticGMWWireP<T>&& in_gmw,
    plain::ArithmeticPlainWireP<T>&& in_plain)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      input_gmw_(std::move(in_gmw)),
      input_plain_(std::move(in_plain)),
      output_(std::make_shared<ArithmeticGMWWire<T>>(input_gmw_->get_num_simd())) {
  if (input_gmw_->get_num_simd() != input_plain_->get_num_simd()) {
    throw std::logic_error("number of SIMD values need to be the same for all wires");
  }
}

template class BasicArithmeticGMWPlainBinaryGate<std::uint8_t>;
template class BasicArithmeticGMWPlainBinaryGate<std::uint16_t>;
template class BasicArithmeticGMWPlainBinaryGate<std::uint32_t>;
template class BasicArithmeticGMWPlainBinaryGate<std::uint64_t>;

}  // namespace detail

void BooleanGMWXORPlainGate::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWXORPlainGate::evaluate_online start", gate_id_));
    }
  }

  if (gmw_provider_.is_my_job(gate_id_)) {
    for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
      const auto& wire_gmw = inputs_gmw_[wire_i];
      const auto& wire_plain = inputs_plain_[wire_i];
      auto& wire_out = outputs_[wire_i];
      wire_gmw->wait_online();
      wire_plain->wait_online();
      wire_out->get_share() = wire_gmw->get_share() ^ wire_plain->get_data();
      wire_out->set_online_ready();
    }
  } else {
    for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
      const auto& wire_gmw = inputs_gmw_[wire_i];
      auto& wire_out = outputs_[wire_i];
      wire_gmw->wait_online();
      wire_out->get_share() = wire_gmw->get_share();
      wire_out->set_online_ready();
    }
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWXORPlainGate::evaluate_online end", gate_id_));
    }
  }
}

void BooleanGMWANDPlainGate::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWANDPlainGate::evaluate_online start", gate_id_));
    }
  }

  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_gmw = inputs_gmw_[wire_i];
    const auto& wire_plain = inputs_plain_[wire_i];
    auto& wire_out = outputs_[wire_i];
    wire_gmw->wait_online();
    wire_plain->wait_online();
    wire_out->get_share() = wire_gmw->get_share() & wire_plain->get_data();
    wire_out->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWANDPlainGate::evaluate_online end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticGMWADDPlainGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticGMWADDPlainGate<T>::evaluate_online start",
                                   this->gate_id_));
    }
  }

  if (this->gmw_provider_.is_my_job(this->gate_id_)) {
    this->input_gmw_->wait_online();
    this->input_plain_->wait_online();
    this->output_->get_share() =
        Helpers::AddVectors(this->input_gmw_->get_share(), this->input_plain_->get_data());
    this->output_->set_online_ready();
  } else {
    this->input_gmw_->wait_online();
    this->input_plain_->wait_online();
    this->output_->get_share() = this->input_gmw_->get_share();
    this->output_->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticGMWADDPlainGate<T>::evaluate_online end",
                                   this->gate_id_));
    }
  }
}

template class ArithmeticGMWADDPlainGate<std::uint8_t>;
template class ArithmeticGMWADDPlainGate<std::uint16_t>;
template class ArithmeticGMWADDPlainGate<std::uint32_t>;
template class ArithmeticGMWADDPlainGate<std::uint64_t>;

template <typename T>
void ArithmeticGMWMULPlainGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticGMWMULPlainGate<T>::evaluate_online start",
                                   this->gate_id_));
    }
  }

  this->input_gmw_->wait_online();
  this->input_plain_->wait_online();
  this->output_->get_share() =
      Helpers::MultiplyVectors(this->input_gmw_->get_share(), this->input_plain_->get_data());
  this->output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticGMWMULPlainGate<T>::evaluate_online end",
                                   this->gate_id_));
    }
  }
}

template class ArithmeticGMWMULPlainGate<std::uint8_t>;
template class ArithmeticGMWMULPlainGate<std::uint16_t>;
template class ArithmeticGMWMULPlainGate<std::uint32_t>;
template class ArithmeticGMWMULPlainGate<std::uint64_t>;

}  // namespace MOTION::proto::gmw
