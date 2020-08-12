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

#include "beavy_provider.h"
#include "protocols/plain/wire.h"
#include "utility/constants.h"
#include "utility/logger.h"
#include "wire.h"

namespace MOTION::proto::beavy {

namespace detail {

BasicBooleanBEAVYPlainBinaryGate::BasicBooleanBEAVYPlainBinaryGate(
    std::size_t gate_id, BEAVYProvider& beavy_provider, BooleanBEAVYWireVector&& in_beavy,
    plain::BooleanPlainWireVector&& in_plain)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      num_wires_(in_beavy.size()),
      inputs_beavy_(std::move(in_beavy)),
      inputs_plain_(std::move(in_plain)) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  if (num_wires_ != inputs_plain_.size()) {
    throw std::logic_error("number of wires need to be the same for both inputs");
  }
  auto num_simd = inputs_beavy_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    if (inputs_beavy_[wire_i]->get_num_simd() != num_simd ||
        inputs_plain_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<BooleanBEAVYWire>(num_simd); });
}

template <typename T>
BasicArithmeticBEAVYPlainBinaryGate<T>::BasicArithmeticBEAVYPlainBinaryGate(
    std::size_t gate_id, BEAVYProvider& beavy_provider, ArithmeticBEAVYWireP<T>&& in_beavy,
    plain::ArithmeticPlainWireP<T>&& in_plain)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      input_beavy_(std::move(in_beavy)),
      input_plain_(std::move(in_plain)),
      output_(std::make_shared<ArithmeticBEAVYWire<T>>(input_beavy_->get_num_simd())) {
  if (input_beavy_->get_num_simd() != input_plain_->get_num_simd()) {
    throw std::logic_error("number of SIMD values need to be the same for all wires");
  }
}

template class BasicArithmeticBEAVYPlainBinaryGate<std::uint8_t>;
template class BasicArithmeticBEAVYPlainBinaryGate<std::uint16_t>;
template class BasicArithmeticBEAVYPlainBinaryGate<std::uint32_t>;
template class BasicArithmeticBEAVYPlainBinaryGate<std::uint64_t>;

}  // namespace detail

void BooleanBEAVYXORPlainGate::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYXORPlainGate::evaluate_setup start", gate_id_));
    }
  }

  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_beavy = inputs_beavy_[wire_i];
    auto& wire_out = outputs_[wire_i];
    wire_beavy->wait_setup();
    wire_out->get_secret_share() = wire_beavy->get_secret_share();
    wire_out->set_setup_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYXORPlainGate::evaluate_setup end", gate_id_));
    }
  }
}

void BooleanBEAVYXORPlainGate::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYXORPlainGate::evaluate_online start", gate_id_));
    }
  }

  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_beavy = inputs_beavy_[wire_i];
    const auto& wire_plain = inputs_plain_[wire_i];
    auto& wire_out = outputs_[wire_i];
    wire_beavy->wait_online();
    wire_plain->wait_online();
    wire_out->get_public_share() = wire_beavy->get_public_share() ^ wire_plain->get_data();
    wire_out->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYXORPlainGate::evaluate_online end", gate_id_));
    }
  }
}

void BooleanBEAVYANDPlainGate::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYANDPlainGate::evaluate_setup start", gate_id_));
    }
  }

  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_beavy = inputs_beavy_[wire_i];
    const auto& wire_plain = inputs_plain_[wire_i];
    auto& wire_out = outputs_[wire_i];
    wire_beavy->wait_setup();
    wire_plain->wait_online();
    wire_out->get_secret_share() = wire_beavy->get_secret_share() & wire_plain->get_data();
    wire_out->set_setup_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYANDPlainGate::evaluate_setup end", gate_id_));
    }
  }
}

void BooleanBEAVYANDPlainGate::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYANDPlainGate::evaluate_online start", gate_id_));
    }
  }

  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_beavy = inputs_beavy_[wire_i];
    const auto& wire_plain = inputs_plain_[wire_i];
    auto& wire_out = outputs_[wire_i];
    wire_beavy->wait_online();
    wire_plain->wait_online();
    wire_out->get_public_share() = wire_beavy->get_public_share() & wire_plain->get_data();
    wire_out->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYANDPlainGate::evaluate_online end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYADDPlainGate<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYADDPlainGate<T>::evaluate_setup start",
                                   this->gate_id_));
    }
  }

  this->input_beavy_->wait_setup();
  this->output_->get_secret_share() = this->input_beavy_->get_secret_share();
  this->output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYADDPlainGate<T>::evaluate_setup end",
                                   this->gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYADDPlainGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYADDPlainGate<T>::evaluate_online start",
                                   this->gate_id_));
    }
  }

  this->input_beavy_->wait_online();
  this->input_plain_->wait_online();
  this->output_->get_public_share() =
      Helpers::AddVectors(this->input_beavy_->get_public_share(), this->input_plain_->get_data());
  this->output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYADDPlainGate<T>::evaluate_online end",
                                   this->gate_id_));
    }
  }
}

template class ArithmeticBEAVYADDPlainGate<std::uint8_t>;
template class ArithmeticBEAVYADDPlainGate<std::uint16_t>;
template class ArithmeticBEAVYADDPlainGate<std::uint32_t>;
template class ArithmeticBEAVYADDPlainGate<std::uint64_t>;

template <typename T>
void ArithmeticBEAVYMULPlainGate<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYMULPlainGate<T>::evaluate_setup start",
                                   this->gate_id_));
    }
  }

  this->input_beavy_->wait_setup();
  this->input_plain_->wait_online();
  this->output_->get_secret_share() = Helpers::MultiplyVectors(
      this->input_beavy_->get_secret_share(), this->input_plain_->get_data());
  this->output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYMULPlainGate<T>::evaluate_setup end",
                                   this->gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYMULPlainGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYMULPlainGate<T>::evaluate_online start",
                                   this->gate_id_));
    }
  }

  this->input_beavy_->wait_online();
  this->input_plain_->wait_online();
  this->output_->get_public_share() = Helpers::MultiplyVectors(
      this->input_beavy_->get_public_share(), this->input_plain_->get_data());
  this->output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = this->beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYMULPlainGate<T>::evaluate_online end",
                                   this->gate_id_));
    }
  }
}

template class ArithmeticBEAVYMULPlainGate<std::uint8_t>;
template class ArithmeticBEAVYMULPlainGate<std::uint16_t>;
template class ArithmeticBEAVYMULPlainGate<std::uint32_t>;
template class ArithmeticBEAVYMULPlainGate<std::uint64_t>;

}  // namespace MOTION::proto::beavy
