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

#include "conversion.h"

#include <cstdint>

#include <fmt/format.h>

#include "beavy_provider.h"
#include "crypto/oblivious_transfer/ot_flavors.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "protocols/gmw/wire.h"
#include "utility/constants.h"
#include "utility/logger.h"

namespace MOTION::proto::beavy {

template <typename T>
BooleanBitToArithmeticBEAVYGate<T>::BooleanBitToArithmeticBEAVYGate(std::size_t gate_id,
                                                                    BEAVYProvider& beavy_provider,
                                                                    BooleanBEAVYWireP in)
    : NewGate(gate_id), input_(std::move(in)), beavy_provider_(beavy_provider) {
  const auto num_simd = input_->get_num_simd();
  output_ = std::make_shared<beavy::ArithmeticBEAVYWire<T>>(num_simd);
  const auto my_id = beavy_provider_.get_my_id();
  auto& ot_provider = beavy_provider_.get_ot_manager().get_provider(1 - my_id);
  if (my_id == 0) {
    ot_sender_ = ot_provider.RegisterSendACOT<T>(num_simd);
  } else {
    assert(my_id == 1);
    ot_receiver_ = ot_provider.RegisterReceiveACOT<T>(num_simd);
  }
  share_future_ = beavy_provider_.register_for_ints_message<T>(1 - my_id, gate_id_, num_simd);
}

template <typename T>
BooleanBitToArithmeticBEAVYGate<T>::~BooleanBitToArithmeticBEAVYGate() = default;

template <typename T>
void BooleanBitToArithmeticBEAVYGate<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanBitToArithmeticBEAVYGate<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto num_simd = input_->get_num_simd();

  output_->get_secret_share() = Helpers::RandomVector<T>(num_simd);
  output_->set_setup_ready();

  input_->wait_setup();
  const auto& secret_share = input_->get_secret_share();

  std::vector<T> ot_output;
  if (ot_sender_ != nullptr) {
    std::vector<T> correlations(num_simd);
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      if (secret_share.Get(simd_j)) {
        correlations[simd_j] = 1;
      }
    }
    ot_sender_->SetCorrelations(std::move(correlations));
    ot_sender_->SendMessages();
    ot_sender_->ComputeOutputs();
    ot_output = ot_sender_->GetOutputs();
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      T bit = secret_share.Get(simd_j);
      ot_output[simd_j] = bit + 2 * ot_output[simd_j];
    }
  } else {
    assert(ot_receiver_ != nullptr);
    ot_receiver_->SetChoices(secret_share);
    ot_receiver_->SendCorrections();
    ot_receiver_->ComputeOutputs();
    ot_output = ot_receiver_->GetOutputs();
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      T bit = secret_share.Get(simd_j);
      ot_output[simd_j] = bit - 2 * ot_output[simd_j];
    }
  }
  arithmetized_secret_share_ = std::move(ot_output);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBitToArithmeticBEAVYGate<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void BooleanBitToArithmeticBEAVYGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanBitToArithmeticBEAVYGate<T>::evaluate_online start", gate_id_));
    }
  }

  const auto num_simd = input_->get_num_simd();
  const auto my_id = beavy_provider_.get_my_id();
  std::vector<T> arithmetized_public_share(num_simd);
  input_->wait_online();
  const auto& public_share = input_->get_public_share();

  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    if (public_share.Get(simd_j)) {
      arithmetized_public_share[simd_j] = 1;
    }
  }

  const auto& secret_share = output_->get_secret_share();
  std::vector<T> tmp(num_simd);
  if (beavy_provider_.is_my_job(gate_id_)) {
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      const auto p = arithmetized_public_share[simd_j];
      const auto s = arithmetized_secret_share_[simd_j];
      const auto delta = secret_share[simd_j];
      tmp[simd_j] = p + (1 - 2 * p) * s + delta;
    }
  } else {
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      const auto p = arithmetized_public_share[simd_j];
      const auto s = arithmetized_secret_share_[simd_j];
      const auto delta = secret_share[simd_j];
      tmp[simd_j] = (1 - 2 * p) * s + delta;
    }
  }
  beavy_provider_.send_ints_message(1 - my_id, gate_id_, tmp);
  const auto other_share = share_future_.get();
  std::transform(std::begin(tmp), std::end(tmp), std::begin(other_share), std::begin(tmp),
                 std::plus{});
  output_->get_public_share() = std::move(tmp);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanBitToArithmeticBEAVYGate<T>::evaluate_online end", gate_id_));
    }
  }
}

template class BooleanBitToArithmeticBEAVYGate<std::uint8_t>;
template class BooleanBitToArithmeticBEAVYGate<std::uint16_t>;
template class BooleanBitToArithmeticBEAVYGate<std::uint32_t>;
template class BooleanBitToArithmeticBEAVYGate<std::uint64_t>;

template <typename T>
BooleanToArithmeticBEAVYGate<T>::BooleanToArithmeticBEAVYGate(std::size_t gate_id,
                                                              BEAVYProvider& beavy_provider,
                                                              BooleanBEAVYWireVector&& in)
    : NewGate(gate_id), inputs_(std::move(in)), beavy_provider_(beavy_provider) {
  const auto num_wires = inputs_.size();
  if (num_wires != ENCRYPTO::bit_size_v<T>) {
    throw std::logic_error("number of wires need to be equal to bit size of T");
  }
  const auto num_simd = inputs_.at(0)->get_num_simd();
  output_ = std::make_shared<beavy::ArithmeticBEAVYWire<T>>(num_simd);
  const auto my_id = beavy_provider_.get_my_id();
  auto& ot_provider = beavy_provider_.get_ot_manager().get_provider(1 - my_id);
  if (my_id == 0) {
    ot_sender_ = ot_provider.RegisterSendACOT<T>(num_wires * num_simd);
  } else {
    assert(my_id == 1);
    ot_receiver_ = ot_provider.RegisterReceiveACOT<T>(num_wires * num_simd);
  }
  share_future_ = beavy_provider_.register_for_ints_message<T>(1 - my_id, gate_id_, num_simd);
}

template <typename T>
BooleanToArithmeticBEAVYGate<T>::~BooleanToArithmeticBEAVYGate() = default;

template <typename T>
void BooleanToArithmeticBEAVYGate<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanToArithmeticBEAVYGate<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto num_wires = ENCRYPTO::bit_size_v<T>;
  const auto num_simd = output_->get_num_simd();

  output_->get_secret_share() = Helpers::RandomVector<T>(num_simd);
  output_->set_setup_ready();

  std::vector<T> ot_output;
  if (ot_sender_ != nullptr) {
    std::vector<T> correlations(num_wires * num_simd);
    for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
      const auto& wire_in = inputs_[wire_i];
      wire_in->wait_setup();
      const auto& secret_share = wire_in->get_secret_share();
      for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
        if (secret_share.Get(simd_j)) {
          correlations[wire_i * num_simd + simd_j] = 1;
        }
      }
    }
    ot_sender_->SetCorrelations(std::move(correlations));
    ot_sender_->SendMessages();
    ot_sender_->ComputeOutputs();
    ot_output = ot_sender_->GetOutputs();
    for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
      const auto& secret_share = inputs_[wire_i]->get_secret_share();
      for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
        T bit = secret_share.Get(simd_j);
        ot_output[wire_i * num_simd + simd_j] = bit + 2 * ot_output[wire_i * num_simd + simd_j];
      }
    }
  } else {
    assert(ot_receiver_ != nullptr);
    ENCRYPTO::BitVector<> choices;
    choices.Reserve(Helpers::Convert::BitsToBytes(num_wires * num_simd));
    for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
      const auto& wire_in = inputs_[wire_i];
      wire_in->wait_setup();
      choices.Append(wire_in->get_secret_share());
    }
    ot_receiver_->SetChoices(std::move(choices));
    ot_receiver_->SendCorrections();
    ot_receiver_->ComputeOutputs();
    ot_output = ot_receiver_->GetOutputs();
    for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
      const auto& secret_share = inputs_[wire_i]->get_secret_share();
      for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
        T bit = secret_share.Get(simd_j);
        ot_output[wire_i * num_simd + simd_j] = bit - 2 * ot_output[wire_i * num_simd + simd_j];
      }
    }
  }
  arithmetized_secret_share_ = std::move(ot_output);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanToArithmeticBEAVYGate<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void BooleanToArithmeticBEAVYGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanToArithmeticBEAVYGate<T>::evaluate_online start", gate_id_));
    }
  }

  const auto num_wires = ENCRYPTO::bit_size_v<T>;
  const auto num_simd = output_->get_num_simd();
  const auto my_id = beavy_provider_.get_my_id();
  std::vector<T> arithmetized_public_share(num_wires * num_simd);

  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire_in = inputs_[wire_i];
    wire_in->wait_online();
    const auto& public_share = wire_in->get_public_share();
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      if (public_share.Get(simd_j)) {
        arithmetized_public_share[wire_i * num_simd + simd_j] = 1;
      }
    }
  }

  auto tmp = output_->get_secret_share();
  if (beavy_provider_.is_my_job(gate_id_)) {
    for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
      for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
        const auto p = arithmetized_public_share[wire_i * num_simd + simd_j];
        const auto s = arithmetized_secret_share_[wire_i * num_simd + simd_j];
        tmp[simd_j] += (p + (1 - 2 * p) * s) << wire_i;
      }
    }
  } else {
    for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
      for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
        const auto p = arithmetized_public_share[wire_i * num_simd + simd_j];
        const auto s = arithmetized_secret_share_[wire_i * num_simd + simd_j];
        tmp[simd_j] += ((1 - 2 * p) * s) << wire_i;
      }
    }
  }
  beavy_provider_.send_ints_message(1 - my_id, gate_id_, tmp);
  const auto other_share = share_future_.get();
  std::transform(std::begin(tmp), std::end(tmp), std::begin(other_share), std::begin(tmp),
                 std::plus{});
  output_->get_public_share() = std::move(tmp);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanToArithmeticBEAVYGate<T>::evaluate_online end", gate_id_));
    }
  }
}

template class BooleanToArithmeticBEAVYGate<std::uint8_t>;
template class BooleanToArithmeticBEAVYGate<std::uint16_t>;
template class BooleanToArithmeticBEAVYGate<std::uint32_t>;
template class BooleanToArithmeticBEAVYGate<std::uint64_t>;

BooleanBEAVYToGMWGate::BooleanBEAVYToGMWGate(std::size_t gate_id, BEAVYProvider& beavy_provider,
                                             BooleanBEAVYWireVector&& in)
    : NewGate(gate_id), beavy_provider_(beavy_provider), inputs_(std::move(in)) {
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_.at(0)->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires,
                  [num_simd] { return std::make_shared<gmw::BooleanGMWWire>(num_simd); });
}

void BooleanBEAVYToGMWGate::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYToGMWGate::evaluate_online start", gate_id_));
    }
  }

  const auto num_wires = inputs_.size();
  if (beavy_provider_.is_my_job(gate_id_)) {
    for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
      const auto& wire_in = inputs_[wire_i];
      auto& wire_out = outputs_[wire_i];
      wire_in->wait_setup();
      wire_in->wait_online();
      wire_out->get_share() = wire_in->get_public_share() ^ wire_in->get_secret_share();
      wire_out->set_online_ready();
    }
  } else {
    for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
      const auto& wire_in = inputs_[wire_i];
      auto& wire_out = outputs_[wire_i];
      wire_in->wait_setup();
      wire_in->wait_online();
      wire_out->get_share() = wire_in->get_secret_share();
      wire_out->set_online_ready();
    }
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYToGMWGate::evaluate_online end", gate_id_));
    }
  }
}

BooleanGMWToBEAVYGate::BooleanGMWToBEAVYGate(std::size_t gate_id, BEAVYProvider& beavy_provider,
                                             gmw::BooleanGMWWireVector&& in)
    : NewGate(gate_id), beavy_provider_(beavy_provider), inputs_(std::move(in)) {
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_.at(0)->get_num_simd();
  const auto my_id = beavy_provider_.get_my_id();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires,
                  [num_simd] { return std::make_shared<BooleanBEAVYWire>(num_simd); });
  share_future_ =
      beavy_provider_.register_for_bits_message(1 - my_id, gate_id_, num_wires * num_simd);
}

void BooleanGMWToBEAVYGate::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWToBEAVYGate::evaluate_setup start", gate_id_));
    }
  }

  const auto num_simd = inputs_.at(0)->get_num_simd();
  for (auto& wire_out : outputs_) {
    wire_out->get_secret_share() = ENCRYPTO::BitVector<>::Random(num_simd);
    wire_out->set_setup_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: BooleanGMWToBEAVYGate::evaluate_setup end", gate_id_));
    }
  }
}

void BooleanGMWToBEAVYGate::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWToBEAVYGate::evaluate_online start", gate_id_));
    }
  }

  const auto my_id = beavy_provider_.get_my_id();
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_.at(0)->get_num_simd();
  ENCRYPTO::BitVector<> my_share;
  my_share.Reserve(Helpers::Convert::BitsToBytes(num_wires * num_simd));
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire_in = inputs_[wire_i];
    const auto& wire_out = outputs_[wire_i];
    wire_in->wait_online();
    my_share.Append(wire_in->get_share() ^ wire_out->get_secret_share());
  }
  beavy_provider_.send_bits_message(1 - my_id, gate_id_, my_share);
  my_share ^= share_future_.get();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire_out = outputs_[wire_i];
    wire_out->get_public_share() = my_share.Subset(wire_i * num_simd, (wire_i + 1) * num_simd);
    wire_out->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWToBEAVYGate::evaluate_online end", gate_id_));
    }
  }
}

template <typename T>
ArithmeticBEAVYToGMWGate<T>::ArithmeticBEAVYToGMWGate(std::size_t gate_id,
                                                      BEAVYProvider& beavy_provider,
                                                      ArithmeticBEAVYWireP<T> in)
    : NewGate(gate_id), beavy_provider_(beavy_provider), input_(std::move(in)) {
  const auto num_simd = input_->get_num_simd();
  output_ = std::make_shared<gmw::ArithmeticGMWWire<T>>(num_simd);
  output_->get_share().resize(num_simd);
}

template <typename T>
void ArithmeticBEAVYToGMWGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYToGMWGate<T>::evaluate_online start", gate_id_));
    }
  }

  if (beavy_provider_.is_my_job(gate_id_)) {
    input_->wait_setup();
    input_->wait_online();
    const auto& pshare = input_->get_public_share();
    const auto& sshare = input_->get_secret_share();
    std::transform(std::begin(pshare), std::end(pshare), std::begin(sshare),
                   std::begin(output_->get_share()), std::minus{});
    output_->set_online_ready();
  } else {
    input_->wait_setup();
    const auto& sshare = input_->get_secret_share();
    std::transform(std::begin(sshare), std::end(sshare), std::begin(output_->get_share()),
                   std::negate{});
    output_->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYToGMWGate<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYToGMWGate<std::uint8_t>;
template class ArithmeticBEAVYToGMWGate<std::uint16_t>;
template class ArithmeticBEAVYToGMWGate<std::uint32_t>;
template class ArithmeticBEAVYToGMWGate<std::uint64_t>;

template <typename T>
ArithmeticGMWToBEAVYGate<T>::ArithmeticGMWToBEAVYGate(std::size_t gate_id,
                                                      BEAVYProvider& beavy_provider,
                                                      gmw::ArithmeticGMWWireP<T> in)
    : NewGate(gate_id), beavy_provider_(beavy_provider), input_(std::move(in)) {
  const auto num_simd = input_->get_num_simd();
  const auto my_id = beavy_provider_.get_my_id();
  output_ = std::make_shared<ArithmeticBEAVYWire<T>>(num_simd);
  share_future_ = beavy_provider_.register_for_ints_message<T>(1 - my_id, gate_id_, num_simd);
}

template <typename T>
void ArithmeticGMWToBEAVYGate<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWToBEAVYGate<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto num_simd = input_->get_num_simd();
  output_->get_secret_share() = Helpers::RandomVector<T>(num_simd);
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWToBEAVYGate<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticGMWToBEAVYGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWToBEAVYGate<T>::evaluate_online start", gate_id_));
    }
  }

  const auto my_id = beavy_provider_.get_my_id();
  input_->wait_online();
  auto my_share = input_->get_share();
  std::transform(std::begin(my_share), std::end(my_share), std::begin(output_->get_secret_share()),
                 std::begin(my_share), std::plus{});
  beavy_provider_.send_ints_message(1 - my_id, gate_id_, my_share);
  const auto other_share = share_future_.get();
  std::transform(std::begin(my_share), std::end(my_share), std::begin(other_share),
                 std::begin(my_share), std::plus{});
  output_->get_public_share() = std::move(my_share);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWToBEAVYGate<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWToBEAVYGate<std::uint8_t>;
template class ArithmeticGMWToBEAVYGate<std::uint16_t>;
template class ArithmeticGMWToBEAVYGate<std::uint32_t>;
template class ArithmeticGMWToBEAVYGate<std::uint64_t>;

}  // namespace MOTION::proto::beavy
