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

#include "gate.h"
#include <openssl/bn.h>
#include <stdexcept>

#include "base/gate_factory.h"
#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/mt_provider.h"
#include "crypto/multiplication_triple/sp_provider.h"
#include "crypto/sharing_randomness_generator.h"
#include "gmw_provider.h"
#include "utility/helpers.h"
#include "utility/logger.h"
#include "wire.h"

namespace MOTION::proto::gmw {

namespace detail {

BasicBooleanGMWBinaryGate::BasicBooleanGMWBinaryGate(std::size_t gate_id,
                                                     BooleanGMWWireVector&& in_b,
                                                     BooleanGMWWireVector&& in_a)
    : NewGate(gate_id),
      num_wires_(in_a.size()),
      inputs_a_(std::move(in_a)),
      inputs_b_(std::move(in_b)) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  if (num_wires_ != inputs_b_.size()) {
    throw std::logic_error("number of wires need to be the same for both inputs");
  }
  auto num_simd = inputs_a_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    if (inputs_a_[wire_i]->get_num_simd() != num_simd ||
        inputs_b_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<BooleanGMWWire>(num_simd); });
}

BasicBooleanGMWUnaryGate::BasicBooleanGMWUnaryGate(std::size_t gate_id, BooleanGMWWireVector&& in,
                                                   bool forward)
    : NewGate(gate_id), num_wires_(in.size()), inputs_(std::move(in)) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    if (inputs_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  if (forward) {
    outputs_ = inputs_;
  } else {
    outputs_.reserve(num_wires_);
    std::generate_n(std::back_inserter(outputs_), num_wires_,
                    [num_simd] { return std::make_shared<BooleanGMWWire>(num_simd); });
  }
}

}  // namespace detail

BooleanGMWInputGateSender::BooleanGMWInputGateSender(
    std::size_t gate_id, GMWProvider& gmw_provider, std::size_t num_wires, std::size_t num_simd,
    ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&& input_future)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_wires_(num_wires),
      num_simd_(num_simd),
      input_id_(gmw_provider.get_next_input_id(num_wires)),
      input_future_(std::move(input_future)) {
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<BooleanGMWWire>(num_simd); });
}

void BooleanGMWInputGateSender::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_setup start", gate_id_));
    }
  }

  auto my_id = gmw_provider_.get_my_id();
  auto num_parties = gmw_provider_.get_num_parties();
  auto& mbp = gmw_provider_.get_motion_base_provider();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    auto& wire = outputs_[wire_i];
    wire->get_share().Resize(num_simd_);
    for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
      if (party_id == my_id) {
        continue;
      }
      auto& rng = mbp.get_my_randomness_generator(party_id);
      wire->get_share() ^= rng.GetBits(input_id_ + wire_i, num_simd_);
    }
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_setup end", gate_id_));
    }
  }
}

void BooleanGMWInputGateSender::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_online start", gate_id_));
    }
  }

  // wait for input value
  const auto inputs = input_future_.get();

  // compute my share
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    auto& w_o = outputs_[wire_i];
    auto& share = w_o->get_share();
    const auto& input_bits = inputs.at(wire_i);
    if (input_bits.GetSize() != num_simd_) {
      throw std::runtime_error("size of input bit vector != num_simd_");
    }
    share ^= input_bits;
    w_o->set_online_ready();
    assert(w_o->get_share().GetSize() == num_simd_);
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_online end", gate_id_));
    }
  }
}

BooleanGMWInputGateReceiver::BooleanGMWInputGateReceiver(std::size_t gate_id,
                                                         GMWProvider& gmw_provider,
                                                         std::size_t num_wires,
                                                         std::size_t num_simd,
                                                         std::size_t input_owner)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_wires_(num_wires),
      num_simd_(num_simd),
      input_owner_(input_owner),
      input_id_(gmw_provider.get_next_input_id(num_wires)) {
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<BooleanGMWWire>(num_simd); });
}

void BooleanGMWInputGateReceiver::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateReceiver::evaluate_setup start", gate_id_));
    }
  }

  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& rng = mbp.get_their_randomness_generator(input_owner_);
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    auto& wire = outputs_[wire_i];
    wire->get_share() = rng.GetBits(input_id_ + wire_i, num_simd_);
    assert(wire->get_share().GetSize() == num_simd_);
    wire->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateReceiver::evaluate_setup end", gate_id_));
    }
  }
}

// Determine the total number of bits in a collection of wires.
static std::size_t count_bits(const BooleanGMWWireVector& wires) {
  return std::transform_reduce(std::begin(wires), std::end(wires), 0, std::plus<>(),
                               [](const auto& a) { return a->get_num_simd(); });
}

BooleanGMWOutputGate::BooleanGMWOutputGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                           BooleanGMWWireVector&& inputs, std::size_t output_owner)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_wires_(inputs.size()),
      output_owner_(output_owner),
      inputs_(std::move(inputs)) {
  std::size_t my_id = gmw_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    share_futures_ = gmw_provider_.register_for_bits_messages(gate_id_, count_bits(inputs_));
  }
}

ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>
BooleanGMWOutputGate::get_output_future() {
  std::size_t my_id = gmw_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    return output_promise_.get_future();
  } else {
    throw std::logic_error("not this parties output");
  }
}

void BooleanGMWOutputGate::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWOutputGate::evaluate_online start", gate_id_));
    }
  }

  std::size_t my_id = gmw_provider_.get_my_id();
  ENCRYPTO::BitVector<> my_share;
  // auto num_bits = count_bits(inputs_);  // TODO: reserve
  for (const auto& wire : inputs_) {
    wire->wait_online();
    my_share.Append(wire->get_share());
  }
  if (output_owner_ != my_id) {
    if (output_owner_ == ALL_PARTIES) {
      gmw_provider_.broadcast_bits_message(gate_id_, my_share);
    } else {
      gmw_provider_.send_bits_message(output_owner_, gate_id_, my_share);
    }
  }
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    std::size_t num_parties = gmw_provider_.get_num_parties();
    for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
      if (party_id == my_id) {
        continue;
      }
      const auto other_share = share_futures_[party_id].get();
      my_share ^= other_share;
    }
    std::vector<ENCRYPTO::BitVector<>> outputs;
    outputs.reserve(num_wires_);
    std::size_t bit_offset = 0;
    for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
      auto num_simd = inputs_[wire_i]->get_num_simd();
      outputs.push_back(my_share.Subset(bit_offset, bit_offset + num_simd));
      bit_offset += num_simd;
    }
    output_promise_.set_value(std::move(outputs));
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: BooleanGMWOutputGate::evaluate_online end", gate_id_));
    }
  }
}

BooleanGMWINVGate::BooleanGMWINVGate(std::size_t gate_id, const GMWProvider& gmw_provider,
                                     BooleanGMWWireVector&& in)
    : detail::BasicBooleanGMWUnaryGate(gate_id, std::move(in), !gmw_provider.is_my_job(gate_id)),
      is_my_job_(gmw_provider.is_my_job(gate_id)) {}

void BooleanGMWINVGate::evaluate_online() {
  if (!is_my_job_) {
    return;
  }

  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& w_in = inputs_[wire_i];
    w_in->wait_online();
    auto& w_o = outputs_[wire_i];
    w_o->get_share() = ~w_in->get_share();
    w_o->set_online_ready();
  }
}

void BooleanGMWXORGate::evaluate_online() {
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& w_a = inputs_a_[wire_i];
    const auto& w_b = inputs_b_[wire_i];
    w_a->wait_online();
    w_b->wait_online();
    auto& w_o = outputs_[wire_i];
    w_o->get_share() = w_a->get_share() ^ w_b->get_share();
    assert(w_o->get_share().GetSize() == w_a->get_num_simd());
    w_o->set_online_ready();
  }
}

BooleanGMWANDGate::BooleanGMWANDGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                     BooleanGMWWireVector&& in_a, BooleanGMWWireVector&& in_b)
    : detail::BasicBooleanGMWBinaryGate(gate_id, std::move(in_a), std::move(in_b)),
      gmw_provider_(gmw_provider) {
  auto num_wires = inputs_a_.size();
  auto num_simd = inputs_a_.at(0)->get_num_simd();
  mt_offset_ = gmw_provider_.get_mt_provider().RequestBinaryMTs(num_wires * num_simd);
  share_futures_ = gmw_provider_.register_for_bits_messages(gate_id_, 2 * count_bits(inputs_a_));
}

void BooleanGMWANDGate::evaluate_online() {
  auto num_simd = inputs_a_[0]->get_num_simd();
  auto num_bits = num_wires_ * num_simd;
  const auto& mtp = gmw_provider_.get_mt_provider();
  auto mts = mtp.GetBinary(mt_offset_, num_bits);
  ENCRYPTO::BitVector<> x;
  ENCRYPTO::BitVector<> y;
  x.Reserve(Helpers::Convert::BitsToBytes(num_bits));
  y.Reserve(Helpers::Convert::BitsToBytes(num_bits));

  // collect all shares into a single buffer
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& wire_x = inputs_a_[wire_i];
    wire_x->wait_online();
    assert(wire_x->get_share().GetSize() == num_simd);
    x.Append(wire_x->get_share());
    const auto& wire_y = inputs_b_[wire_i];
    wire_y->wait_online();
    assert(wire_y->get_share().GetSize() == num_simd);
    y.Append(wire_y->get_share());
  }
  // mask values with a, b
  auto de = x ^ mts.a;
  de.Append(y ^ mts.b);
  gmw_provider_.broadcast_bits_message(gate_id_, de);
  // compute d, e
  auto num_parties = gmw_provider_.get_num_parties();
  auto my_id = gmw_provider_.get_my_id();
  for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
    if (party_id == my_id) {
      continue;
    }
    de ^= share_futures_[party_id].get();
  }
  auto e = de.Subset(num_bits, 2 * num_bits);
  auto d = std::move(de);
  d.Resize(num_bits);
  x &= e;  // x & e
  y &= d;  // y & d
  d &= e;  // d & e
  auto result = std::move(mts.c);
  result ^= x;
  result ^= y;
  if (gmw_provider_.is_my_job(gate_id_)) {
    result ^= d;
  }

  // distribute data among wires
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    auto& wire_o = outputs_[wire_i];
    wire_o->get_share() = result.Subset(wire_i * num_simd, (wire_i + 1) * num_simd);
    assert(wire_o->get_share().GetSize() == num_simd);
    wire_o->set_online_ready();
  }
}

namespace detail {

template <typename T>
BasicArithmeticGMWBinaryGate<T>::BasicArithmeticGMWBinaryGate(std::size_t gate_id, GMWProvider&,
                                                              ArithmeticGMWWireP<T>&& in_a,
                                                              ArithmeticGMWWireP<T>&& in_b)
    : NewGate(gate_id),
      input_a_(std::move(in_a)),
      input_b_(std::move(in_b)),
      output_(std::make_shared<ArithmeticGMWWire<T>>(input_a_->get_num_simd())) {
  if (input_a_->get_num_simd() != input_b_->get_num_simd()) {
    throw std::logic_error("number of SIMD values need to be the same for all wires");
  }
}

template class BasicArithmeticGMWBinaryGate<std::uint8_t>;
template class BasicArithmeticGMWBinaryGate<std::uint16_t>;
template class BasicArithmeticGMWBinaryGate<std::uint32_t>;
template class BasicArithmeticGMWBinaryGate<std::uint64_t>;

template <typename T>
BasicArithmeticGMWUnaryGate<T>::BasicArithmeticGMWUnaryGate(std::size_t gate_id, GMWProvider&,
                                                            ArithmeticGMWWireP<T>&& in)
    : NewGate(gate_id),
      input_(std::move(in)),
      output_(std::make_shared<ArithmeticGMWWire<T>>(input_->get_num_simd())) {}

template class BasicArithmeticGMWUnaryGate<std::uint8_t>;
template class BasicArithmeticGMWUnaryGate<std::uint16_t>;
template class BasicArithmeticGMWUnaryGate<std::uint32_t>;
template class BasicArithmeticGMWUnaryGate<std::uint64_t>;

}  // namespace detail

template <typename T>
ArithmeticGMWInputGateSender<T>::ArithmeticGMWInputGateSender(
    std::size_t gate_id, GMWProvider& gmw_provider, std::size_t num_simd,
    ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& input_future)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_simd_(num_simd),
      input_id_(gmw_provider.get_next_input_id(1)),
      input_future_(std::move(input_future)),
      output_(std::make_shared<ArithmeticGMWWire<T>>(num_simd)) {
  output_->get_share().resize(num_simd, 0);
}

template <typename T>
void ArithmeticGMWInputGateSender<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateSender<T>::evaluate_setup start", gate_id_));
    }
  }

  auto my_id = gmw_provider_.get_my_id();
  auto num_parties = gmw_provider_.get_num_parties();
  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& share = output_->get_share();
  for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
    if (party_id == my_id) {
      continue;
    }
    auto& rng = mbp.get_my_randomness_generator(party_id);
    std::transform(std::begin(share), std::end(share),
                   std::begin(rng.GetUnsigned<T>(input_id_, num_simd_)), std::begin(share),
                   std::minus{});
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateSender<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticGMWInputGateSender<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateSender<T>::evaluate_online start", gate_id_));
    }
  }

  // wait for input value
  const auto input = input_future_.get();
  if (input.size() != num_simd_) {
    throw std::runtime_error("size of input bit vector != num_simd_");
  }

  // compute my share
  auto& share = output_->get_share();
  std::transform(std::begin(share), std::end(share), std::begin(input), std::begin(share),
                 std::plus{});
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWInputGateSender::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWInputGateSender<std::uint8_t>;
template class ArithmeticGMWInputGateSender<std::uint16_t>;
template class ArithmeticGMWInputGateSender<std::uint32_t>;
template class ArithmeticGMWInputGateSender<std::uint64_t>;

template <typename T>
ArithmeticGMWInputGateReceiver<T>::ArithmeticGMWInputGateReceiver(std::size_t gate_id,
                                                                  GMWProvider& gmw_provider,
                                                                  std::size_t num_simd,
                                                                  std::size_t input_owner)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_simd_(num_simd),
      input_owner_(input_owner),
      input_id_(gmw_provider.get_next_input_id(1)),
      output_(std::make_shared<ArithmeticGMWWire<T>>(num_simd)) {}

template <typename T>
void ArithmeticGMWInputGateReceiver<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateReceiver::evaluate_setup start", gate_id_));
    }
  }

  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& rng = mbp.get_their_randomness_generator(input_owner_);
  output_->get_share() = rng.GetUnsigned<T>(input_id_, num_simd_);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWInputGateReceiver::evaluate_setup end", gate_id_));
    }
  }
}

template class ArithmeticGMWInputGateReceiver<std::uint8_t>;
template class ArithmeticGMWInputGateReceiver<std::uint16_t>;
template class ArithmeticGMWInputGateReceiver<std::uint32_t>;
template class ArithmeticGMWInputGateReceiver<std::uint64_t>;

template <typename T>
ArithmeticGMWOutputGate<T>::ArithmeticGMWOutputGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                                    ArithmeticGMWWireP<T>&& input,
                                                    std::size_t output_owner)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      output_owner_(output_owner),
      input_(std::move(input)) {
  std::size_t my_id = gmw_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    share_futures_ = gmw_provider_.register_for_ints_messages<T>(gate_id_, input_->get_num_simd());
  }
}

template <typename T>
ENCRYPTO::ReusableFiberFuture<std::vector<T>> ArithmeticGMWOutputGate<T>::get_output_future() {
  std::size_t my_id = gmw_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    return output_promise_.get_future();
  } else {
    throw std::logic_error("not this parties output");
  }
}

template <typename T>
void ArithmeticGMWOutputGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWOutputGate<T>::evaluate_online start", gate_id_));
    }
  }

  std::size_t my_id = gmw_provider_.get_my_id();
  input_->wait_online();
  auto my_share = input_->get_share();
  if (output_owner_ != my_id) {
    if (output_owner_ == ALL_PARTIES) {
      gmw_provider_.broadcast_ints_message(gate_id_, my_share);
    } else {
      gmw_provider_.send_ints_message(output_owner_, gate_id_, my_share);
    }
  }
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    std::size_t num_parties = gmw_provider_.get_num_parties();
    for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
      if (party_id == my_id) {
        continue;
      }
      const auto other_share = share_futures_[party_id].get();
      std::transform(std::begin(my_share), std::end(my_share), std::begin(other_share),
                     std::begin(my_share), std::plus{});
    }
    output_promise_.set_value(std::move(my_share));
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWOutputGate<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWOutputGate<std::uint8_t>;
template class ArithmeticGMWOutputGate<std::uint16_t>;
template class ArithmeticGMWOutputGate<std::uint32_t>;
template class ArithmeticGMWOutputGate<std::uint64_t>;

template <typename T>
ArithmeticGMWOutputShareGate<T>::ArithmeticGMWOutputShareGate(std::size_t gate_id,
                                                    ArithmeticGMWWireP<T>&& input)
    : NewGate(gate_id),
      input_(std::move(input)) {
}

template <typename T>
ENCRYPTO::ReusableFiberFuture<std::vector<T>> ArithmeticGMWOutputShareGate<T>::get_output_future() {
  return output_promise_.get_future();
}

template <typename T>
void ArithmeticGMWOutputShareGate<T>::evaluate_online() {
  input_->wait_online();
  output_promise_.set_value(input_->get_share());
}

template class ArithmeticGMWOutputShareGate<std::uint8_t>;
template class ArithmeticGMWOutputShareGate<std::uint16_t>;
template class ArithmeticGMWOutputShareGate<std::uint32_t>;
template class ArithmeticGMWOutputShareGate<std::uint64_t>;

template <typename T>
ArithmeticGMWNEGGate<T>::ArithmeticGMWNEGGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                              ArithmeticGMWWireP<T>&& in)
    : detail::BasicArithmeticGMWUnaryGate<T>(gate_id, gmw_provider, std::move(in)) {
  this->output_->get_share().resize(this->input_->get_num_simd());
}

template <typename T>
void ArithmeticGMWNEGGate<T>::evaluate_online() {
  this->input_->wait_online();
  assert(this->output_->get_share().size() == this->input_->get_num_simd());
  std::transform(std::begin(this->input_->get_share()), std::end(this->input_->get_share()),
                 std::begin(this->output_->get_share()), std::negate{});
  this->output_->set_online_ready();
}

template class ArithmeticGMWNEGGate<std::uint8_t>;
template class ArithmeticGMWNEGGate<std::uint16_t>;
template class ArithmeticGMWNEGGate<std::uint32_t>;
template class ArithmeticGMWNEGGate<std::uint64_t>;

template <typename T>
ArithmeticGMWADDGate<T>::ArithmeticGMWADDGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                              ArithmeticGMWWireP<T>&& in_a,
                                              ArithmeticGMWWireP<T>&& in_b)
    : detail::BasicArithmeticGMWBinaryGate<T>(gate_id, gmw_provider, std::move(in_a),
                                              std::move(in_b)) {
  this->output_->get_share().resize(this->input_a_->get_num_simd());
}

template <typename T>
void ArithmeticGMWADDGate<T>::evaluate_online() {
  this->input_a_->wait_online();
  this->input_b_->wait_online();
  assert(this->output_->get_share().size() == this->input_a_->get_num_simd());
  std::transform(std::begin(this->input_a_->get_share()), std::end(this->input_a_->get_share()),
                 std::begin(this->input_b_->get_share()), std::begin(this->output_->get_share()),
                 std::plus{});
  this->output_->set_online_ready();
}

template class ArithmeticGMWADDGate<std::uint8_t>;
template class ArithmeticGMWADDGate<std::uint16_t>;
template class ArithmeticGMWADDGate<std::uint32_t>;
template class ArithmeticGMWADDGate<std::uint64_t>;

template <typename T>
ArithmeticGMWMULGate<T>::ArithmeticGMWMULGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                              ArithmeticGMWWireP<T>&& in_a,
                                              ArithmeticGMWWireP<T>&& in_b)
    : detail::BasicArithmeticGMWBinaryGate<T>(gate_id, gmw_provider, std::move(in_a),
                                              std::move(in_b)),
      gmw_provider_(gmw_provider),
      mt_offset_(
          gmw_provider.get_mt_provider().RequestArithmeticMTs<T>(this->input_a_->get_num_simd())),
      share_futures_(gmw_provider_.register_for_ints_messages<T>(
          this->gate_id_, 2 * this->input_a_->get_num_simd())) {}

template <typename T>
void ArithmeticGMWMULGate<T>::evaluate_online() {
  auto num_simd = this->input_a_->get_num_simd();
  const auto& mtp = gmw_provider_.get_mt_provider();
  const auto& all_mts = mtp.GetIntegerAll<T>();
  this->input_a_->wait_online();
  this->input_b_->wait_online();
  const auto& x = this->input_a_->get_share();
  const auto& y = this->input_b_->get_share();

  //  mask inputs
  std::vector<T> de(2 * num_simd);
  auto it = std::transform(std::begin(x), std::end(x), std::begin(all_mts.a) + this->mt_offset_,
                           std::begin(de), std::minus{});
  std::transform(std::begin(y), std::end(y), std::begin(all_mts.b) + this->mt_offset_, it,
                 std::minus{});
  this->gmw_provider_.broadcast_ints_message(this->gate_id_, de);

  // compute d, e
  auto num_parties = gmw_provider_.get_num_parties();
  auto my_id = gmw_provider_.get_my_id();
  for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
    if (party_id == my_id) {
      continue;
    }
    auto other_share = share_futures_[party_id].get();
    std::transform(std::begin(de), std::end(de), std::begin(other_share), std::begin(de),
                   std::plus{});
  }
  // result = c ...
  std::vector<T> result(std::begin(all_mts.c) + mt_offset_,
                        std::begin(all_mts.c) + mt_offset_ + num_simd);
  // ... - d * e ...
  if (this->gmw_provider_.is_my_job(this->gate_id_)) {
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      result[simd_j] -= de[simd_j] * de[simd_j + num_simd];
    }
  }
  // ... + e * x + d * y
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    result[simd_j] += de[simd_j] * y[simd_j] + de[simd_j + num_simd] * x[simd_j];
  }
  this->output_->get_share() = std::move(result);
  this->output_->set_online_ready();
}

template class ArithmeticGMWMULGate<std::uint8_t>;
template class ArithmeticGMWMULGate<std::uint16_t>;
template class ArithmeticGMWMULGate<std::uint32_t>;
template class ArithmeticGMWMULGate<std::uint64_t>;

template <typename T>
ArithmeticGMWSQRGate<T>::ArithmeticGMWSQRGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                              ArithmeticGMWWireP<T>&& in)
    : detail::BasicArithmeticGMWUnaryGate<T>(gate_id, gmw_provider, std::move(in)),
      gmw_provider_(gmw_provider),
      sp_offset_(
          gmw_provider.get_sp_provider().RequestSPs<T>(this->input_->get_num_simd())),
      share_futures_(gmw_provider_.register_for_ints_messages<T>(
          this->gate_id_, this->input_->get_num_simd())) {}

template <typename T>
void ArithmeticGMWSQRGate<T>::evaluate_online() {
  auto num_simd = this->input_->get_num_simd();
  const auto& spp = gmw_provider_.get_sp_provider();
  const auto& all_sps = spp.GetSPsAll<T>();
  this->input_->wait_online();
  const auto& x = this->input_->get_share();

  //  mask inputs
  std::vector<T> d(num_simd);
  std::transform(std::begin(x), std::end(x), std::begin(all_sps.a) + this->sp_offset_,
                           std::begin(d), std::minus{});
  this->gmw_provider_.broadcast_ints_message(this->gate_id_, d);

  // compute d
  auto num_parties = gmw_provider_.get_num_parties();
  auto my_id = gmw_provider_.get_my_id();
  for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
    if (party_id == my_id) {
      continue;
    }
    auto other_share = share_futures_[party_id].get();
    std::transform(std::begin(d), std::end(d), std::begin(other_share), std::begin(d),
                   std::plus{});
  }
  // result = c ...
  std::vector<T> result(std::begin(all_sps.c) + sp_offset_,
                        std::begin(all_sps.c) + sp_offset_ + num_simd);
  // ... - d^2 ...
  if (this->gmw_provider_.is_my_job(this->gate_id_)) {
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      result[simd_j] -= d[simd_j] * d[simd_j];
    }
  }
  // ... + 2 * d * x
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    result[simd_j] += 2 * d[simd_j] * x[simd_j];
  }
  this->output_->get_share() = std::move(result);
  this->output_->set_online_ready();
}

template class ArithmeticGMWSQRGate<std::uint8_t>;
template class ArithmeticGMWSQRGate<std::uint16_t>;
template class ArithmeticGMWSQRGate<std::uint32_t>;
template class ArithmeticGMWSQRGate<std::uint64_t>;

}  // namespace MOTION::proto::gmw
