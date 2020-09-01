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

#include "crypto/multiplication_triple/sb_provider.h"
#include "gmw_provider.h"
#include "utility/constants.h"
#include "utility/logger.h"

namespace MOTION::proto::gmw {

template <typename T>
BooleanToArithmeticGMWGate<T>::BooleanToArithmeticGMWGate(std::size_t gate_id,
                                                          GMWProvider& gmw_provider,
                                                          BooleanGMWWireVector&& in)
    : NewGate(gate_id), inputs_(std::move(in)), gmw_provider_(gmw_provider) {
  const auto num_wires = inputs_.size();
  if (num_wires != ENCRYPTO::bit_size_v<T>) {
    throw std::logic_error("number of wires need to be equal to bit size of T");
  }
  const auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    if (inputs_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  output_ = std::make_shared<gmw::ArithmeticGMWWire<T>>(num_simd);
  output_->get_share().resize(num_simd, 0);
  sb_offset_ = gmw_provider_.get_sb_provider().RequestSBs<T>(ENCRYPTO::bit_size_v<T> * num_simd);
  const auto my_id = gmw_provider_.get_my_id();
  t_share_future_ =
      gmw_provider_.register_for_bits_message(1 - my_id, gate_id_, num_wires * num_simd);
}

template <typename T>
void BooleanToArithmeticGMWGate<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanToArithmeticGMWGate<T>::evaluate_online start", gate_id_));
    }
  }
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_[0]->get_num_simd();

  // let `sbs` point to our SBs
  const auto& all_sbs = gmw_provider_.get_sb_provider().GetSBsAll<T>();
  const auto* sbs = &all_sbs[sb_offset_];

  ENCRYPTO::BitVector<> t;
  t.Reserve(Helpers::Convert::BitsToBytes(num_wires * num_simd));

  // collect all shares into a single buffer
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire = inputs_[wire_i];
    wire->wait_online();
    t.Append(wire->get_share());
  }

  // indexing function
  const auto idx = [num_simd](auto wire_i, auto simd_j) { return wire_i * num_simd + simd_j; };

  // mask them with the shared bits
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      auto x = t.Get(idx(wire_i, simd_j));
      auto r = bool(sbs[idx(wire_i, simd_j)] & 1);
      t.Set(x ^ r, idx(wire_i, simd_j));
    }
  }

  // reconstruct masked values
  gmw_provider_.broadcast_bits_message(gate_id_, t);
  t ^= t_share_future_.get();

  const auto is_my_job = gmw_provider_.is_my_job(gate_id_);

  // remove mask in arithmetic sharing
  auto& output = output_->get_share();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      auto t_ij = T(t.Get(idx(wire_i, simd_j)));
      auto r_ij = sbs[idx(wire_i, simd_j)];
      T value = r_ij - 2 * t_ij * r_ij;
      if (is_my_job) {
        value += t_ij;
      }
      output[simd_j] += value << wire_i;
    }
  }

  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanToArithmeticGMWGate<T>::evaluate_online end", gate_id_));
    }
  }
}

template class BooleanToArithmeticGMWGate<std::uint8_t>;
template class BooleanToArithmeticGMWGate<std::uint16_t>;
template class BooleanToArithmeticGMWGate<std::uint32_t>;
template class BooleanToArithmeticGMWGate<std::uint64_t>;

}  // namespace MOTION::proto::gmw
