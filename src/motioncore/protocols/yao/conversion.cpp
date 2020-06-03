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

#include "crypto/motion_base_provider.h"
#include "crypto/oblivious_transfer/ot_flavors.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "crypto/sharing_randomness_generator.h"
#include "protocols/gmw/wire.h"
#include "utility/helpers.h"
#include "wire.h"
#include "yao_provider.h"

namespace MOTION::proto::yao {

YaoToBooleanGMWGateGarbler::YaoToBooleanGMWGateGarbler(std::size_t gate_id, YaoProvider&,
                                                       YaoWireVector&& in)
    : NewGate(gate_id), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires, [num_simd] {
    auto wire = std::make_shared<gmw::BooleanGMWWire>(num_simd);
    wire->get_share().Resize(num_simd);
    return wire;
  });
}

void YaoToBooleanGMWGateGarbler::evaluate_setup() {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire_yao = inputs_[wire_i];
    auto& wire_gmw = outputs_[wire_i];
    wire_yao->wait_setup();
    const auto& keys = wire_yao->get_keys();
    auto& share = wire_gmw->get_share();
    assert(share.GetSize() == num_simd);
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      share.Set(bool(*keys[simd_j].data() & std::byte(0x01)), simd_j);
    }
    outputs_[wire_i]->set_online_ready();
  }
}

void YaoToBooleanGMWGateGarbler::evaluate_online() {
  // nothing to do
}

YaoToBooleanGMWGateEvaluator::YaoToBooleanGMWGateEvaluator(std::size_t gate_id, YaoProvider&,
                                                           YaoWireVector&& in)
    : NewGate(gate_id), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires, [num_simd] {
    auto wire = std::make_shared<gmw::BooleanGMWWire>(num_simd);
    wire->get_share().Resize(num_simd);
    return wire;
  });
}

void YaoToBooleanGMWGateEvaluator::evaluate_setup() {
  // nothing to do
}

void YaoToBooleanGMWGateEvaluator::evaluate_online() {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire_yao = inputs_[wire_i];
    auto& wire_gmw = outputs_[wire_i];
    wire_yao->wait_online();
    const auto& keys = wire_yao->get_keys();
    auto& share = wire_gmw->get_share();
    assert(share.GetSize() == num_simd);
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      share.Set(bool(*keys[simd_j].data() & std::byte(0x01)), simd_j);
    }
    outputs_[wire_i]->set_online_ready();
  }
}

BooleanGMWToYaoGateGarbler::BooleanGMWToYaoGateGarbler(std::size_t gate_id, YaoProvider& yao_provider,
                                                       gmw::BooleanGMWWireVector&& in)
    : NewGate(gate_id), yao_provider_(yao_provider), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires, [num_simd] {
    return std::make_shared<YaoWire>(num_simd);
  });
  ot_sender_ = yao_provider.get_ot_provider().RegisterSendGOT128(num_wires * num_simd);
  ot_inputs_.resize(2 * num_wires * num_simd);
}

BooleanGMWToYaoGateGarbler::~BooleanGMWToYaoGateGarbler() = default;

void BooleanGMWToYaoGateGarbler::evaluate_setup() {
  for (auto& wire : outputs_) {
    wire->get_keys().set_to_random();
    wire->set_setup_ready();
  }
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  auto idx = [num_simd](auto wire_i, auto simd_j) { return 2 * wire_i * num_simd + 2 * simd_j; };
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& zero_keys = outputs_[wire_i]->get_keys();
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      ot_inputs_[idx(wire_i, simd_j)] = zero_keys[simd_j];
      ot_inputs_[idx(wire_i, simd_j) + 1] = zero_keys[simd_j];
    }
  }
}

void BooleanGMWToYaoGateGarbler::evaluate_online() {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  ENCRYPTO::block128_vector ot_inputs(num_wires * num_simd);
  const auto& global_offset = yao_provider_.get_global_offset();
  auto idx = [num_simd](auto wire_i, auto simd_j) { return 2 * wire_i * num_simd + 2 * simd_j; };
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    inputs_[wire_i]->wait_online();
    const auto& share = inputs_[wire_i]->get_share();
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      if (share.Get(simd_j)) {
        ot_inputs_[idx(wire_i, simd_j)] ^= global_offset;
      } else {
        ot_inputs_[idx(wire_i, simd_j) + 1] ^= global_offset;
      }
    }
  }
  ot_sender_->SetInputs(std::move(ot_inputs_));
  ot_sender_->SendMessages();
}

BooleanGMWToYaoGateEvaluator::BooleanGMWToYaoGateEvaluator(std::size_t gate_id,
                                                           YaoProvider& yao_provider,
                                                           gmw::BooleanGMWWireVector&& in)
    : NewGate(gate_id), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires,
                  [num_simd] { return std::make_shared<YaoWire>(num_simd); });
  ot_receiver_ = yao_provider.get_ot_provider().RegisterReceiveGOT128(num_wires * num_simd);
}

BooleanGMWToYaoGateEvaluator::~BooleanGMWToYaoGateEvaluator() = default;

void BooleanGMWToYaoGateEvaluator::evaluate_setup() {
  // nothing to do
}

void BooleanGMWToYaoGateEvaluator::evaluate_online() {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  ENCRYPTO::BitVector<> shares;
  shares.Reserve(Helpers::Convert::BitsToBytes(num_wires * num_simd));
  for (const auto& wire : inputs_) {
    wire->wait_online();
    shares.Append(wire->get_share());
  }
  ot_receiver_->SetChoices(std::move(shares));
  ot_receiver_->SendCorrections();
  ot_receiver_->ComputeOutputs();
  const auto all_keys = ot_receiver_->GetOutputs();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    auto& wire_keys = outputs_[wire_i]->get_keys();
    std::copy_n(&all_keys[wire_i * num_simd], num_simd, wire_keys.data());
    outputs_[wire_i]->set_online_ready();
  }
}

template <typename T>
YaoToArithmeticGMWGateGarbler<T>::YaoToArithmeticGMWGateGarbler(std::size_t gate_id,
                                                                YaoProvider& yao_provider,
                                                                std::size_t num_simd)
    : NewGate(gate_id),
      output_(std::make_shared<gmw::ArithmeticGMWWire<T>>(num_simd)),
      yao_provider_(yao_provider) {}

template <typename T>
void YaoToArithmeticGMWGateGarbler<T>::evaluate_setup() {
  auto num_simd = output_->get_num_simd();
  auto mask = Helpers::RandomVector<T>(output_->get_num_simd());
  auto& mbp = yao_provider_.get_motion_base_provider();
  auto& rng = mbp.get_their_randomness_generator(1);
  auto& share = output_->get_share();
  share = rng.GetUnsigned<T>(gate_id_, num_simd);
  std::transform(std::begin(share), std::end(share), std::begin(mask), std::begin(share), std::minus{});
  output_->set_online_ready();

  auto bit_vectors = ENCRYPTO::ToInput(mask);
  mask_promise_.set_value(std::move(bit_vectors));
}

template class YaoToArithmeticGMWGateGarbler<std::uint8_t>;
template class YaoToArithmeticGMWGateGarbler<std::uint16_t>;
template class YaoToArithmeticGMWGateGarbler<std::uint32_t>;
template class YaoToArithmeticGMWGateGarbler<std::uint64_t>;

template <typename T>
YaoToArithmeticGMWGateEvaluator<T>::YaoToArithmeticGMWGateEvaluator(
    std::size_t gate_id, YaoProvider& yao_provider, std::size_t num_simd)
    : NewGate(gate_id),
      output_(std::make_shared<gmw::ArithmeticGMWWire<T>>(num_simd)),
      yao_provider_(yao_provider) {}

template <typename T>
void YaoToArithmeticGMWGateEvaluator<T>::evaluate_setup() {
  auto& mbp = yao_provider_.get_motion_base_provider();
  auto& rng = mbp.get_my_randomness_generator(0);
  auto num_simd = output_->get_num_simd();
  output_->get_share() = rng.GetUnsigned<T>(gate_id_, num_simd);
}

template <typename T>
void YaoToArithmeticGMWGateEvaluator<T>::evaluate_online() {
  auto masked_value = masked_value_future_.get();
  std::vector<T> masked_value_int_ = ENCRYPTO::ToVectorOutput<T>(std::move(masked_value));
  auto& share = output_->get_share();
  std::transform(std::begin(masked_value_int_), std::end(masked_value_int_), std::begin(share),
                 std::begin(share), std::minus{});
  output_->set_online_ready();
}

template class YaoToArithmeticGMWGateEvaluator<std::uint8_t>;
template class YaoToArithmeticGMWGateEvaluator<std::uint16_t>;
template class YaoToArithmeticGMWGateEvaluator<std::uint32_t>;
template class YaoToArithmeticGMWGateEvaluator<std::uint64_t>;

}  // namespace MOTION::proto::yao
