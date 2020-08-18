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

#include <cstdint>
#include "conversion.h"

#include "crypto/motion_base_provider.h"
#include "crypto/oblivious_transfer/ot_flavors.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "crypto/sharing_randomness_generator.h"
#include "protocols/beavy/wire.h"
#include "protocols/gmw/wire.h"
#include "tools.h"
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

BooleanGMWToYaoGateGarbler::BooleanGMWToYaoGateGarbler(std::size_t gate_id,
                                                       YaoProvider& yao_provider,
                                                       gmw::BooleanGMWWireVector&& in)
    : NewGate(gate_id), yao_provider_(yao_provider), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires,
                  [num_simd] { return std::make_shared<YaoWire>(num_simd); });
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
  std::transform(std::begin(share), std::end(share), std::begin(mask), std::begin(share),
                 std::minus{});
  output_->set_online_ready();

  auto bit_vectors = ENCRYPTO::ToInput(mask);
  mask_promise_.set_value(std::move(bit_vectors));
}

template class YaoToArithmeticGMWGateGarbler<std::uint8_t>;
template class YaoToArithmeticGMWGateGarbler<std::uint16_t>;
template class YaoToArithmeticGMWGateGarbler<std::uint32_t>;
template class YaoToArithmeticGMWGateGarbler<std::uint64_t>;

template <typename T>
YaoToArithmeticGMWGateEvaluator<T>::YaoToArithmeticGMWGateEvaluator(std::size_t gate_id,
                                                                    YaoProvider& yao_provider,
                                                                    std::size_t num_simd)
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

YaoToBooleanBEAVYGateGarbler::YaoToBooleanBEAVYGateGarbler(std::size_t gate_id,
                                                           YaoProvider& yao_provider,
                                                           YaoWireVector&& in)
    : NewGate(gate_id), yao_provider_(yao_provider), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires,
                  [num_simd] { return std::make_shared<beavy::BooleanBEAVYWire>(num_simd); });
  public_share_future_ = yao_provider_.register_for_bits_message(gate_id, num_wires * num_simd);
}

void YaoToBooleanBEAVYGateGarbler::evaluate_setup() {
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire_yao = inputs_[wire_i];
    auto& wire_beavy = outputs_[wire_i];
    wire_yao->wait_setup();
    const auto& keys = wire_yao->get_keys();
    ENCRYPTO::BitVector<> lsbs;
    get_lsbs_from_keys(lsbs, keys.data(), num_simd);
    wire_beavy->get_secret_share() = std::move(lsbs);
    wire_beavy->set_setup_ready();
  }
}

void YaoToBooleanBEAVYGateGarbler::evaluate_online() {
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_[0]->get_num_simd();
  const auto public_share = public_share_future_.get();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    auto& wire_beavy = outputs_[wire_i];
    auto& pshare = wire_beavy->get_public_share();
    pshare = public_share.Subset(wire_i * num_simd, (wire_i + 1) * num_simd);
    wire_beavy->set_online_ready();
  }
}

YaoToBooleanBEAVYGateEvaluator::YaoToBooleanBEAVYGateEvaluator(std::size_t gate_id,
                                                               YaoProvider& yao_provider,
                                                               YaoWireVector&& in)
    : NewGate(gate_id), yao_provider_(yao_provider), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires,
                  [num_simd] { return std::make_shared<beavy::BooleanBEAVYWire>(num_simd); });
}

void YaoToBooleanBEAVYGateEvaluator::evaluate_setup() {
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    auto& wire_beavy = outputs_[wire_i];
    wire_beavy->get_secret_share() = ENCRYPTO::BitVector<>::Random(num_simd);
    wire_beavy->set_setup_ready();
  }
}

void YaoToBooleanBEAVYGateEvaluator::evaluate_online() {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  ENCRYPTO::BitVector<> public_share;
  public_share.Reserve(Helpers::Convert::BitsToBytes(num_wires * num_simd));
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& wire_yao = inputs_[wire_i];
    auto& wire_beavy = outputs_[wire_i];
    wire_yao->wait_online();
    const auto& keys = wire_yao->get_keys();
    ENCRYPTO::BitVector<> tmp;
    get_lsbs_from_keys(tmp, keys.data(), num_simd);
    tmp ^= wire_beavy->get_secret_share();
    public_share.Append(tmp);
    wire_beavy->get_public_share() = std::move(tmp);
    wire_beavy->set_online_ready();
  }
  yao_provider_.send_bits_message(gate_id_, public_share);
}

BooleanBEAVYToYaoGateGarbler::BooleanBEAVYToYaoGateGarbler(std::size_t gate_id,
                                                           YaoProvider& yao_provider,
                                                           beavy::BooleanBEAVYWireVector&& in)
    : NewGate(gate_id), yao_provider_(yao_provider), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires,
                  [num_simd] { return std::make_shared<YaoWire>(num_simd); });
  auto& ot_provider = yao_provider.get_ot_provider();
  ot_sender_ = ot_provider.RegisterSendFixedXCOT128(num_wires * num_simd);
}

BooleanBEAVYToYaoGateGarbler::~BooleanBEAVYToYaoGateGarbler() = default;

void BooleanBEAVYToYaoGateGarbler::evaluate_setup() {
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_[0]->get_num_simd();
  const auto& global_offset = yao_provider_.get_global_offset();
  ot_sender_->SetCorrelation(global_offset);
  ot_sender_->SendMessages();
  ot_sender_->ComputeOutputs();
  auto keys_e = ot_sender_->GetOutputs();
  keys_g_ = ENCRYPTO::block128_vector::make_random(num_wires * num_simd);
  keys_e ^= keys_g_;
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    auto& wire_yao = outputs_[wire_i];
    wire_yao->get_keys() = ENCRYPTO::block128_vector(num_simd, keys_e[wire_i * num_simd].data());
    wire_yao->set_setup_ready();
  }
}

void BooleanBEAVYToYaoGateGarbler::evaluate_online() {
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_[0]->get_num_simd();
  const auto& global_offset = yao_provider_.get_global_offset();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    auto& wire_beavy = inputs_[wire_i];
    wire_beavy->wait_setup();
    wire_beavy->wait_online();
    const auto bits = wire_beavy->get_secret_share() ^ wire_beavy->get_public_share();
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      if (bits.Get(simd_j)) {
        keys_g_[wire_i * num_simd + simd_j] ^= global_offset;
      }
    }
  }
  yao_provider_.send_blocks_message(gate_id_, std::move(keys_g_));
}

BooleanBEAVYToYaoGateEvaluator::BooleanBEAVYToYaoGateEvaluator(std::size_t gate_id,
                                                               YaoProvider& yao_provider,
                                                               beavy::BooleanBEAVYWireVector&& in)
    : NewGate(gate_id), inputs_(std::move(in)) {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  outputs_.reserve(num_wires);
  std::generate_n(std::back_inserter(outputs_), num_wires,
                  [num_simd] { return std::make_shared<YaoWire>(num_simd); });
  garbler_keys_future_ = yao_provider.register_for_blocks_message(gate_id, num_wires * num_simd);
  auto& ot_provider = yao_provider.get_ot_provider();
  ot_receiver_ = ot_provider.RegisterReceiveFixedXCOT128(num_wires * num_simd);
}

BooleanBEAVYToYaoGateEvaluator::~BooleanBEAVYToYaoGateEvaluator() = default;

void BooleanBEAVYToYaoGateEvaluator::evaluate_setup() {
  const auto num_wires = inputs_.size();
  const auto num_simd = inputs_[0]->get_num_simd();
  auto ot_inputs = ENCRYPTO::BitVector<>();
  ot_inputs.Reserve(Helpers::Convert::BitsToBytes(num_wires * num_simd));
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    auto& wire_beavy = inputs_[wire_i];
    wire_beavy->wait_setup();
    ot_inputs.Append(wire_beavy->get_secret_share());
  }
  ot_receiver_->SetChoices(std::move(ot_inputs));
  ot_receiver_->SendCorrections();
  ot_receiver_->ComputeOutputs();
}

void BooleanBEAVYToYaoGateEvaluator::evaluate_online() {
  auto num_wires = inputs_.size();
  auto num_simd = inputs_[0]->get_num_simd();
  auto keys = ot_receiver_->GetOutputs();
  keys ^= garbler_keys_future_.get();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    auto& wire_yao = outputs_[wire_i];
    wire_yao->get_keys() = ENCRYPTO::block128_vector(num_simd, keys[wire_i * num_simd].data());
    wire_yao->set_online_ready();
  }
}

template <typename T>
YaoToArithmeticBEAVYGateGarbler<T>::YaoToArithmeticBEAVYGateGarbler(std::size_t gate_id,
                                                                    YaoProvider& yao_provider,
                                                                    std::size_t num_simd)
    : NewGate(gate_id),
      output_(std::make_shared<beavy::ArithmeticBEAVYWire<T>>(num_simd)),
      masked_value_public_share_future_(
          yao_provider.register_for_bits_message(gate_id_, ENCRYPTO::bit_size_v<T> * num_simd)),
      yao_provider_(yao_provider) {}

template <typename T>
void YaoToArithmeticBEAVYGateGarbler<T>::evaluate_setup() {
  auto num_simd = output_->get_num_simd();
  auto mask = Helpers::RandomVector<T>(num_simd);
  auto& mbp = yao_provider_.get_motion_base_provider();
  auto& rng = mbp.get_their_randomness_generator(1);
  auto& public_share = output_->get_public_share();
  auto& secret_share = output_->get_secret_share();
  // generate shares:
  //   Delta = public_share of mask ||
  //   delta_1 = secret_share^1 of mask ||
  //   delta'_0 = secret_share^0 of masked value
  auto random_ints = rng.GetUnsigned<T>(gate_id_, 3 * num_simd);
  public_share.resize(num_simd);
  // initialize public_share with Delta (Delta - Delta' is computed in the online phase)
  std::copy_n(std::begin(random_ints), num_simd, std::begin(public_share));
  secret_share.resize(num_simd);
  // the secret_share is delta'_0 - delta_0 = delta'_0 + delta_1 - Delta + r
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    secret_share[simd_j] = random_ints[2 * num_simd + simd_j] + random_ints[num_simd + simd_j] -
                           random_ints[simd_j] + mask[simd_j];
  }
  output_->set_setup_ready();

  // pass the mask via future
  auto bit_vectors = ENCRYPTO::ToInput(mask);
  mask_promise_.set_value(std::move(bit_vectors));
}

template <typename T>
void YaoToArithmeticBEAVYGateGarbler<T>::evaluate_online() {
  const auto num_simd = output_->get_num_simd();
  auto& public_share = output_->get_public_share();
  const auto masked_value_public_share_bits = masked_value_public_share_future_.get();
  std::vector<T> masked_value_public_share(num_simd);
  std::memcpy(masked_value_public_share.data(), masked_value_public_share_bits.GetData().data(),
              num_simd * sizeof(T));
  std::transform(std::begin(masked_value_public_share), std::end(masked_value_public_share),
                 std::begin(public_share), std::begin(public_share), std::minus{});
  output_->set_online_ready();
}

template class YaoToArithmeticBEAVYGateGarbler<std::uint8_t>;
template class YaoToArithmeticBEAVYGateGarbler<std::uint16_t>;
template class YaoToArithmeticBEAVYGateGarbler<std::uint32_t>;
template class YaoToArithmeticBEAVYGateGarbler<std::uint64_t>;

template <typename T>
YaoToArithmeticBEAVYGateEvaluator<T>::YaoToArithmeticBEAVYGateEvaluator(std::size_t gate_id,
                                                                        YaoProvider& yao_provider,
                                                                        std::size_t num_simd)
    : NewGate(gate_id),
      output_(std::make_shared<beavy::ArithmeticBEAVYWire<T>>(num_simd)),
      yao_provider_(yao_provider) {}

template <typename T>
void YaoToArithmeticBEAVYGateEvaluator<T>::evaluate_setup() {
  auto& mbp = yao_provider_.get_motion_base_provider();
  auto& rng = mbp.get_my_randomness_generator(0);
  auto num_simd = output_->get_num_simd();
  // secret_share of masked value
  masked_value_secret_share_ = Helpers::RandomVector<T>(num_simd);
  auto& public_share = output_->get_public_share();
  auto& secret_share = output_->get_secret_share();
  // initialize public_share with -Delta (Delta' is added in the online phase)
  // generate shares:
  //   Delta = public_share of mask ||
  //   delta_1 = secret_share^1 of mask ||
  //   delta'_0 = secret_share^0 of masked value
  auto random_ints = rng.GetUnsigned<T>(gate_id_, 3 * num_simd);
  public_share.resize(num_simd);
  // initialize public_share with Delta (Delta' - Delta is computed in the online phase)
  std::copy_n(std::begin(random_ints), num_simd, std::begin(public_share));
  secret_share.resize(num_simd);
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    secret_share[simd_j] = masked_value_secret_share_[simd_j] - random_ints[num_simd + simd_j];
  }
  // prepared public_share of masked value (add masked value to this in online phase)
  masked_value_public_share_.resize(num_simd);
  std::transform(std::begin(masked_value_secret_share_), std::end(masked_value_secret_share_),
                 std::begin(random_ints) + 2 * num_simd, std::begin(masked_value_public_share_),
                 std::plus{});

  output_->set_setup_ready();
}

template <typename T>
void YaoToArithmeticBEAVYGateEvaluator<T>::evaluate_online() {
  auto masked_value_bits = masked_value_future_.get();
  auto masked_value = ENCRYPTO::ToVectorOutput<T>(masked_value_bits);
  std::transform(std::begin(masked_value), std::end(masked_value),
                 std::begin(masked_value_public_share_), std::begin(masked_value_public_share_),
                 std::plus{});
  // hack, since YaoProvider currenty supports only bit or block messages
  ENCRYPTO::BitVector<> bit_msg(
      reinterpret_cast<const std::byte*>(masked_value_public_share_.data()),
      ENCRYPTO::bit_size_v<T> * masked_value_public_share_.size());
  yao_provider_.send_bits_message(gate_id_, std::move(bit_msg));
  auto& public_share = output_->get_public_share();
  std::transform(std::begin(masked_value_public_share_), std::end(masked_value_public_share_),
                 std::begin(public_share), std::begin(public_share), std::minus{});
  output_->set_online_ready();
}

template class YaoToArithmeticBEAVYGateEvaluator<std::uint8_t>;
template class YaoToArithmeticBEAVYGateEvaluator<std::uint16_t>;
template class YaoToArithmeticBEAVYGateEvaluator<std::uint32_t>;
template class YaoToArithmeticBEAVYGateEvaluator<std::uint64_t>;

}  // namespace MOTION::proto::yao
