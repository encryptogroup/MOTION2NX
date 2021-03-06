// MIT License
//
// Copyright (c) 2019 Oleksandr Tkachenko
// Cryptography and Privacy Engineering Group (ENCRYPTO)
// TU Darmstadt, Germany
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

#include "conversion_gate.h"

#include <cassert>

#include "base/backend.h"
#include "communication/bmr_message.h"
#include "communication/communication_layer.h"
#include "crypto/bmr_provider.h"
#include "data_storage/bmr_data.h"
#include "gate/bmr_gate.h"
#include "secure_type/secure_unsigned_integer.h"
#include "share/bmr_share.h"
#include "share/boolean_gmw_share.h"
#include "utility/bit_vector.h"
#include "utility/constants.h"
#include "utility/fiber_condition.h"
#include "wire/bmr_wire.h"
#include "wire/boolean_gmw_wire.h"

namespace MOTION::Gates::Conversion {

BMRToGMWGate::BMRToGMWGate(const Shares::SharePtr &parent) : OneGate(parent->GetBackend()) {
  parent_ = parent->GetWires();

  assert(parent_.size() > 0);
  assert(parent_.at(0)->GetBitLength() > 0);
  for ([[maybe_unused]] const auto &wire : parent_) assert(wire->GetProtocol() == MPCProtocol::BMR);

  requires_online_interaction_ = false;
  gate_type_ = GateType::NonInteractive;
  gate_id_ = GetRegister().NextGateId();

  for (auto &wire : parent_) {
    RegisterWaitingFor(wire->GetWireId());
    wire->RegisterWaitingGate(gate_id_);
  }

  // create output wires
  auto num_wires = parent_.size();
  output_wires_.reserve(num_wires);
  for (size_t i = 0; i < num_wires; ++i) {
    auto &w = output_wires_.emplace_back(std::static_pointer_cast<Wires::Wire>(
        std::make_shared<Wires::GMWWire>(parent->GetNumOfSIMDValues(), backend_)));
    GetRegister().RegisterNextWire(w);
  }

  if constexpr (MOTION_DEBUG) {
    auto gate_info = fmt::format("gate id {}, parent wires: ", gate_id_);
    for (const auto &wire : parent_) gate_info.append(fmt::format("{} ", wire->GetWireId()));
    gate_info.append(" output wires: ");
    for (const auto &wire : output_wires_) gate_info.append(fmt::format("{} ", wire->GetWireId()));
    GetLogger().LogDebug(fmt::format(
        "Created a BMR to Boolean GMW conversion gate with following properties: {}", gate_info));
  }
}

void BMRToGMWGate::EvaluateSetup() {
  SetSetupIsReady();
  GetRegister().IncrementEvaluatedGatesSetupCounter();
}

void BMRToGMWGate::EvaluateOnline() {
  WaitSetup();
  if constexpr (MOTION_DEBUG) {
    GetLogger().LogDebug(fmt::format(
        "Start evaluating online phase of BMR to Boolean GMW Gate with id#{}", gate_id_));
  }

  for (auto i = 0ull; i < parent_.size(); ++i) {
    auto bmr_in{std::dynamic_pointer_cast<const Wires::BMRWire>(parent_.at(i))};
    assert(bmr_in);

    auto gmw_out{std::dynamic_pointer_cast<Wires::GMWWire>(output_wires_.at(i))};
    assert(gmw_out);

    bmr_in->GetIsReadyCondition().Wait();
    const auto& comm_layer = get_communication_layer();
    const auto my_id = comm_layer.get_my_id();
    const auto num_parties = comm_layer.get_num_parties();
    auto &v{gmw_out->GetMutableValues()};

    // set current gmw shared bits on wire to permutation bits of parent BMR wire
    v = bmr_in->GetPermutationBits();

    // one party needs to XOR shared GMW bits with the public values of BMR wire
    // the party doing this is chosen based on the wire id for the purpose of load balancing
    if ((gmw_out->GetWireId() % num_parties) == my_id) v ^= bmr_in->GetPublicValues();
  }

  if constexpr (MOTION_DEBUG) {
    GetLogger().LogDebug(fmt::format(
        "Finished evaluating online phase of BMR to Boolean GMW Gate with id#{}", gate_id_));
  }
  SetOnlineIsReady();
  GetRegister().IncrementEvaluatedGatesOnlineCounter();
}

const Shares::GMWSharePtr BMRToGMWGate::GetOutputAsGMWShare() const {
  auto result = std::make_shared<Shares::GMWShare>(output_wires_);
  assert(result);
  return result;
}

const Shares::SharePtr BMRToGMWGate::GetOutputAsShare() const {
  auto result = std::static_pointer_cast<Shares::Share>(GetOutputAsGMWShare());
  assert(result);
  return result;
}

GMWToBMRGate::GMWToBMRGate(const Shares::SharePtr &parent) : OneGate(parent->GetBackend()) {
  parent_ = parent->GetWires();
  const auto num_simd{parent->GetNumOfSIMDValues()};

  assert(parent_.size() > 0);
  assert(parent_.at(0)->GetBitLength() > 0);
  for ([[maybe_unused]] const auto &wire : parent_)
    assert(wire->GetProtocol() == MPCProtocol::BooleanGMW);

  requires_online_interaction_ = false;
  gate_type_ = GateType::NonInteractive;
  gate_id_ = GetRegister().NextGateId();

  for (auto &wire : parent_) {
    RegisterWaitingFor(wire->GetWireId());
    wire->RegisterWaitingGate(gate_id_);
  }

  output_wires_.resize(parent_.size());
  for (auto &w : output_wires_) {
    w = std::make_shared<Wires::BMRWire>(parent->GetNumOfSIMDValues(), backend_);
    GetRegister().RegisterNextWire(w);
  }

  assert(gate_id_ >= 0);

  auto& bmr_provider = backend_.get_bmr_provider();
  received_public_keys_ = bmr_provider.register_for_input_keys(gate_id_, num_simd * output_wires_.size());
  received_public_values_ = bmr_provider.register_for_input_public_values(gate_id_, num_simd * output_wires_.size());

  if constexpr (MOTION_DEBUG) {
    auto gate_info = fmt::format("gate id {}, parent wires: ", gate_id_);
    for (const auto &wire : parent_) gate_info.append(fmt::format("{} ", wire->GetWireId()));
    gate_info.append(" output wires: ");
    for (const auto &wire : output_wires_) gate_info.append(fmt::format("{} ", wire->GetWireId()));
    GetLogger().LogDebug(fmt::format(
        "Created a Boolean GMW to BMR conversion gate with following properties: {}", gate_info));
  }
}

void GMWToBMRGate::EvaluateSetup() {
  if constexpr (MOTION_DEBUG) {
    GetLogger().LogDebug(fmt::format(
        "Start evaluating setup phase of Boolean GMW to BMR Gate with id#{}", gate_id_));
  }

  for (auto wire_i = 0ull; wire_i < output_wires_.size(); ++wire_i) {
    auto bmr_out = std::dynamic_pointer_cast<Wires::BMRWire>(output_wires_.at(wire_i));
    assert(bmr_out);
    bmr_out->GenerateRandomPrivateKeys();
    bmr_out->GenerateRandomPermutationBits();
    bmr_out->SetSetupIsReady();
  }
  if constexpr (MOTION_DEBUG) {
    GetLogger().LogDebug(fmt::format(
        "Finished evaluating setup phase of Boolean GMW to BMR Gate with id#{}", gate_id_));
  }
  SetSetupIsReady();
  GetRegister().IncrementEvaluatedGatesSetupCounter();
}

void GMWToBMRGate::EvaluateOnline() {
  WaitSetup();
  if constexpr (MOTION_DEBUG) {
    GetLogger().LogDebug(fmt::format(
        "Start evaluating online phase of Boolean GMW to BMR Gate with id#{}", gate_id_));
  }

  const auto num_simd{output_wires_.at(0)->GetNumOfSIMDValues()};
  const auto num_wires{output_wires_.size()};
  auto& comm_layer = get_communication_layer();
  const auto my_id = comm_layer.get_my_id();
  const auto num_parties = comm_layer.get_num_parties();
  const auto &R = backend_.get_bmr_provider().get_global_offset();
  ENCRYPTO::BitVector<> buffer;

  // mask and publish inputs
  for (auto i = 0ull; i < output_wires_.size(); ++i) {
    auto gmw_in = std::dynamic_pointer_cast<const Wires::GMWWire>(parent_.at(i));
    assert(gmw_in);
    auto bmr_out = std::dynamic_pointer_cast<Wires::BMRWire>(output_wires_.at(i));
    assert(bmr_out);
    gmw_in->GetIsReadyCondition().Wait();
    bmr_out->GetMutablePublicValues() = gmw_in->GetValues() ^ bmr_out->GetPermutationBits();
    buffer.Append(bmr_out->GetPublicValues());
  }
  const std::vector<std::uint8_t> payload_pub_vals(
      reinterpret_cast<const std::uint8_t *>(buffer.GetData().data()),
      reinterpret_cast<const std::uint8_t *>(buffer.GetData().data()) + buffer.GetData().size());
  comm_layer.broadcast_message(Communication::BuildBMRInput0Message(gate_id_, payload_pub_vals));

  // receive masked values if not my input
  for (auto party_id = 0ull; party_id < num_parties; ++party_id) {
    if (party_id == my_id) continue;
    buffer = received_public_values_.at(party_id).get();
    for (auto i = 0ull; i < output_wires_.size(); ++i) {
      auto bmr_out = std::dynamic_pointer_cast<Wires::BMRWire>(output_wires_.at(i));
      assert(bmr_out);
      bmr_out->GetMutablePublicValues() ^= buffer.Subset(i * num_simd, (i + 1) * num_simd);
    }
  }

  // rearrange keys corresponding to the public values into one buffer
  ENCRYPTO::block128_vector my_keys_buffer(num_wires * num_simd);
  for (auto wire_i = 0ull; wire_i < num_wires; ++wire_i) {
    const auto wire = std::dynamic_pointer_cast<const Wires::BMRWire>(output_wires_.at(wire_i));
    assert(wire);
    const auto &keys = wire->GetSecretKeys();
    // copy the "0 keys" into the buffer
    std::copy(std::begin(keys), std::end(keys), std::begin(my_keys_buffer) + wire_i * num_simd);
    const auto &public_values = wire->GetPublicValues();
    for (auto simd_j = 0ull; simd_j < num_simd; ++simd_j) {
      // xor the offset on a key if the corresponding public value is 1
      if (public_values[simd_j]) {
        my_keys_buffer.at(wire_i * num_simd + simd_j) ^= R;
      }
    }
  }

  // send the selected keys to all other parties
  const std::vector<std::uint8_t> payload(
      reinterpret_cast<const std::uint8_t *>(my_keys_buffer.data()),
      reinterpret_cast<const std::uint8_t *>(my_keys_buffer.data()) + my_keys_buffer.byte_size());
  comm_layer.broadcast_message(Communication::BuildBMRInput1Message(gate_id_, payload));

  // index function for the public/active keys stored in the wires
  const auto pk_index = [num_parties](auto simd_i, auto party_i) {
    return simd_i * num_parties + party_i;
  };

  // receive the published keys from the other parties
  // and construct the active super keys for the output wires
  for (auto party_i = 0ull; party_i < num_parties; ++party_i) {
    if (party_i == my_id) {
      // our case: we can copy the keys we have already prepared above in
      // my_keys_buffer to the right positions
      for (auto wire_j = 0ull; wire_j < num_wires; ++wire_j) {
        auto wire = std::dynamic_pointer_cast<Wires::BMRWire>(output_wires_.at(wire_j));
        assert(wire);
        auto &public_keys = wire->GetMutablePublicKeys();
        for (auto simd_k = 0ull; simd_k < num_simd; ++simd_k) {
          public_keys.at(pk_index(simd_k, my_id)) = my_keys_buffer.at(wire_j * num_simd + simd_k);
        }
      }
    } else {
      // other party: we copy the received keys to the right position
      auto received_keys_buffer = received_public_keys_.at(party_i).get();
      assert(received_keys_buffer.size() == num_wires * num_simd);
      for (auto wire_j = 0ull; wire_j < num_wires; ++wire_j) {
        auto wire = std::dynamic_pointer_cast<Wires::BMRWire>(output_wires_.at(wire_j));
        assert(wire);
        auto &public_keys = wire->GetMutablePublicKeys();
        for (auto simd_k = 0ull; simd_k < num_simd; ++simd_k) {
          public_keys.at(pk_index(simd_k, party_i)) =
              received_keys_buffer.at(wire_j * num_simd + simd_k);
        }
      }
    }
  }

  if constexpr (MOTION_DEBUG) {
    GetLogger().LogDebug(fmt::format(
        "Finished evaluating online phase of Boolean GMW to BMR Gate with id#{}", gate_id_));
  }
  SetOnlineIsReady();
  GetRegister().IncrementEvaluatedGatesOnlineCounter();
}

const Shares::BMRSharePtr GMWToBMRGate::GetOutputAsBMRShare() const {
  auto result = std::make_shared<Shares::BMRShare>(output_wires_);
  assert(result);
  return result;
}

const Shares::SharePtr GMWToBMRGate::GetOutputAsShare() const {
  auto result = std::static_pointer_cast<Shares::Share>(GetOutputAsBMRShare());
  assert(result);
  return result;
}

AGMWToBMRGate::AGMWToBMRGate(const Shares::SharePtr &parent) : OneGate(parent->GetBackend()) {
  parent_ = parent->GetWires();

  assert(parent_.size() == 1);
  assert(parent_[0]->GetBitLength() > 0);
  for ([[maybe_unused]] const auto &wire : parent_)
    assert(wire->GetProtocol() == MPCProtocol::ArithmeticGMW);

  requires_online_interaction_ = true;
  gate_type_ = GateType::Interactive;
  gate_id_ = GetRegister().NextGateId();

  // AGMWToBMRGate does not own its output wires, since these are the output wires of the last
  // BMR addition circuit. Thus, Gate::SetOnlineReady should not mark the output wires online-ready.
  own_output_wires_ = false;

  for (auto &wire : parent_) {
    RegisterWaitingFor(wire->GetWireId());
    wire->RegisterWaitingGate(gate_id_);
  }

  assert(gate_id_ >= 0);
  const auto& comm_layer = get_communication_layer();
  const auto my_id = comm_layer.get_my_id();
  const auto num_parties = comm_layer.get_num_parties();
  const auto bitlen{parent_[0]->GetBitLength()};
  const auto num_simd{parent_[0]->GetNumOfSIMDValues()};

  std::vector<SecureUnsignedInteger> shares;
  shares.reserve(num_parties);
  // each party re-shares its arithmetic GMW share in BMR
  for (auto party_id = 0ull; party_id < num_parties; ++party_id) {
    const auto in_gate =
        std::make_shared<MOTION::Gates::BMR::BMRInputGate>(num_simd, bitlen, party_id, backend_);
    GetRegister().RegisterNextInputGate(in_gate);
    // the party owning the share takes the input promise to assign its input when the parent wires
    // are online-ready
    if (party_id == my_id) input_promise_ = &in_gate->GetInputPromise();

    shares.emplace_back(Shares::ShareWrapper(in_gate->GetOutputAsShare()));
  }

  // securely compute the sum of the arithmetic GMW shares to get a valid BMR share
  auto result{shares[0]};
  for (auto share_i = 1ull; share_i < shares.size(); ++share_i) result += shares[share_i];

  // the sum of the shares is a valid BMR share, which output wires are the output wires of the
  // AGMW to BMR conversion gate
  output_wires_ = result.Get()->GetWires();

  if constexpr (MOTION_DEBUG) {
    auto gate_info = fmt::format("gate id {}, parent wires: ", gate_id_);
    for (const auto &wire : parent_) gate_info.append(fmt::format("{} ", wire->GetWireId()));
    gate_info.append(" output wires: ");
    for (const auto &wire : output_wires_) gate_info.append(fmt::format("{} ", wire->GetWireId()));
    GetLogger().LogDebug(
        fmt::format("Created a Arithmetic GMW to BMR conversion gate with following properties: {}",
                    gate_info));
  }
}

void AGMWToBMRGate::EvaluateSetup() {
  if constexpr (MOTION_DEBUG) {
    GetLogger().LogDebug(fmt::format(
        "Nothing to do in the setup phase of Arithmetic GMW to BMR Gate with id#{}", gate_id_));
  }

  SetSetupIsReady();
  GetRegister().IncrementEvaluatedGatesSetupCounter();
}

void AGMWToBMRGate::EvaluateOnline() {
  WaitSetup();
  if constexpr (MOTION_DEBUG) {
    GetLogger().LogDebug(fmt::format(
        "Start evaluating online phase of Boolean GMW to BMR Gate with id#{}", gate_id_));
  }

  const auto bitlen = parent_[0]->GetBitLength();
  parent_[0]->GetIsReadyCondition().Wait();

  switch (bitlen) {
    case 8: {
      auto w{std::dynamic_pointer_cast<MOTION::Wires::ArithmeticWire<std::uint8_t>>(parent_[0])};
      assert(w);
      input_promise_->set_value(ENCRYPTO::ToInput(w->GetValues()));
      break;
    }
    case 16: {
      auto w{std::dynamic_pointer_cast<MOTION::Wires::ArithmeticWire<std::uint16_t>>(parent_[0])};
      assert(w);
      input_promise_->set_value(ENCRYPTO::ToInput(w->GetValues()));
      break;
    }
    case 32: {
      auto w{std::dynamic_pointer_cast<MOTION::Wires::ArithmeticWire<std::uint32_t>>(parent_[0])};
      assert(w);
      input_promise_->set_value(ENCRYPTO::ToInput(w->GetValues()));
      break;
    }
    case 64: {
      auto w{std::dynamic_pointer_cast<MOTION::Wires::ArithmeticWire<std::uint64_t>>(parent_[0])};
      assert(w);
      input_promise_->set_value(ENCRYPTO::ToInput(w->GetValues()));
      break;
    }
    default:
      throw std::logic_error(fmt::format("Illegal bitlength: {}", bitlen));
  }

  if constexpr (MOTION_DEBUG) {
    GetLogger().LogDebug(fmt::format(
        "Finished evaluating online phase of Boolean GMW to BMR Gate with id#{}", gate_id_));
  }
  SetOnlineIsReady();
  GetRegister().IncrementEvaluatedGatesOnlineCounter();
}

const Shares::BMRSharePtr AGMWToBMRGate::GetOutputAsBMRShare() const {
  auto result = std::make_shared<Shares::BMRShare>(output_wires_);
  assert(result);
  return result;
}

const Shares::SharePtr AGMWToBMRGate::GetOutputAsShare() const {
  auto result = std::static_pointer_cast<Shares::Share>(GetOutputAsBMRShare());
  assert(result);
  return result;
}

}  // namespace MOTION::Gates::Conversion
