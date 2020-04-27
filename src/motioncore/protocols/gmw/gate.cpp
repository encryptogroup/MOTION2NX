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

#include "base/gate_factory.h"
#include "wire.h"

namespace MOTION::proto::gmw {

BooleanGMWInputGateSender::BooleanGMWInputGateSender(
    std::size_t gate_id, GMWProvider& gmw_provider, std::size_t num_wires, std::size_t num_simd,
    ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&& input_future)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      num_wires_(num_wires),
      num_simd_(num_simd),
      input_future_(std::move(input_future)) {
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<BooleanGMWWire>(num_simd); });
}

void BooleanGMWInputGateSender::evaluate_setup() {
  // TODO:
  // - for each other party:
  //  - get shared randomness generator from GMWProvider
  //  - generate their share and xor to share
}

void BooleanGMWInputGateSender::evaluate_online() {
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
      input_owner_(input_owner) {
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<BooleanGMWWire>(num_simd); });
}

void BooleanGMWInputGateReceiver::evaluate_setup() {
  // TODO:
  // - get shared randomness generator from GMWProvider
  // - generate my share
  // - set output wires online ready
}

void BooleanGMWInputGateReceiver::evaluate_online() {
  // nothing to do
}

BooleanGMWOutputGate::BooleanGMWOutputGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                           BooleanGMWWireVector&& inputs, std::size_t output_owner)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      output_owner_(output_owner),
      inputs_(std::move(inputs)) {
  std::size_t my_id;        // TODO: get from provider
  std::size_t num_parties;  // TODO: get from provider
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    share_futures_.reserve(num_parties);
    for (std::size_t party_id; party_id < num_parties; ++party_id) {
      if (party_id == my_id) {
        share_futures_.emplace_back();
      }
      // TODO: register for output message
    }
  }
}

ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>
BooleanGMWOutputGate::get_output_future() {
  std::size_t my_id;  // TODO: get from provider
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    return output_promise_.get_future();
  } else {
    throw std::logic_error("not this parties output");
  }
}

void BooleanGMWOutputGate::evaluate_setup() {
  // nothing to do
}

void BooleanGMWOutputGate::evaluate_online() {
  std::size_t my_id;  // TODO: get from provider
  if (output_owner_ == ALL_PARTIES) {
    // TODO: broadcast share
  } else if (output_owner_ != my_id) {
    // TODO: send share to output_owner_
  }
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    std::size_t num_parties;  // TODO: get from provider
    for (std::size_t party_id; party_id < num_parties; ++party_id) {
      if (party_id == my_id) {
        continue;
      }
      const auto other_share = share_futures_[party_id].get();
      // TODO: xor on my share
    }
    // TODO: set_value of output_promise_
  }
}

}  // namespace MOTION::proto::gmw
