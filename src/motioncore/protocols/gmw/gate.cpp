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
#include "gmw_provider.h"
#include "wire.h"

namespace MOTION::proto::gmw {

namespace detail {

template <typename WireType>
BasicGMWBinaryGate<WireType>::BasicGMWBinaryGate(std::size_t gate_id, GMWWireVector&& in_b,
                                                 GMWWireVector&& in_a)
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
                  [num_simd] { return std::make_shared<WireType>(num_simd); });
}

template class BasicGMWBinaryGate<BooleanGMWWire>;
template class BasicGMWBinaryGate<ArithmeticGMWWire<std::uint8_t>>;
template class BasicGMWBinaryGate<ArithmeticGMWWire<std::uint16_t>>;
template class BasicGMWBinaryGate<ArithmeticGMWWire<std::uint32_t>>;
template class BasicGMWBinaryGate<ArithmeticGMWWire<std::uint64_t>>;

template <typename WireType>
BasicGMWUnaryGate<WireType>::BasicGMWUnaryGate(std::size_t gate_id, GMWWireVector&& in,
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
                    [num_simd] { return std::make_shared<WireType>(num_simd); });
  }
}

template class BasicGMWUnaryGate<BooleanGMWWire>;
template class BasicGMWUnaryGate<ArithmeticGMWWire<std::uint8_t>>;
template class BasicGMWUnaryGate<ArithmeticGMWWire<std::uint16_t>>;
template class BasicGMWUnaryGate<ArithmeticGMWWire<std::uint32_t>>;
template class BasicGMWUnaryGate<ArithmeticGMWWire<std::uint64_t>>;

}  // namespace detail

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

BooleanGMWINVGate::BooleanGMWINVGate(std::size_t gate_id, const GMWProvider& gmw_provider,
                                     GMWWireVector&& in)
    : detail::BasicGMWUnaryGate<BooleanGMWWire>(gate_id, std::move(in),
                                                gmw_provider.is_my_job(gate_id)),
      is_my_job_(gmw_provider.is_my_job(gate_id)) {}

void BooleanGMWINVGate::evaluate_setup() {
  // nothing to do
}

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

void BooleanGMWXORGate::evaluate_setup() {
  // nothing to do
}

void BooleanGMWXORGate::evaluate_online() {
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& w_a = inputs_a_[wire_i];
    const auto& w_b = inputs_b_[wire_i];
    w_a->wait_online();
    w_b->wait_online();
    auto& w_o = outputs_[wire_i];
    w_o->get_share() = w_a->get_share() ^ w_b->get_share();
    w_o->set_online_ready();
  }
}

BooleanGMWANDGate::BooleanGMWANDGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                     GMWWireVector&& in_a, GMWWireVector&& in_b)
    : detail::BasicGMWBinaryGate<BooleanGMWWire>(gate_id, std::move(in_a), std::move(in_b)),
      gmw_provider_(gmw_provider) {
  // TODO: register MTs
}

void BooleanGMWANDGate::evaluate_setup() {
  // TODO: wait for MTs
}

void BooleanGMWANDGate::evaluate_online() {
  // TODO: compute AND
}

template <typename T>
ArithmeticGMWNEGGate<T>::ArithmeticGMWNEGGate(std::size_t gate_id, const GMWProvider&,
                                              GMWWireVector&& in)
    : detail::BasicGMWUnaryGate<ArithmeticGMWWire<T>>(gate_id, std::move(in), false) {}

template <typename T>
void ArithmeticGMWNEGGate<T>::evaluate_setup() {
  // nothing to do
}

template <typename T>
void ArithmeticGMWNEGGate<T>::evaluate_online() {
  for (std::size_t wire_i = 0; wire_i < this->num_wires_; ++wire_i) {
    const auto& w_in = this->inputs_[wire_i];
    w_in->wait_online();
    auto& w_o = this->outputs_[wire_i];
    assert(w_o->get_share().size() == w_in->get_num_simd());
    std::transform(std::begin(w_in->get_share()), std::end(w_in->get_share()),
                   std::begin(w_o->get_share()), [](auto x) { return -x; });
    w_o->set_online_ready();
  }
}

template class ArithmeticGMWNEGGate<std::uint8_t>;
template class ArithmeticGMWNEGGate<std::uint16_t>;
template class ArithmeticGMWNEGGate<std::uint32_t>;
template class ArithmeticGMWNEGGate<std::uint64_t>;

template <typename T>
void ArithmeticGMWADDGate<T>::evaluate_setup() {
  // nothing to do
}

template <typename T>
void ArithmeticGMWADDGate<T>::evaluate_online() {
  for (std::size_t wire_i = 0; wire_i < this->num_wires_; ++wire_i) {
    const auto& w_a = this->inputs_a_[wire_i];
    const auto& w_b = this->inputs_b_[wire_i];
    w_a->wait_online();
    w_b->wait_online();
    auto& w_o = this->outputs_[wire_i];
    std::transform(std::begin(w_a->get_share()), std::end(w_a->get_share()),
                   std::begin(w_b->get_share()), std::begin(w_o->get_share()),
                   [](auto x, auto y) { return x + y; });
    w_o->set_online_ready();
  }
}

template class ArithmeticGMWADDGate<std::uint8_t>;
template class ArithmeticGMWADDGate<std::uint16_t>;
template class ArithmeticGMWADDGate<std::uint32_t>;
template class ArithmeticGMWADDGate<std::uint64_t>;

template <typename T>
ArithmeticGMWMULGate<T>::ArithmeticGMWMULGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                              GMWWireVector&& in_a, GMWWireVector&& in_b)
    : detail::BasicGMWBinaryGate<ArithmeticGMWWire<T>>(gate_id, std::move(in_a), std::move(in_b)),
      gmw_provider_(gmw_provider) {
  // TODO: register MTs
}

template <typename T>
void ArithmeticGMWMULGate<T>::evaluate_setup() {
  // TODO: wait for MTs
}

template <typename T>
void ArithmeticGMWMULGate<T>::evaluate_online() {
  // TODO: compute MUL
}

template class ArithmeticGMWMULGate<std::uint8_t>;
template class ArithmeticGMWMULGate<std::uint16_t>;
template class ArithmeticGMWMULGate<std::uint32_t>;
template class ArithmeticGMWMULGate<std::uint64_t>;

template <typename T>
ArithmeticGMWSQRGate<T>::ArithmeticGMWSQRGate(std::size_t gate_id, GMWProvider& gmw_provider,
                                              GMWWireVector&& in_a, GMWWireVector&& in_b)
    : detail::BasicGMWBinaryGate<ArithmeticGMWWire<T>>(gate_id, std::move(in_a), std::move(in_b)),
      gmw_provider_(gmw_provider) {
  // TODO: register MTs
}

template <typename T>
void ArithmeticGMWSQRGate<T>::evaluate_setup() {
  // TODO: wait for MTs
}

template <typename T>
void ArithmeticGMWSQRGate<T>::evaluate_online() {
  // TODO: compute SQR
}

template class ArithmeticGMWSQRGate<std::uint8_t>;
template class ArithmeticGMWSQRGate<std::uint16_t>;
template class ArithmeticGMWSQRGate<std::uint32_t>;
template class ArithmeticGMWSQRGate<std::uint64_t>;

}  // namespace MOTION::proto::gmw
