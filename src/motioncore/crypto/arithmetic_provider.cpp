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

#include "arithmetic_provider.h"

#include <stdexcept>
#include <vector>

#include "communication/communication_layer.h"
#include "oblivious_transfer/ot_flavors.h"
#include "oblivious_transfer/ot_provider.h"

namespace MOTION {

// ---------- IntegerMultiplicationSender ----------

template <typename T>
IntegerMultiplicationSender<T>::IntegerMultiplicationSender(
    std::size_t batch_size, std::size_t vector_size,
    ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider)
    : batch_size_(batch_size),
      vector_size_(vector_size),
      ot_sender_(
          ot_provider.RegisterSendACOT<T>(batch_size * ENCRYPTO::bit_size_v<T>, vector_size)) {}

template <typename T>
IntegerMultiplicationSender<T>::~IntegerMultiplicationSender() = default;

template <typename T>
void IntegerMultiplicationSender<T>::set_inputs(std::vector<T>&& inputs) {
  set_inputs(inputs);
}

template <typename T>
void IntegerMultiplicationSender<T>::set_inputs(const std::vector<T>& inputs) {
  if (inputs.size() != batch_size_ * vector_size_) {
    throw std::invalid_argument("input has unexpected size");
  }
  set_inputs(inputs.data());
}

template <typename T>
void IntegerMultiplicationSender<T>::set_inputs(const T* inputs) {
  constexpr auto bit_size = ENCRYPTO::bit_size_v<T>;

  const auto idx = [this](auto input_i, auto vector_enty_k, auto bit_j) {
    return input_i * vector_size_ * bit_size + bit_j * vector_size_ + vector_enty_k;
  };

  std::vector<T> ot_inputs(batch_size_ * ENCRYPTO::bit_size_v<T> * vector_size_);
  for (std::size_t input_i = 0; input_i < batch_size_; ++input_i) {
    for (std::size_t vector_enty_k = 0; vector_enty_k < vector_size_; ++vector_enty_k) {
      const T value = inputs[input_i * vector_size_ + vector_enty_k];
      for (std::size_t bit_j = 0; bit_j < bit_size; ++bit_j) {
        ot_inputs[idx(input_i, vector_enty_k, bit_j)] = value << bit_j;
      }
    }
  }
  ot_sender_->SetCorrelations(std::move(ot_inputs));
  ot_sender_->SendMessages();
}

template <typename T>
void IntegerMultiplicationSender<T>::compute_outputs() {
  constexpr auto bit_size = ENCRYPTO::bit_size_v<T>;

  const auto idx = [this](auto output_i, auto vector_enty_k, auto bit_j) {
    return output_i * vector_size_ * bit_size + bit_j * vector_size_ + vector_enty_k;
  };

  ot_sender_->ComputeOutputs();
  auto ot_outputs = ot_sender_->GetOutputs();
  assert(ot_outputs.size() == batch_size_ * vector_size_ * bit_size);
  outputs_.resize(batch_size_ * vector_size_);
  for (std::size_t output_i = 0; output_i < batch_size_; ++output_i) {
    for (std::size_t vector_enty_k = 0; vector_enty_k < vector_size_; ++vector_enty_k) {
      T value = 0;
      for (std::size_t bit_j = 0; bit_j < bit_size; ++bit_j) {
        value -= ot_outputs[idx(output_i, vector_enty_k, bit_j)];
      }
      outputs_[output_i * vector_size_ + vector_enty_k] = value;
    }
  }
}

template <typename T>
std::vector<T> IntegerMultiplicationSender<T>::get_outputs() {
  // TODO: check output is ready
  return std::move(outputs_);
}

// ---------- IntegerMultiplicationReceiver ----------

template <typename T>
IntegerMultiplicationReceiver<T>::IntegerMultiplicationReceiver(
    std::size_t batch_size, std::size_t vector_size,
    ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider)
    : batch_size_(batch_size),
      vector_size_(vector_size),
      ot_receiver_(
          ot_provider.RegisterReceiveACOT<T>(batch_size * ENCRYPTO::bit_size_v<T>, vector_size)) {}

template <typename T>
IntegerMultiplicationReceiver<T>::~IntegerMultiplicationReceiver() = default;

template <typename T>
void IntegerMultiplicationReceiver<T>::set_inputs(std::vector<T>&& inputs) {
  set_inputs(inputs);
}

template <typename T>
void IntegerMultiplicationReceiver<T>::set_inputs(const std::vector<T>& inputs) {
  if (inputs.size() != batch_size_) {
    throw std::invalid_argument("input has unexpected size");
  }
  set_inputs(inputs.data());
}

template <typename T>
void IntegerMultiplicationReceiver<T>::set_inputs(const T* inputs) {
  ENCRYPTO::BitVector<> ot_choices(batch_size_ * ENCRYPTO::bit_size_v<T>);
  std::copy_n(inputs, batch_size_, reinterpret_cast<T*>(ot_choices.GetMutableData().data()));
  ot_receiver_->SetChoices(std::move(ot_choices));
  ot_receiver_->SendCorrections();
}

template <typename T>
void IntegerMultiplicationReceiver<T>::compute_outputs() {
  constexpr auto bit_size = ENCRYPTO::bit_size_v<T>;

  const auto idx = [this](auto output_i, auto vector_enty_k, auto bit_j) {
    return output_i * vector_size_ * bit_size + bit_j * vector_size_ + vector_enty_k;
  };

  ot_receiver_->ComputeOutputs();
  auto ot_outputs = ot_receiver_->GetOutputs();
  assert(ot_outputs.size() == batch_size_ * vector_size_ * bit_size);
  outputs_.resize(batch_size_ * vector_size_);
  for (std::size_t output_i = 0; output_i < batch_size_; ++output_i) {
    for (std::size_t vector_enty_k = 0; vector_enty_k < vector_size_; ++vector_enty_k) {
      T value = 0;
      for (std::size_t bit_j = 0; bit_j < bit_size; ++bit_j) {
        value += ot_outputs[idx(output_i, vector_enty_k, bit_j)];
      }
      outputs_[output_i * vector_size_ + vector_enty_k] = value;
    }
  }
}

template <typename T>
std::vector<T> IntegerMultiplicationReceiver<T>::get_outputs() {
  // TODO: check output is ready
  return std::move(outputs_);
}

// ---------- ArithmeticProvider ----------

ArithmeticProvider::ArithmeticProvider(ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider,
                                       std::shared_ptr<Logger> logger)
    : ot_provider_(ot_provider), logger_(logger) {}

template <typename T>
std::unique_ptr<IntegerMultiplicationSender<T>>
ArithmeticProvider::register_integer_multiplication_send(std::size_t batch_size,
                                                         std::size_t vector_size) {
  return std::make_unique<IntegerMultiplicationSender<T>>(batch_size, vector_size, ot_provider_);
}

template <typename T>
std::unique_ptr<IntegerMultiplicationReceiver<T>>
ArithmeticProvider::register_integer_multiplication_receive(std::size_t batch_size,
                                                            std::size_t vector_size) {
  return std::make_unique<IntegerMultiplicationReceiver<T>>(batch_size, vector_size, ot_provider_);
}

// ---------- ArithmeticProviderManager ----------

ArithmeticProviderManager::ArithmeticProviderManager(
    MOTION::Communication::CommunicationLayer& comm_layer,
    ENCRYPTO::ObliviousTransfer::OTProviderManager& ot_provider_manager,
    std::shared_ptr<Logger> logger)
    : comm_layer_(comm_layer), providers_(comm_layer_.get_num_parties()) {
  auto my_id = comm_layer_.get_my_id();
  auto num_parties = comm_layer_.get_num_parties();

  for (std::size_t party_id = 0; party_id < num_parties; ++party_id) {
    if (party_id == my_id) {
      continue;
    }
    providers_.at(party_id) =
        std::make_unique<ArithmeticProvider>(ot_provider_manager.get_provider(party_id), logger);
  }
}

ArithmeticProviderManager::~ArithmeticProviderManager() = default;

// ---------- template instantiations ----------

template class IntegerMultiplicationSender<std::uint8_t>;
template class IntegerMultiplicationSender<std::uint16_t>;
template class IntegerMultiplicationSender<std::uint32_t>;
template class IntegerMultiplicationSender<std::uint64_t>;
template class IntegerMultiplicationSender<__uint128_t>;

template class IntegerMultiplicationReceiver<std::uint8_t>;
template class IntegerMultiplicationReceiver<std::uint16_t>;
template class IntegerMultiplicationReceiver<std::uint32_t>;
template class IntegerMultiplicationReceiver<std::uint64_t>;
template class IntegerMultiplicationReceiver<__uint128_t>;

template std::unique_ptr<IntegerMultiplicationSender<std::uint8_t>>
    ArithmeticProvider::register_integer_multiplication_send<std::uint8_t>(std::size_t,
                                                                           std::size_t);
template std::unique_ptr<IntegerMultiplicationSender<std::uint16_t>>
    ArithmeticProvider::register_integer_multiplication_send<std::uint16_t>(std::size_t,
                                                                            std::size_t);
template std::unique_ptr<IntegerMultiplicationSender<std::uint32_t>>
    ArithmeticProvider::register_integer_multiplication_send<std::uint32_t>(std::size_t,
                                                                            std::size_t);
template std::unique_ptr<IntegerMultiplicationSender<std::uint64_t>>
    ArithmeticProvider::register_integer_multiplication_send<std::uint64_t>(std::size_t,
                                                                            std::size_t);
template std::unique_ptr<IntegerMultiplicationSender<__uint128_t>>
    ArithmeticProvider::register_integer_multiplication_send<__uint128_t>(std::size_t, std::size_t);

template std::unique_ptr<IntegerMultiplicationReceiver<std::uint8_t>>
    ArithmeticProvider::register_integer_multiplication_receive<std::uint8_t>(std::size_t,
                                                                              std::size_t);
template std::unique_ptr<IntegerMultiplicationReceiver<std::uint16_t>>
    ArithmeticProvider::register_integer_multiplication_receive<std::uint16_t>(std::size_t,
                                                                               std::size_t);
template std::unique_ptr<IntegerMultiplicationReceiver<std::uint32_t>>
    ArithmeticProvider::register_integer_multiplication_receive<std::uint32_t>(std::size_t,
                                                                               std::size_t);
template std::unique_ptr<IntegerMultiplicationReceiver<std::uint64_t>>
    ArithmeticProvider::register_integer_multiplication_receive<std::uint64_t>(std::size_t,
                                                                               std::size_t);
template std::unique_ptr<IntegerMultiplicationReceiver<__uint128_t>>
    ArithmeticProvider::register_integer_multiplication_receive<__uint128_t>(std::size_t,
                                                                             std::size_t);

}  // namespace MOTION
