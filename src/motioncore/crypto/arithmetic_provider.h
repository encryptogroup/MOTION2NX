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

#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "oblivious_transfer/ot_provider.h"
#include "utility/reusable_future.h"
#include "utility/type_traits.hpp"

namespace ENCRYPTO::ObliviousTransfer {
class OTProvider;
class OTProviderManager;
template <typename T>
class ACOTSender;
template <typename T>
class ACOTReceiver;
}  // namespace ENCRYPTO::ObliviousTransfer

namespace MOTION {

namespace Communication {
class CommunicationLayer;
}

class Logger;

template <typename T>
class IntegerMultiplicationSender {
 public:
  IntegerMultiplicationSender(std::size_t batch_size, ENCRYPTO::ObliviousTransfer::OTProvider&);
  ~IntegerMultiplicationSender();
  void set_inputs(std::vector<T>&& inputs);
  void set_inputs(const std::vector<T>& inputs);
  void set_inputs(const T* inputs);
  void compute_outputs();
  std::vector<T> get_outputs();

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::size_t batch_size_;
  std::vector<T> inputs_;
  std::vector<T> outputs_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTSender<T>> ot_sender_;
};

template <typename T>
class IntegerMultiplicationReceiver {
 public:
  IntegerMultiplicationReceiver(std::size_t batch_size, ENCRYPTO::ObliviousTransfer::OTProvider&);
  ~IntegerMultiplicationReceiver();
  void set_inputs(std::vector<T>&& inputs);
  void set_inputs(const std::vector<T>& inputs);
  void set_inputs(const T* inputs);
  void compute_outputs();
  std::vector<T> get_outputs();

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::size_t batch_size_;
  std::vector<T> inputs_;
  std::vector<T> outputs_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTReceiver<T>> ot_receiver_;
  std::shared_ptr<Logger> logger_;
};

class ArithmeticProvider {
 public:
  ArithmeticProvider(ENCRYPTO::ObliviousTransfer::OTProvider&, std::shared_ptr<Logger>);

  template <typename T>
  std::unique_ptr<IntegerMultiplicationSender<T>> register_integer_multiplication_send(
      std::size_t batch_size);
  template <typename T>
  std::unique_ptr<IntegerMultiplicationReceiver<T>> register_integer_multiplication_receive(
      std::size_t batch_size);

 private:
  ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider_;
  std::shared_ptr<Logger> logger_;
};

class ArithmeticProviderManager {
 public:
  ArithmeticProviderManager(MOTION::Communication::CommunicationLayer&,
                            ENCRYPTO::ObliviousTransfer::OTProviderManager&,
                            std::shared_ptr<Logger>);
  ~ArithmeticProviderManager();

  ArithmeticProvider& get_provider(std::size_t party_id) { return *providers_.at(party_id); }

 private:
  MOTION::Communication::CommunicationLayer& comm_layer_;
  std::vector<std::unique_ptr<ArithmeticProvider>> providers_;
};

}  // namespace MOTION
