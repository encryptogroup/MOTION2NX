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

#include <cstddef>
#include <memory>
#include <vector>

#include "gate/new_gate.h"
#include "utility/bit_vector.h"
#include "utility/reusable_future.h"
#include "wire.h"

namespace ENCRYPTO::ObliviousTransfer {
template <typename T>
class ACOTSender;
template <typename T>
class ACOTReceiver;
}  // namespace ENCRYPTO::ObliviousTransfer

namespace MOTION::proto::gmw {
class BooleanGMWWire;
using BooleanGMWWireVector = std::vector<std::shared_ptr<BooleanGMWWire>>;
template <typename T>
class ArithmeticGMWWire;
template <typename T>
using ArithmeticGMWWireP = std::shared_ptr<ArithmeticGMWWire<T>>;
}  // namespace MOTION::proto::gmw

namespace MOTION::proto::beavy {

class BooleanBEAVYWire;
using BooleanBEAVYWireP = std::shared_ptr<BooleanBEAVYWire>;
using BooleanBEAVYWireVector = std::vector<BooleanBEAVYWireP>;
template <typename T>
class ArithmeticBEAVYWire;
template <typename T>
using ArithmeticBEAVYWireP = std::shared_ptr<ArithmeticBEAVYWire<T>>;

class BEAVYProvider;

template <typename T>
class BooleanBitToArithmeticBEAVYGate : public NewGate {
 public:
  BooleanBitToArithmeticBEAVYGate(std::size_t gate_id, BEAVYProvider&, BooleanBEAVYWireP);
  ~BooleanBitToArithmeticBEAVYGate();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  beavy::ArithmeticBEAVYWireP<T>& get_output_wire() noexcept { return output_; };

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  beavy::BooleanBEAVYWireP input_;
  beavy::ArithmeticBEAVYWireP<T> output_;
  BEAVYProvider& beavy_provider_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTSender<T>> ot_sender_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTReceiver<T>> ot_receiver_;
  std::vector<T> arithmetized_secret_share_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
};

template <typename T>
class BooleanToArithmeticBEAVYGate : public NewGate {
 public:
  BooleanToArithmeticBEAVYGate(std::size_t gate_id, BEAVYProvider&, BooleanBEAVYWireVector&&);
  ~BooleanToArithmeticBEAVYGate();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  beavy::ArithmeticBEAVYWireP<T>& get_output_wire() noexcept { return output_; };

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  beavy::BooleanBEAVYWireVector inputs_;
  beavy::ArithmeticBEAVYWireP<T> output_;
  BEAVYProvider& beavy_provider_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTSender<T>> ot_sender_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTReceiver<T>> ot_receiver_;
  std::vector<T> arithmetized_secret_share_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
};

class BooleanBEAVYToGMWGate : public NewGate {
 public:
  BooleanBEAVYToGMWGate(std::size_t gate_id, BEAVYProvider&, BooleanBEAVYWireVector&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  gmw::BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; };

 private:
  BEAVYProvider& beavy_provider_;
  BooleanBEAVYWireVector inputs_;
  gmw::BooleanGMWWireVector outputs_;
};

class BooleanGMWToBEAVYGate : public NewGate {
 public:
  BooleanGMWToBEAVYGate(std::size_t gate_id, BEAVYProvider&, gmw::BooleanGMWWireVector&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  BooleanBEAVYWireVector& get_output_wires() noexcept { return outputs_; };

 private:
  BEAVYProvider& beavy_provider_;
  gmw::BooleanGMWWireVector inputs_;
  BooleanBEAVYWireVector outputs_;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> share_future_;
};

template <typename T>
class ArithmeticBEAVYToGMWGate : public NewGate {
 public:
  ArithmeticBEAVYToGMWGate(std::size_t gate_id, BEAVYProvider&, ArithmeticBEAVYWireP<T>);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  gmw::ArithmeticGMWWireP<T>& get_output_wire() noexcept { return output_; };

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  BEAVYProvider& beavy_provider_;
  ArithmeticBEAVYWireP<T> input_;
  gmw::ArithmeticGMWWireP<T> output_;
};

template <typename T>
class ArithmeticGMWToBEAVYGate : public NewGate {
 public:
  ArithmeticGMWToBEAVYGate(std::size_t gate_id, BEAVYProvider&, gmw::ArithmeticGMWWireP<T>);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  ArithmeticBEAVYWireP<T>& get_output_wire() noexcept { return output_; };

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  BEAVYProvider& beavy_provider_;
  gmw::ArithmeticGMWWireP<T> input_;
  ArithmeticBEAVYWireP<T> output_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
};

}  // namespace MOTION::proto::beavy
