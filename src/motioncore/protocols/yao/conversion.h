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
#include "utility/block.h"
#include "utility/reusable_future.h"

namespace ENCRYPTO::ObliviousTransfer {
class GOT128Receiver;
class GOT128Sender;
}

namespace MOTION::proto {

namespace gmw {
class BooleanGMWWire;
using BooleanGMWWireVector = std::vector<std::shared_ptr<BooleanGMWWire>>;
template <typename T>
class ArithmeticGMWWire;
template <typename T>
using ArithmeticGMWWireP = std::shared_ptr<ArithmeticGMWWire<T>>;
}  // namespace gmw

namespace yao {

class YaoProvider;
class YaoWire;
using YaoWireVector = std::vector<std::shared_ptr<YaoWire>>;

class YaoToBooleanGMWGateGarbler : public NewGate {
 public:
  YaoToBooleanGMWGateGarbler(std::size_t gate_id, YaoProvider&, YaoWireVector&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return false; }
  void evaluate_setup() override;
  void evaluate_online() override;
  gmw::BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; };

 private:
  const YaoWireVector inputs_;
  gmw::BooleanGMWWireVector outputs_;
};

class YaoToBooleanGMWGateEvaluator : public NewGate {
 public:
  YaoToBooleanGMWGateEvaluator(std::size_t gate_id, YaoProvider&, YaoWireVector&&);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  gmw::BooleanGMWWireVector& get_output_wires() noexcept { return outputs_; };

 private:
  const YaoWireVector inputs_;
  gmw::BooleanGMWWireVector outputs_;
};

class BooleanGMWToYaoGateGarbler : public NewGate {
 public:
  BooleanGMWToYaoGateGarbler(std::size_t gate_id, YaoProvider&, gmw::BooleanGMWWireVector&&);
  ~BooleanGMWToYaoGateGarbler();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  YaoWireVector& get_output_wires() noexcept { return outputs_; };

 private:
  const YaoProvider& yao_provider_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::GOT128Sender> ot_sender_;
  ENCRYPTO::block128_vector ot_inputs_;
  const gmw::BooleanGMWWireVector inputs_;
  YaoWireVector outputs_;
};

class BooleanGMWToYaoGateEvaluator : public NewGate {
 public:
  BooleanGMWToYaoGateEvaluator(std::size_t gate_id, YaoProvider&, gmw::BooleanGMWWireVector&&);
  ~BooleanGMWToYaoGateEvaluator();
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  YaoWireVector& get_output_wires() noexcept { return outputs_; };

 private:
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::GOT128Receiver> ot_receiver_;
  const gmw::BooleanGMWWireVector inputs_;
  YaoWireVector outputs_;
};

template <typename T>
class YaoToArithmeticGMWGateGarbler : public NewGate {
 public:
  YaoToArithmeticGMWGateGarbler(std::size_t gate_id, YaoProvider&, std::size_t num_simd);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return false; }
  void evaluate_setup() override;
  void evaluate_online() override {}
  ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>> get_mask_future() noexcept {
    return mask_promise_.get_future();
  };
  gmw::ArithmeticGMWWireP<T>& get_output_wire() noexcept { return output_; };

 private:
  gmw::ArithmeticGMWWireP<T> output_;
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> mask_promise_;
  YaoProvider& yao_provider_;
};

template <typename T>
class YaoToArithmeticGMWGateEvaluator : public NewGate {
 public:
  YaoToArithmeticGMWGateEvaluator(std::size_t gate_id, YaoProvider&, std::size_t num_simd);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  gmw::ArithmeticGMWWireP<T>& get_output_wire() noexcept { return output_; };
  void set_masked_value_future(
      ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&& future) {
    masked_value_future_ = std::move(future);
  }

 private:
  gmw::ArithmeticGMWWireP<T> output_;
  ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>> masked_value_future_;
  YaoProvider& yao_provider_;
};

}  // namespace yao
}  // namespace MOTION::proto
