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

#pragma once

#include <future>

#include "gate.h"

#include "boolean_gmw_gate.h"
#include "share/bmr_share.h"
#include "utility/bit_vector.h"
#include "utility/block.h"
#include "utility/reusable_future.h"

namespace ENCRYPTO::ObliviousTransfer {
class OTVectorSender;
class OTVectorReceiver;
class FixedXCOT128Sender;
class FixedXCOT128Receiver;
class XCOTBitSender;
class XCOTBitReceiver;
}  // namespace ENCRYPTO::ObliviousTransfer

namespace MOTION::Gates::BMR {

class BMRInputGate final : public Gates::Interfaces::InputGate {
 public:
  BMRInputGate(std::size_t num_simd, std::size_t bit_size, std::size_t input_owner_id,
               Backend &backend);

  BMRInputGate(const std::vector<ENCRYPTO::BitVector<>> &input, std::size_t input_owner_id,
               Backend &backend);

  BMRInputGate(std::vector<ENCRYPTO::BitVector<>> &&input, std::size_t input_owner_id,
               Backend &backend);

  void InitializationHelper();

  ~BMRInputGate() final = default;

  void EvaluateSetup() final;

  void EvaluateOnline() final;

  const Shares::BMRSharePtr GetOutputAsBMRShare() const;

  const Shares::SharePtr GetOutputAsShare() const;

  auto &GetInputPromise() { return input_promise_; }

 protected:
  std::size_t num_simd_{0};  ///< Number of parallel values on wires
  std::size_t bit_size_{0};  ///< Number of wires
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> received_public_values_;
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector>> received_public_keys_;
  ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>> input_future_;
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> input_promise_;
};

constexpr std::size_t ALL = std::numeric_limits<std::int64_t>::max();

class BMROutputGate final : public Interfaces::OutputGate {
 public:
  BMROutputGate(const Shares::SharePtr &parent, std::size_t output_owner = ALL);

  ~BMROutputGate() final = default;

  void EvaluateSetup() final;

  void EvaluateOnline() final;

  const Shares::BMRSharePtr GetOutputAsBMRShare() const;

  const Shares::SharePtr GetOutputAsShare() const;

 protected:
  MOTION::Shares::GMWSharePtr gmw_out_share_;
  std::shared_ptr<MOTION::Gates::GMW::GMWOutputGate> out_;

  std::vector<ENCRYPTO::BitVector<>> output_;
  std::vector<std::vector<ENCRYPTO::BitVector<>>> shared_outputs_;

  bool is_my_output_ = false;

  std::mutex m;
};

class BMRXORGate final : public Gates::Interfaces::TwoGate {
 public:
  BMRXORGate(const Shares::SharePtr &a, const Shares::SharePtr &b);

  ~BMRXORGate() final = default;

  void EvaluateSetup() final;

  void EvaluateOnline() final;

  const Shares::BMRSharePtr GetOutputAsBMRShare() const;

  const Shares::SharePtr GetOutputAsShare() const;

  BMRXORGate() = delete;

  BMRXORGate(const Gate &) = delete;
};

class BMRINVGate final : public Gates::Interfaces::OneGate {
 public:
  BMRINVGate(const Shares::SharePtr &parent);

  ~BMRINVGate() final = default;

  void EvaluateSetup() final;

  void EvaluateOnline() final;

  const Shares::BMRSharePtr GetOutputAsBMRShare() const;

  const Shares::SharePtr GetOutputAsShare() const;

  BMRINVGate() = delete;

  BMRINVGate(const Gate &) = delete;
};

class BMRANDGate final : public Gates::Interfaces::TwoGate {
 public:
  BMRANDGate(const Shares::SharePtr &a, const Shares::SharePtr &b);

  ~BMRANDGate() final;

  void EvaluateSetup() final;

  void EvaluateOnline() final;

  const Shares::BMRSharePtr GetOutputAsBMRShare() const;

  const Shares::SharePtr GetOutputAsShare() const;

  BMRANDGate() = delete;

  BMRANDGate(const Gate &) = delete;

 private:
  std::vector<std::vector<std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitSender>>>
      s_ots_1_;
  std::vector<std::vector<std::unique_ptr<ENCRYPTO::ObliviousTransfer::FixedXCOT128Sender>>>
      s_ots_kappa_;
  std::vector<std::vector<std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitReceiver>>>
      r_ots_1_;
  std::vector<std::vector<std::unique_ptr<ENCRYPTO::ObliviousTransfer::FixedXCOT128Receiver>>>
      r_ots_kappa_;

  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector>> received_garbled_rows_;

  // buffer to store all garbled tables for all wires
  // structure: wires X (simd X (row_00 || row_01 || row_10 || row_11))
  ENCRYPTO::block128_vector garbled_tables_;

  void GenerateRandomness();
};

}  // namespace MOTION::Gates::BMR
