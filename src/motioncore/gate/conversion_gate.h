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

#include "gate.h"

#include "utility/bit_vector.h"
#include "utility/block.h"
#include "utility/reusable_future.h"

namespace MOTION {

namespace Shares {
class Share;
using SharePtr = std::shared_ptr<Shares::Share>;

class ShareWrapper;

class GMWShare;
using GMWSharePtr = std::shared_ptr<Shares::GMWShare>;

class BMRShare;
using BMRSharePtr = std::shared_ptr<Shares::BMRShare>;
}  // namespace Shares

namespace Gates::Conversion {

class BMRToGMWGate final : public Gates::Interfaces::OneGate {
 public:
  BMRToGMWGate(const Shares::SharePtr &parent);

  ~BMRToGMWGate() final = default;

  void EvaluateSetup() final;

  void EvaluateOnline() final;

  const Shares::GMWSharePtr GetOutputAsGMWShare() const;

  const Shares::SharePtr GetOutputAsShare() const;

  BMRToGMWGate() = delete;

  BMRToGMWGate(const Gate &) = delete;
};

class GMWToBMRGate final : public Gates::Interfaces::OneGate {
 public:
  GMWToBMRGate(const Shares::SharePtr &parent);

  ~GMWToBMRGate() final = default;

  void EvaluateSetup() final;

  void EvaluateOnline() final;

  const Shares::BMRSharePtr GetOutputAsBMRShare() const;

  const Shares::SharePtr GetOutputAsShare() const;

  GMWToBMRGate() = delete;

  GMWToBMRGate(const Gate &) = delete;

 private:
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>> received_public_values_;
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector>> received_public_keys_;
};

class AGMWToBMRGate final : public Gates::Interfaces::OneGate {
 public:
  AGMWToBMRGate(const Shares::SharePtr &parent);

  ~AGMWToBMRGate() final = default;

  void EvaluateSetup() final;

  void EvaluateOnline() final;

  const Shares::BMRSharePtr GetOutputAsBMRShare() const;

  const Shares::SharePtr GetOutputAsShare() const;

  AGMWToBMRGate() = delete;

  AGMWToBMRGate(const Gate &) = delete;

 private:
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> * input_promise_;
};

}  // namespace Gates::Conversion
}  // namespace MOTION
