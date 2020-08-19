// IT License
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

#include "gate/new_gate.h"
#include "tensor.h"
#include "tensor/tensor_op.h"
#include "utility/reusable_future.h"

namespace ENCRYPTO::ObliviousTransfer {
template <typename T>
class ACOTSender;
template <typename T>
class ACOTReceiver;
}

namespace MOTION::proto::gmw {

class GMWProvider;

template <typename T>
class ArithmeticGMWTensorInputSender : public NewGate {
 public:
  ArithmeticGMWTensorInputSender(std::size_t gate_id, GMWProvider&,
                                 const tensor::TensorDimensions& dimensions,
                                 ENCRYPTO::ReusableFiberFuture<std::vector<T>>&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  std::shared_ptr<const ArithmeticGMWTensor<T>> get_output_tensor() const noexcept {
    return output_;
  }

 private:
  GMWProvider& gmw_provider_;
  std::size_t input_id_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> input_future_;
  ArithmeticGMWTensorP<T> output_;
  constexpr static std::size_t bit_size_ = ENCRYPTO::bit_size_v<T>;
};

template <typename T>
class ArithmeticGMWTensorInputReceiver : public NewGate {
 public:
  ArithmeticGMWTensorInputReceiver(std::size_t gate_id, GMWProvider&,
                                   const tensor::TensorDimensions& dimensions);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return false; }
  void evaluate_setup() override;
  void evaluate_online() override {}
  std::shared_ptr<const ArithmeticGMWTensor<T>> get_output_tensor() const noexcept {
    return output_;
  }

 private:
  GMWProvider& gmw_provider_;
  std::size_t input_id_;
  ArithmeticGMWTensorP<T> output_;
  constexpr static std::size_t bit_size_ = ENCRYPTO::bit_size_v<T>;
};

template <typename T>
class ArithmeticGMWTensorOutput : public NewGate {
 public:
  ArithmeticGMWTensorOutput(std::size_t gate_id, GMWProvider&, ArithmeticGMWTensorCP<T>,
                            std::size_t output_owner);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> get_output_future();

 private:
  GMWProvider& gmw_provider_;
  ENCRYPTO::ReusableFiberPromise<std::vector<T>> output_promise_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
  std::size_t output_owner_;
  const ArithmeticGMWTensorCP<T> input_;
};

template <typename T>
class ArithmeticGMWTensorFlatten : public NewGate {
 public:
  ArithmeticGMWTensorFlatten(std::size_t gate_id, GMWProvider&, std::size_t axis,
                             const ArithmeticGMWTensorCP<T> input);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  const ArithmeticGMWTensorP<T>& get_output_tensor() const { return output_; }

 private:
  GMWProvider& gmw_provider_;
  const ArithmeticGMWTensorCP<T> input_;
  std::shared_ptr<ArithmeticGMWTensor<T>> output_;
};

template <typename T>
class ArithmeticGMWTensorConv2D : public NewGate {
 public:
  ArithmeticGMWTensorConv2D(std::size_t gate_id, GMWProvider&, tensor::Conv2DOp,
                            const ArithmeticGMWTensorCP<T> input,
                            const ArithmeticGMWTensorCP<T> kernel,
                            const ArithmeticGMWTensorCP<T> bias);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  const ArithmeticGMWTensorP<T>& get_output_tensor() const { return output_; }

 private:
  GMWProvider& gmw_provider_;
  tensor::Conv2DOp conv_op_;
  const ArithmeticGMWTensorCP<T> input_;
  const ArithmeticGMWTensorCP<T> kernel_;
  const ArithmeticGMWTensorCP<T> bias_;
  std::shared_ptr<ArithmeticGMWTensor<T>> output_;
  std::size_t triple_index_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
};

template <typename T>
class ArithmeticGMWTensorGemm : public NewGate {
 public:
  ArithmeticGMWTensorGemm(std::size_t gate_id, GMWProvider&, tensor::GemmOp,
                          const ArithmeticGMWTensorCP<T> input_A,
                          const ArithmeticGMWTensorCP<T> input_B);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  const ArithmeticGMWTensorP<T>& get_output_tensor() const { return output_; }

 private:
  GMWProvider& gmw_provider_;
  tensor::GemmOp gemm_op_;
  const ArithmeticGMWTensorCP<T> input_A_;
  const ArithmeticGMWTensorCP<T> input_B_;
  std::shared_ptr<ArithmeticGMWTensor<T>> output_;
  std::size_t triple_index_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
};

template <typename T>
class ArithmeticGMWTensorSqr : public NewGate {
 public:
  ArithmeticGMWTensorSqr(std::size_t gate_id, GMWProvider&, const ArithmeticGMWTensorCP<T> input);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  const ArithmeticGMWTensorP<T>& get_output_tensor() const { return output_; }

 private:
  GMWProvider& gmw_provider_;
  std::size_t data_size_;
  const ArithmeticGMWTensorCP<T> input_;
  std::shared_ptr<ArithmeticGMWTensor<T>> output_;
  std::size_t triple_index_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
};

template <typename T>
class BooleanToArithmeticGMWTensorConversion : public NewGate {
 public:
  BooleanToArithmeticGMWTensorConversion(std::size_t gate_id, GMWProvider&,
                                         const BooleanGMWTensorCP input);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  gmw::ArithmeticGMWTensorCP<T> get_output_tensor() const noexcept { return output_; }

 private:
  GMWProvider& gmw_provider_;
  static constexpr auto bit_size_ = ENCRYPTO::bit_size_v<T>;
  const std::size_t data_size_;
  const BooleanGMWTensorCP input_;
  ArithmeticGMWTensorP<T> output_;
  std::size_t sb_offset_;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> t_share_future_;
};

class BooleanGMWTensorRelu : public NewGate {
 public:
  BooleanGMWTensorRelu(std::size_t gate_id, GMWProvider&, const BooleanGMWTensorCP input);
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  const BooleanGMWTensorP& get_output_tensor() const { return output_; }

 private:
  GMWProvider& gmw_provider_;
  const std::size_t bit_size_;
  const std::size_t data_size_;
  const BooleanGMWTensorCP input_;
  BooleanGMWTensorP output_;
  const std::size_t triple_index_;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> share_future_;
};

template <typename T>
class BooleanXArithmeticGMWTensorRelu : public NewGate {
 public:
  BooleanXArithmeticGMWTensorRelu(std::size_t gate_id, GMWProvider&, const BooleanGMWTensorCP,
                                  const ArithmeticGMWTensorCP<T>);
  ~BooleanXArithmeticGMWTensorRelu();
  bool need_setup() const noexcept override { return false; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override {}
  void evaluate_online() override;
  const ArithmeticGMWTensorP<T>& get_output_tensor() const { return output_; }

 private:
  GMWProvider& gmw_provider_;
  static constexpr auto bit_size_ = ENCRYPTO::bit_size_v<T>;
  const std::size_t data_size_;
  const BooleanGMWTensorCP input_bool_;
  const ArithmeticGMWTensorCP<T> input_arith_;
  ArithmeticGMWTensorP<T> output_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTSender<T>> ot_sender_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTReceiver<T>> ot_receiver_;
};

}  // namespace MOTION::proto::gmw
