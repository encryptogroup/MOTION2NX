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

namespace ENCRYPTO {

struct AlgorithmDescription;

namespace ObliviousTransfer {
template <typename T>
class ACOTSender;
template <typename T>
class ACOTReceiver;
class XCOTBitSender;
class XCOTBitReceiver;
}  // namespace ObliviousTransfer

}  // namespace ENCRYPTO

namespace MOTION {
template <typename T>
class BitIntegerMultiplicationIntSide;
template <typename T>
class BitIntegerMultiplicationBitSide;
template <typename T>
class ConvolutionInputSide;
template <typename T>
class ConvolutionKernelSide;
template <typename T>
class IntegerMultiplicationSender;
template <typename T>
class IntegerMultiplicationReceiver;
template <typename T>
class MatrixMultiplicationLHS;
template <typename T>
class MatrixMultiplicationRHS;
}  // namespace MOTION

namespace MOTION::proto::beavy {

class BooleanBEAVYWire;
using BooleanBEAVYWireVector = std::vector<std::shared_ptr<BooleanBEAVYWire>>;

class BEAVYProvider;

template <typename T>
class ArithmeticBEAVYTensorInputSender : public NewGate {
 public:
  ArithmeticBEAVYTensorInputSender(std::size_t gate_id, BEAVYProvider&,
                                 const tensor::TensorDimensions& dimensions,
                                 ENCRYPTO::ReusableFiberFuture<std::vector<T>>&&);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  std::shared_ptr<const ArithmeticBEAVYTensor<T>> get_output_tensor() const noexcept {
    return output_;
  }

 private:
  BEAVYProvider& beavy_provider_;
  const tensor::TensorDimensions dimensions_;
  std::size_t input_id_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> input_future_;
  ArithmeticBEAVYTensorP<T> output_;
  constexpr static std::size_t bit_size_ = ENCRYPTO::bit_size_v<T>;
};

template <typename T>
class ArithmeticBEAVYTensorInputReceiver : public NewGate {
 public:
  ArithmeticBEAVYTensorInputReceiver(std::size_t gate_id, BEAVYProvider&,
                                   const tensor::TensorDimensions& dimensions);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  std::shared_ptr<const ArithmeticBEAVYTensor<T>> get_output_tensor() const noexcept {
    return output_;
  }

 private:
  BEAVYProvider& beavy_provider_;
  const tensor::TensorDimensions dimensions_;
  std::size_t input_id_;
  ArithmeticBEAVYTensorP<T> output_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> public_share_future_;
  constexpr static std::size_t bit_size_ = ENCRYPTO::bit_size_v<T>;
};

template <typename T>
class ArithmeticBEAVYTensorOutput : public NewGate {
 public:
  ArithmeticBEAVYTensorOutput(std::size_t gate_id, BEAVYProvider&, ArithmeticBEAVYTensorCP<T>,
                            std::size_t output_owner);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> get_output_future();

 private:
  BEAVYProvider& beavy_provider_;
  ENCRYPTO::ReusableFiberPromise<std::vector<T>> output_promise_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> secret_share_future_;
  std::vector<T> secret_shares_;
  std::size_t output_owner_;
  const ArithmeticBEAVYTensorCP<T> input_;
};

template <typename T>
class ArithmeticBEAVYTensorFlatten : public NewGate {
 public:
  ArithmeticBEAVYTensorFlatten(std::size_t gate_id, BEAVYProvider&, std::size_t axis,
                               const ArithmeticBEAVYTensorCP<T> input);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  const ArithmeticBEAVYTensorP<T>& get_output_tensor() const { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  const ArithmeticBEAVYTensorCP<T> input_;
  std::shared_ptr<ArithmeticBEAVYTensor<T>> output_;
};

template <typename T>
class ArithmeticBEAVYTensorConv2D : public NewGate {
 public:
  ArithmeticBEAVYTensorConv2D(std::size_t gate_id, BEAVYProvider&, tensor::Conv2DOp,
                              const ArithmeticBEAVYTensorCP<T> input,
                              const ArithmeticBEAVYTensorCP<T> kernel,
                              const ArithmeticBEAVYTensorCP<T> bias, std::size_t fractional_bits);
  ~ArithmeticBEAVYTensorConv2D();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  const ArithmeticBEAVYTensorP<T>& get_output_tensor() const { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  tensor::Conv2DOp conv_op_;
  std::size_t fractional_bits_;
  const ArithmeticBEAVYTensorCP<T> input_;
  const ArithmeticBEAVYTensorCP<T> kernel_;
  const ArithmeticBEAVYTensorCP<T> bias_;
  std::shared_ptr<ArithmeticBEAVYTensor<T>> output_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
  std::vector<T> Delta_y_share_;
  std::unique_ptr<MOTION::ConvolutionInputSide<T>> conv_input_side_;
  std::unique_ptr<MOTION::ConvolutionKernelSide<T>> conv_kernel_side_;
};

template <typename T>
class ArithmeticBEAVYTensorGemm : public NewGate {
 public:
  ArithmeticBEAVYTensorGemm(std::size_t gate_id, BEAVYProvider&, tensor::GemmOp,
                            const ArithmeticBEAVYTensorCP<T> input_A,
                            const ArithmeticBEAVYTensorCP<T> input_B, std::size_t fractional_bits);
  ~ArithmeticBEAVYTensorGemm();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  const ArithmeticBEAVYTensorP<T>& get_output_tensor() const { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  tensor::GemmOp gemm_op_;
  std::size_t fractional_bits_;
  const ArithmeticBEAVYTensorCP<T> input_A_;
  const ArithmeticBEAVYTensorCP<T> input_B_;
  std::shared_ptr<ArithmeticBEAVYTensor<T>> output_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
  std::vector<T> Delta_y_share_;
  std::unique_ptr<MOTION::MatrixMultiplicationRHS<T>> mm_rhs_side_;
  std::unique_ptr<MOTION::MatrixMultiplicationLHS<T>> mm_lhs_side_;
};

template <typename T>
class ArithmeticBEAVYTensorMul : public NewGate {
 public:
  ArithmeticBEAVYTensorMul(std::size_t gate_id, BEAVYProvider&,
                           const ArithmeticBEAVYTensorCP<T> input_A,
                           const ArithmeticBEAVYTensorCP<T> input_B, std::size_t fractional_bits);
  ~ArithmeticBEAVYTensorMul();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  const ArithmeticBEAVYTensorP<T>& get_output_tensor() const { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  std::size_t fractional_bits_;
  const ArithmeticBEAVYTensorCP<T> input_A_;
  const ArithmeticBEAVYTensorCP<T> input_B_;
  std::shared_ptr<ArithmeticBEAVYTensor<T>> output_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
  std::vector<T> Delta_y_share_;
  std::unique_ptr<MOTION::IntegerMultiplicationSender<T>> mult_sender_;
  std::unique_ptr<MOTION::IntegerMultiplicationReceiver<T>> mult_receiver_;
};

template <typename T>
class ArithmeticBEAVYTensorAveragePool : public NewGate {
 public:
  ArithmeticBEAVYTensorAveragePool(std::size_t gate_id, BEAVYProvider&, tensor::AveragePoolOp,
                                   const ArithmeticBEAVYTensorCP<T> input,
                                   std::size_t fractional_bits);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  const ArithmeticBEAVYTensorP<T>& get_output_tensor() const { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  tensor::AveragePoolOp avgpool_op_;
  std::size_t data_size_;
  std::size_t fractional_bits_;
  const ArithmeticBEAVYTensorCP<T> input_;
  std::shared_ptr<ArithmeticBEAVYTensor<T>> output_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
  T factor_;
  std::vector<T> tmp_in_;
  std::vector<T> tmp_out_;
};

template <typename T>
class BooleanToArithmeticBEAVYTensorConversion : public NewGate {
 public:
  BooleanToArithmeticBEAVYTensorConversion(std::size_t gate_id, BEAVYProvider&,
                                           const BooleanBEAVYTensorCP input);
  ~BooleanToArithmeticBEAVYTensorConversion();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  ArithmeticBEAVYTensorCP<T> get_output_tensor() const noexcept { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  static constexpr auto bit_size_ = ENCRYPTO::bit_size_v<T>;
  const std::size_t data_size_;
  const BooleanBEAVYTensorCP input_;
  ArithmeticBEAVYTensorP<T> output_;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> t_share_future_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTSender<T>> ot_sender_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ACOTReceiver<T>> ot_receiver_;
  std::vector<T> arithmetized_secret_share_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
};

class BooleanBEAVYTensorRelu : public NewGate {
 public:
  BooleanBEAVYTensorRelu(std::size_t gate_id, BEAVYProvider&, const BooleanBEAVYTensorCP input);
  ~BooleanBEAVYTensorRelu();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  const BooleanBEAVYTensorP& get_output_tensor() const { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  const std::size_t bit_size_;
  const std::size_t data_size_;
  const BooleanBEAVYTensorCP input_;
  BooleanBEAVYTensorP output_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitSender> ot_sender_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitReceiver> ot_receiver_;
  ENCRYPTO::BitVector<> Delta_y_share_;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> share_future_;
};

template <typename T>
class BooleanXArithmeticBEAVYTensorRelu : public NewGate {
 public:
  BooleanXArithmeticBEAVYTensorRelu(std::size_t gate_id, BEAVYProvider&, const BooleanBEAVYTensorCP,
                                    const ArithmeticBEAVYTensorCP<T>);
  ~BooleanXArithmeticBEAVYTensorRelu();
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_online() override;
  const ArithmeticBEAVYTensorP<T>& get_output_tensor() const { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  static constexpr auto bit_size_ = ENCRYPTO::bit_size_v<T>;
  const std::size_t data_size_;
  const BooleanBEAVYTensorCP input_bool_;
  const ArithmeticBEAVYTensorCP<T> input_arith_;
  ArithmeticBEAVYTensorP<T> output_;
  std::unique_ptr<BitIntegerMultiplicationIntSide<T>> mult_int_side_;
  std::unique_ptr<BitIntegerMultiplicationBitSide<T>> mult_bit_side_;
  std::vector<T> delta_b_share_;
  std::vector<T> delta_b_x_delta_n_share_;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> share_future_;
};

class BooleanBEAVYTensorMaxPool : public NewGate {
 public:
  BooleanBEAVYTensorMaxPool(std::size_t gate_id, BEAVYProvider&, tensor::MaxPoolOp maxpool_op,
                            const BooleanBEAVYTensorCP input);
  bool need_setup() const noexcept override { return true; }
  bool need_online() const noexcept override { return true; }
  void evaluate_setup() override;
  void evaluate_setup_with_context(ExecutionContext&) override;
  void evaluate_online() override;
  void evaluate_online_with_context(ExecutionContext&) override;
  const BooleanBEAVYTensorP& get_output_tensor() const { return output_; }

 private:
  BEAVYProvider& beavy_provider_;
  const tensor::MaxPoolOp maxpool_op_;
  const std::size_t bit_size_;
  const std::size_t data_size_;
  const BooleanBEAVYTensorCP input_;
  const BooleanBEAVYTensorP output_;
  const ENCRYPTO::AlgorithmDescription& maxpool_algo_;
  BooleanBEAVYWireVector input_wires_;
  BooleanBEAVYWireVector output_wires_;
  std::vector<std::unique_ptr<NewGate>> gates_;
};

}  // namespace MOTION::proto::beavy
