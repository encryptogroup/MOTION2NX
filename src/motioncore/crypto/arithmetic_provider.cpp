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

#include <unsupported/Eigen/CXX11/Tensor>

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

// ---------- MatrixMultiplicationRHS ----------

template <typename T>
MatrixMultiplicationRHS<T>::MatrixMultiplicationRHS(std::size_t l, std::size_t m, std::size_t n,
                                                    ArithmeticProvider& arith_provider)
    : dims_({l, m, n}),
      mult_sender_(arith_provider.register_integer_multiplication_send<T>(l * m, n)),
      is_output_ready_(false) {}

template <typename T>
MatrixMultiplicationRHS<T>::~MatrixMultiplicationRHS() = default;

template <typename T>
void MatrixMultiplicationRHS<T>::set_input(std::vector<T>&& inputs) {
  set_input(inputs);
}

template <typename T>
void MatrixMultiplicationRHS<T>::set_input(const std::vector<T>& inputs) {
  if (inputs.size() != dims_[1] * dims_[2]) {
    throw std::invalid_argument("input has unexpected size");
  }
  set_input(inputs.data());
}

template <typename T>
void MatrixMultiplicationRHS<T>::set_input(const T* inputs) {
  const auto input_size = dims_[1] * dims_[2];
  std::vector<T> mult_inputs(dims_[0] * input_size);
  for (std::size_t i = 0; i < dims_[0]; ++i) {
    std::copy_n(inputs, input_size, std::begin(mult_inputs) + i * input_size);
  }
  mult_sender_->set_inputs(std::move(mult_inputs));
}

template <typename T>
void MatrixMultiplicationRHS<T>::compute_output() {
  mult_sender_->compute_outputs();
  auto mult_output = mult_sender_->get_outputs();
  output_.resize(dims_[0] * dims_[2]);
  using TensorType2 = Eigen::Tensor<T, 2, Eigen::RowMajor>;
  using TensorType3 = Eigen::Tensor<T, 3, Eigen::RowMajor>;
  Eigen::TensorMap<TensorType3> input_tensor(mult_output.data(), dims_[0], dims_[1], dims_[2]);
  Eigen::TensorMap<TensorType2> output_tensor(output_.data(), dims_[0], dims_[2]);
  Eigen::array<Eigen::Index, 1> reduction_dimensions = {1};
  output_tensor = input_tensor.sum(reduction_dimensions);
  is_output_ready_ = true;
}

template <typename T>
std::vector<T> MatrixMultiplicationRHS<T>::get_output() {
  assert(is_output_ready_);
  return std::move(output_);
}

// ---------- MatrixMultiplicationLHS ----------

template <typename T>
MatrixMultiplicationLHS<T>::MatrixMultiplicationLHS(std::size_t l, std::size_t m, std::size_t n,
                                                    ArithmeticProvider& arith_provider)
    : dims_({l, m, n}),
      mult_receiver_(arith_provider.register_integer_multiplication_receive<T>(l * m, n)),
      is_output_ready_(false) {}

template <typename T>
MatrixMultiplicationLHS<T>::~MatrixMultiplicationLHS() = default;

template <typename T>
void MatrixMultiplicationLHS<T>::set_input(std::vector<T>&& inputs) {
  if (inputs.size() != dims_[0] * dims_[1]) {
    throw std::invalid_argument("input has unexpected size");
  }
  mult_receiver_->set_inputs(std::move(inputs));
}

template <typename T>
void MatrixMultiplicationLHS<T>::set_input(const std::vector<T>& inputs) {
  if (inputs.size() != dims_[0] * dims_[1]) {
    throw std::invalid_argument("input has unexpected size");
  }
  set_input(inputs.data());
}

template <typename T>
void MatrixMultiplicationLHS<T>::set_input(const T* inputs) {
  std::vector<T> mult_inputs(inputs, inputs + dims_[0] * dims_[1]);
  mult_receiver_->set_inputs(std::move(mult_inputs));
}

template <typename T>
void MatrixMultiplicationLHS<T>::compute_output() {
  mult_receiver_->compute_outputs();
  auto mult_output = mult_receiver_->get_outputs();
  assert(mult_output.size() == dims_[0] * dims_[1] * dims_[2]);
  output_.resize(dims_[0] * dims_[2]);
  using TensorType2 = Eigen::Tensor<T, 2, Eigen::RowMajor>;
  using TensorType3 = Eigen::Tensor<T, 3, Eigen::RowMajor>;
  Eigen::TensorMap<TensorType3> input_tensor(mult_output.data(), dims_[0], dims_[1], dims_[2]);
  Eigen::TensorMap<TensorType2> output_tensor(output_.data(), dims_[0], dims_[2]);
  Eigen::array<Eigen::Index, 1> reduction_dimensions = {1};
  output_tensor = input_tensor.sum(reduction_dimensions);
  is_output_ready_ = true;
}

template <typename T>
std::vector<T> MatrixMultiplicationLHS<T>::get_output() {
  assert(is_output_ready_);
  return std::move(output_);
}

// ---------- ConvolutionInputSide ----------

template <typename T>
ConvolutionInputSide<T>::ConvolutionInputSide(tensor::Conv2DOp conv_op,
                                              ArithmeticProvider& arith_provider)
    : conv_op_(conv_op), is_output_ready_(false) {
  const auto kernel_matrix_shape = conv_op_.compute_kernel_matrix_shape();
  const auto input_matrix_shape = conv_op_.compute_input_matrix_shape();
  matrix_rhs_ = arith_provider.register_matrix_multiplication_rhs<T>(
      kernel_matrix_shape.first, kernel_matrix_shape.second, input_matrix_shape.second);
}

template <typename T>
ConvolutionInputSide<T>::~ConvolutionInputSide() = default;

template <typename T>
void ConvolutionInputSide<T>::set_input(std::vector<T>&& input) {
  set_input(input);
}

template <typename T>
void ConvolutionInputSide<T>::set_input(const std::vector<T>& input) {
  if (input.size() != conv_op_.compute_input_size()) {
    throw std::invalid_argument("input has unexpected size");
  }
  set_input(input.data());
}

template <typename T>
void ConvolutionInputSide<T>::set_input(const T* input_buffer) {
  const auto matrix_shape = conv_op_.compute_input_matrix_shape();
  std::vector<T> input_matrix_buffer(matrix_shape.first * matrix_shape.second);
  using TensorType2 = Eigen::Tensor<T, 2, Eigen::RowMajor>;
  using CTensorType3 = Eigen::Tensor<const T, 3, Eigen::RowMajor>;
  Eigen::TensorMap<CTensorType3> input(input_buffer, conv_op_.input_shape_[0],
                                       conv_op_.input_shape_[1], conv_op_.input_shape_[2]);
  Eigen::TensorMap<TensorType2> input_matrix(input_matrix_buffer.data(), matrix_shape.first,
                                             matrix_shape.second);
  input_matrix =
      input.shuffle(Eigen::array<Eigen::Index, 3>{2, 1, 0})
          .extract_image_patches(conv_op_.kernel_shape_[2], conv_op_.kernel_shape_[3],
                                 conv_op_.strides_[0], conv_op_.strides_[1], conv_op_.dilations_[0],
                                 conv_op_.dilations_[1], 1, 1, conv_op_.pads_[0], conv_op_.pads_[2],
                                 conv_op_.pads_[1], conv_op_.pads_[3], 0)
          .reshape(Eigen::array<Eigen::Index, 2>{static_cast<Eigen::Index>(matrix_shape.second),
                                                 static_cast<Eigen::Index>(matrix_shape.first)})
          .shuffle(Eigen::array<Eigen::Index, 2>{1, 0});
  matrix_rhs_->set_input(std::move(input_matrix_buffer));
}

template <typename T>
void ConvolutionInputSide<T>::compute_output() {
  matrix_rhs_->compute_output();
  output_ = matrix_rhs_->get_output();
  assert(output_.size() == conv_op_.compute_output_size());
  using CTensorType2 = Eigen::Tensor<T, 2, Eigen::RowMajor>;
  using TensorType3 = Eigen::Tensor<T, 3, Eigen::RowMajor>;
  const auto matrix_shape = conv_op_.compute_output_matrix_shape();
  // be careful about aliasing, use eval()
  Eigen::TensorMap<CTensorType2> output_matrix(output_.data(), matrix_shape.first,
                                               matrix_shape.second);
  Eigen::TensorMap<TensorType3> output(output_.data(), conv_op_.output_shape_[0],
                                       conv_op_.output_shape_[1], conv_op_.output_shape_[2]);
  const std::array<Eigen::Index, 3> rev_output_dimensions = {
      static_cast<Eigen::Index>(conv_op_.output_shape_[2]),
      static_cast<Eigen::Index>(conv_op_.output_shape_[1]),
      static_cast<Eigen::Index>(conv_op_.output_shape_[0])};
  output = output_matrix.shuffle(std::array<Eigen::Index, 2>{1, 0})
               .reshape(rev_output_dimensions)
               .shuffle(Eigen::array<Eigen::Index, 3>{2, 1, 0})
               .eval();
  is_output_ready_ = true;
}

template <typename T>
std::vector<T> ConvolutionInputSide<T>::get_output() {
  assert(is_output_ready_);
  return std::move(output_);
}

// ---------- ConvolutionKernelSide ----------

template <typename T>
ConvolutionKernelSide<T>::ConvolutionKernelSide(tensor::Conv2DOp conv_op,
                                                ArithmeticProvider& arith_provider)
    : conv_op_(conv_op), is_output_ready_(false) {
  const auto kernel_matrix_shape = conv_op_.compute_kernel_matrix_shape();
  const auto input_matrix_shape = conv_op_.compute_input_matrix_shape();
  matrix_lhs_ = arith_provider.register_matrix_multiplication_lhs<T>(
      kernel_matrix_shape.first, kernel_matrix_shape.second, input_matrix_shape.second);
}

template <typename T>
ConvolutionKernelSide<T>::~ConvolutionKernelSide() = default;

template <typename T>
void ConvolutionKernelSide<T>::set_input(std::vector<T>&& kernel_buffer) {
  set_input(kernel_buffer);
}

template <typename T>
void ConvolutionKernelSide<T>::set_input(const std::vector<T>& kernel) {
  if (kernel.size() != conv_op_.compute_kernel_size()) {
    throw std::invalid_argument("kernel has unexpected size");
  }
  set_input(kernel.data());
}

template <typename T>
void ConvolutionKernelSide<T>::set_input(const T* kernel_buffer) {
  using TensorType2 = Eigen::Tensor<T, 2, Eigen::RowMajor>;
  using CTensorType4 = Eigen::Tensor<const T, 4, Eigen::RowMajor>;
  Eigen::TensorMap<CTensorType4> kernel(kernel_buffer, conv_op_.kernel_shape_[0],
                                        conv_op_.kernel_shape_[1], conv_op_.kernel_shape_[2],
                                        conv_op_.kernel_shape_[3]);
  const auto matrix_shape = conv_op_.compute_kernel_matrix_shape();
  std::vector<T> kernel_matrix_buffer(matrix_shape.first * matrix_shape.second);
  Eigen::TensorMap<TensorType2> kernel_matrix(kernel_matrix_buffer.data(), matrix_shape.first,
                                              matrix_shape.second);
  kernel_matrix =
      kernel.shuffle(std::array<Eigen::Index, 4>{3, 2, 1, 0})
          .reshape(std::array<Eigen::Index, 2>{static_cast<Eigen::Index>(matrix_shape.second),
                                               static_cast<Eigen::Index>(matrix_shape.first)})
          .shuffle(std::array<Eigen::Index, 2>{1, 0});
  matrix_lhs_->set_input(std::move(kernel_matrix_buffer));
}

template <typename T>
void ConvolutionKernelSide<T>::compute_output() {
  matrix_lhs_->compute_output();
  output_ = matrix_lhs_->get_output();
  assert(output_.size() == conv_op_.compute_output_size());
  using CTensorType2 = Eigen::Tensor<T, 2, Eigen::RowMajor>;
  using TensorType3 = Eigen::Tensor<T, 3, Eigen::RowMajor>;
  const auto matrix_shape = conv_op_.compute_output_matrix_shape();
  // be careful about aliasing, use eval()
  Eigen::TensorMap<CTensorType2> output_matrix(output_.data(), matrix_shape.first,
                                               matrix_shape.second);
  Eigen::TensorMap<TensorType3> output(output_.data(), conv_op_.output_shape_[0],
                                       conv_op_.output_shape_[1], conv_op_.output_shape_[2]);
  const std::array<Eigen::Index, 3> rev_output_dimensions = {
      static_cast<Eigen::Index>(conv_op_.output_shape_[2]),
      static_cast<Eigen::Index>(conv_op_.output_shape_[1]),
      static_cast<Eigen::Index>(conv_op_.output_shape_[0])};
  output = output_matrix.shuffle(std::array<Eigen::Index, 2>{1, 0})
               .reshape(rev_output_dimensions)
               .shuffle(Eigen::array<Eigen::Index, 3>{2, 1, 0})
               .eval();
  is_output_ready_ = true;
}

template <typename T>
std::vector<T> ConvolutionKernelSide<T>::get_output() {
  assert(is_output_ready_);
  return std::move(output_);
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

template <typename T>
std::unique_ptr<MatrixMultiplicationRHS<T>> ArithmeticProvider::register_matrix_multiplication_rhs(
    std::size_t dim_l, std::size_t dim_m, std::size_t dim_n) {
  return std::make_unique<MatrixMultiplicationRHS<T>>(dim_l, dim_m, dim_n, *this);
}

template <typename T>
std::unique_ptr<MatrixMultiplicationLHS<T>> ArithmeticProvider::register_matrix_multiplication_lhs(
    std::size_t dim_l, std::size_t dim_m, std::size_t dim_n) {
  return std::make_unique<MatrixMultiplicationLHS<T>>(dim_l, dim_m, dim_n, *this);
}

template <typename T>
std::unique_ptr<ConvolutionInputSide<T>> ArithmeticProvider::register_convolution_input_side(
    tensor::Conv2DOp conv_op) {
  return std::make_unique<ConvolutionInputSide<T>>(conv_op, *this);
}

template <typename T>
std::unique_ptr<ConvolutionKernelSide<T>> ArithmeticProvider::register_convolution_kernel_side(
    tensor::Conv2DOp conv_op) {
  return std::make_unique<ConvolutionKernelSide<T>>(conv_op, *this);
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

template class MatrixMultiplicationRHS<std::uint8_t>;
template class MatrixMultiplicationRHS<std::uint16_t>;
template class MatrixMultiplicationRHS<std::uint32_t>;
template class MatrixMultiplicationRHS<std::uint64_t>;
template class MatrixMultiplicationRHS<__uint128_t>;

template class MatrixMultiplicationLHS<std::uint8_t>;
template class MatrixMultiplicationLHS<std::uint16_t>;
template class MatrixMultiplicationLHS<std::uint32_t>;
template class MatrixMultiplicationLHS<std::uint64_t>;
template class MatrixMultiplicationLHS<__uint128_t>;

template class ConvolutionInputSide<std::uint8_t>;
template class ConvolutionInputSide<std::uint16_t>;
template class ConvolutionInputSide<std::uint32_t>;
template class ConvolutionInputSide<std::uint64_t>;
template class ConvolutionInputSide<__uint128_t>;

template class ConvolutionKernelSide<std::uint8_t>;
template class ConvolutionKernelSide<std::uint16_t>;
template class ConvolutionKernelSide<std::uint32_t>;
template class ConvolutionKernelSide<std::uint64_t>;
template class ConvolutionKernelSide<__uint128_t>;

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

template std::unique_ptr<MatrixMultiplicationRHS<std::uint8_t>>
    ArithmeticProvider::register_matrix_multiplication_rhs<std::uint8_t>(std::size_t, std::size_t,
                                                                         std::size_t);
template std::unique_ptr<MatrixMultiplicationRHS<std::uint16_t>>
    ArithmeticProvider::register_matrix_multiplication_rhs<std::uint16_t>(std::size_t, std::size_t,
                                                                          std::size_t);
template std::unique_ptr<MatrixMultiplicationRHS<std::uint32_t>>
    ArithmeticProvider::register_matrix_multiplication_rhs<std::uint32_t>(std::size_t, std::size_t,
                                                                          std::size_t);
template std::unique_ptr<MatrixMultiplicationRHS<std::uint64_t>>
    ArithmeticProvider::register_matrix_multiplication_rhs<std::uint64_t>(std::size_t, std::size_t,
                                                                          std::size_t);
template std::unique_ptr<MatrixMultiplicationRHS<__uint128_t>>
    ArithmeticProvider::register_matrix_multiplication_rhs<__uint128_t>(std::size_t, std::size_t,
                                                                        std::size_t);

template std::unique_ptr<MatrixMultiplicationLHS<std::uint8_t>>
    ArithmeticProvider::register_matrix_multiplication_lhs<std::uint8_t>(std::size_t, std::size_t,
                                                                         std::size_t);
template std::unique_ptr<MatrixMultiplicationLHS<std::uint16_t>>
    ArithmeticProvider::register_matrix_multiplication_lhs<std::uint16_t>(std::size_t, std::size_t,
                                                                          std::size_t);
template std::unique_ptr<MatrixMultiplicationLHS<std::uint32_t>>
    ArithmeticProvider::register_matrix_multiplication_lhs<std::uint32_t>(std::size_t, std::size_t,
                                                                          std::size_t);
template std::unique_ptr<MatrixMultiplicationLHS<std::uint64_t>>
    ArithmeticProvider::register_matrix_multiplication_lhs<std::uint64_t>(std::size_t, std::size_t,
                                                                          std::size_t);
template std::unique_ptr<MatrixMultiplicationLHS<__uint128_t>>
    ArithmeticProvider::register_matrix_multiplication_lhs<__uint128_t>(std::size_t, std::size_t,
                                                                        std::size_t);

template std::unique_ptr<ConvolutionInputSide<std::uint8_t>>
    ArithmeticProvider::register_convolution_input_side<std::uint8_t>(tensor::Conv2DOp);
template std::unique_ptr<ConvolutionInputSide<std::uint16_t>>
    ArithmeticProvider::register_convolution_input_side<std::uint16_t>(tensor::Conv2DOp);
template std::unique_ptr<ConvolutionInputSide<std::uint32_t>>
    ArithmeticProvider::register_convolution_input_side<std::uint32_t>(tensor::Conv2DOp);
template std::unique_ptr<ConvolutionInputSide<std::uint64_t>>
    ArithmeticProvider::register_convolution_input_side<std::uint64_t>(tensor::Conv2DOp);
template std::unique_ptr<ConvolutionInputSide<__uint128_t>>
    ArithmeticProvider::register_convolution_input_side<__uint128_t>(tensor::Conv2DOp);

template std::unique_ptr<ConvolutionKernelSide<std::uint8_t>>
    ArithmeticProvider::register_convolution_kernel_side<std::uint8_t>(tensor::Conv2DOp);
template std::unique_ptr<ConvolutionKernelSide<std::uint16_t>>
    ArithmeticProvider::register_convolution_kernel_side<std::uint16_t>(tensor::Conv2DOp);
template std::unique_ptr<ConvolutionKernelSide<std::uint32_t>>
    ArithmeticProvider::register_convolution_kernel_side<std::uint32_t>(tensor::Conv2DOp);
template std::unique_ptr<ConvolutionKernelSide<std::uint64_t>>
    ArithmeticProvider::register_convolution_kernel_side<std::uint64_t>(tensor::Conv2DOp);
template std::unique_ptr<ConvolutionKernelSide<__uint128_t>>
    ArithmeticProvider::register_convolution_kernel_side<__uint128_t>(tensor::Conv2DOp);

}  // namespace MOTION
