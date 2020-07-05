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

#include "linear_algebra.h"

#include <cassert>

#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

#include "tensor/tensor_op.h"

namespace MOTION {

template <typename T>
void matrix_multiply(std::size_t dim_l, std::size_t dim_m, std::size_t dim_n, const T* A,
                     const T* B, T* output) {
  using MatrixType = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<MatrixType> matrix_output(output, dim_l, dim_n);
  Eigen::Map<const MatrixType> matrix_A(A, dim_l, dim_m);
  Eigen::Map<const MatrixType> matrix_B(B, dim_m, dim_n);
  matrix_output = matrix_A * matrix_B;
}

template <typename T>
std::vector<T> matrix_multiply(std::size_t dim_l, std::size_t dim_m, std::size_t dim_n,
                               const std::vector<T>& A, const std::vector<T>& B) {
  assert(A.size() == dim_l * dim_m);
  assert(B.size() == dim_m * dim_n);
  std::vector<T> output(dim_l * dim_n);
  matrix_multiply(dim_l, dim_m, dim_n, A.data(), B.data(), output.data());
  return output;
}

template std::vector<std::uint8_t> matrix_multiply(std::size_t, std::size_t, std::size_t,
                                                   const std::vector<std::uint8_t>&,
                                                   const std::vector<std::uint8_t>&);
template std::vector<std::uint16_t> matrix_multiply(std::size_t, std::size_t, std::size_t,
                                                    const std::vector<std::uint16_t>&,
                                                    const std::vector<std::uint16_t>&);
template std::vector<std::uint32_t> matrix_multiply(std::size_t, std::size_t, std::size_t,
                                                    const std::vector<std::uint32_t>&,
                                                    const std::vector<std::uint32_t>&);
template std::vector<std::uint64_t> matrix_multiply(std::size_t, std::size_t, std::size_t,
                                                    const std::vector<std::uint64_t>&,
                                                    const std::vector<std::uint64_t>&);
template std::vector<__uint128_t> matrix_multiply(std::size_t, std::size_t, std::size_t,
                                                  const std::vector<__uint128_t>&,
                                                  const std::vector<__uint128_t>&);

template <typename T>
void convolution(const tensor::Conv2DOp& conv_op, const T* input_buffer, const T* kernel_buffer,
                 T* output_buffer) {
  using TensorType3 = Eigen::Tensor<T, 3, Eigen::RowMajor>;
  using CTensorType3 = Eigen::Tensor<const T, 3, Eigen::RowMajor>;
  using CTensorType4 = Eigen::Tensor<const T, 4, Eigen::RowMajor>;
  assert(conv_op.verify());
  const auto& output_shape = conv_op.output_shape_;
  const auto& input_shape = conv_op.input_shape_;
  const auto& kernel_shape = conv_op.kernel_shape_;

  Eigen::TensorMap<CTensorType3> input(input_buffer, input_shape[0], input_shape[1],
                                       input_shape[2]);
  Eigen::TensorMap<CTensorType4> kernel(kernel_buffer, kernel_shape[0], kernel_shape[1],
                                        kernel_shape[2], kernel_shape[3]);
  Eigen::TensorMap<TensorType3> output(output_buffer, output_shape[0], output_shape[1],
                                       output_shape[2]);
  const std::array<Eigen::Index, 2> kernel_matrix_dimensions = {
      static_cast<Eigen::Index>(kernel_shape[1] * kernel_shape[2] * kernel_shape[3]),
      static_cast<Eigen::Index>(kernel_shape[0])};
  const std::array<Eigen::Index, 2> input_matrix_dimensions = {
      static_cast<Eigen::Index>(output_shape[1] * output_shape[2]),
      static_cast<Eigen::Index>(kernel_shape[1] * kernel_shape[2] * kernel_shape[3])};

  auto kernel_matrix =
      kernel.shuffle(std::array<int, 4>{3, 2, 1, 0}).reshape(kernel_matrix_dimensions);

  auto input_matrix =
      input.shuffle(Eigen::array<Eigen::Index, 3>{2, 1, 0})
          .extract_image_patches(kernel_shape[2], kernel_shape[3], conv_op.strides_[0],
                                 conv_op.strides_[1], conv_op.dilations_[0], conv_op.dilations_[1],
                                 1, 1, conv_op.pads_[0], conv_op.pads_[2], conv_op.pads_[1],
                                 conv_op.pads_[3], 0)
          .reshape(input_matrix_dimensions);

  const std::array<Eigen::IndexPair<Eigen::Index>, 1> contraction_dimensions = {
      Eigen::IndexPair<Eigen::Index>(1, 0)};
  auto output_matrix =
      kernel_matrix.shuffle(std::array<Eigen::Index, 2>{1, 0})
          .contract(input_matrix.shuffle(std::array<Eigen::Index, 2>{1, 0}), contraction_dimensions)
          .shuffle(std::array<Eigen::Index, 2>{1, 0});

  const std::array<Eigen::Index, 3> rev_output_dimensions = {
      output.dimension(2), output.dimension(1), output.dimension(0)};
  output =
      output_matrix.reshape(rev_output_dimensions).shuffle(Eigen::array<Eigen::Index, 3>{2, 1, 0});
}

template <typename T>
std::vector<T> convolution(const tensor::Conv2DOp& conv_op, const std::vector<T>& input_buffer,
                           const std::vector<T>& kernel_buffer) {
  assert(conv_op.verify());
  assert(input_buffer.size() == conv_op.compute_input_size());
  assert(kernel_buffer.size() == conv_op.compute_kernel_size());
  std::vector<T> output_buffer(conv_op.compute_output_size());
  convolution(conv_op, input_buffer.data(), kernel_buffer.data(), output_buffer.data());
  return output_buffer;
}

template std::vector<std::uint8_t> convolution(const tensor::Conv2DOp&,
                                               const std::vector<std::uint8_t>&,
                                               const std::vector<std::uint8_t>&);
template std::vector<std::uint16_t> convolution(const tensor::Conv2DOp&,
                                                const std::vector<std::uint16_t>&,
                                                const std::vector<std::uint16_t>&);
template std::vector<std::uint32_t> convolution(const tensor::Conv2DOp&,
                                                const std::vector<std::uint32_t>&,
                                                const std::vector<std::uint32_t>&);
template std::vector<std::uint64_t> convolution(const tensor::Conv2DOp&,
                                                const std::vector<std::uint64_t>&,
                                                const std::vector<std::uint64_t>&);
template std::vector<__uint128_t> convolution(const tensor::Conv2DOp&,
                                              const std::vector<__uint128_t>&,
                                              const std::vector<__uint128_t>&);

}  // namespace MOTION
