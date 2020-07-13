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

#include "tools.h"

#include <stdexcept>
#include <unsupported/Eigen/CXX11/Tensor>

#include "tensor/tensor_op.h"

namespace MOTION::proto::yao {

void transpose_keys(ENCRYPTO::block128_vector& dst, const ENCRYPTO::block128_vector& src,
                    std::size_t bit_size, std::size_t num) {
  if (bit_size * num != src.size()) {
    throw std::invalid_argument("vector size mismatch");
  }
  using TensorTypeC = Eigen::Tensor<const ENCRYPTO::block128_t, 2, Eigen::RowMajor>;
  using TensorType = Eigen::Tensor<ENCRYPTO::block128_t, 2, Eigen::RowMajor>;
  dst.resize(bit_size * num);
  Eigen::TensorMap<TensorTypeC> tensor_src(src.data(), bit_size, num);
  Eigen::TensorMap<TensorType> tensor_dst(dst.data(), num, bit_size);
  tensor_dst = tensor_src.shuffle(Eigen::array<Eigen::Index, 2>{1, 0}).eval();
}

void maxpool_rearrange_keys_in(ENCRYPTO::block128_vector& dst, const ENCRYPTO::block128_vector& src,
                               std::size_t bit_size, const tensor::MaxPoolOp& maxpool_op) {
  using TensorType4C = Eigen::Tensor<const ENCRYPTO::block128_t, 4, Eigen::RowMajor>;
  using TensorType3 = Eigen::Tensor<ENCRYPTO::block128_t, 3, Eigen::RowMajor>;
  const auto in_channels = static_cast<Eigen::Index>(maxpool_op.input_shape_[0]);
  const auto in_rows = static_cast<Eigen::Index>(maxpool_op.input_shape_[1]);
  const auto in_columns = static_cast<Eigen::Index>(maxpool_op.input_shape_[2]);
  const auto kernel_rows = static_cast<Eigen::Index>(maxpool_op.kernel_shape_[0]);
  const auto kernel_columns = static_cast<Eigen::Index>(maxpool_op.kernel_shape_[1]);
  const auto stride_rows = static_cast<Eigen::Index>(maxpool_op.strides_[0]);
  const auto stride_columns = static_cast<Eigen::Index>(maxpool_op.strides_[1]);
  const auto num_patches =
      static_cast<Eigen::Index>(maxpool_op.output_shape_[1] * maxpool_op.output_shape_[2]);
  const auto num_bits = static_cast<Eigen::Index>(bit_size);

  dst.resize(kernel_rows * kernel_columns * num_bits * in_channels * num_patches);

  Eigen::TensorMap<TensorType4C> tensor_src(src.data(), num_bits, in_channels, in_rows, in_columns);
  Eigen::TensorMap<TensorType3> tensor_dst(dst.data(), kernel_rows * kernel_columns, num_bits,
                                           num_patches * in_channels);

  tensor_dst = tensor_src.shuffle(Eigen::array<Eigen::Index, 4>{0, 3, 2, 1})
                   .extract_image_patches(kernel_rows, kernel_columns, stride_rows, stride_columns,
                                          1, 1, 1, 1, 0, 0, 0, 0, ENCRYPTO::block128_t::make_zero())
                   .reshape(Eigen::array<Eigen::Index, 4>{
                       num_bits, num_patches, kernel_rows * kernel_columns, in_channels})
                   .shuffle(Eigen::array<Eigen::Index, 4>{2, 0, 1, 3})
                   .reshape(Eigen::array<Eigen::Index, 3>{kernel_rows * kernel_columns, num_bits,
                                                          num_patches * in_channels});
}

void maxpool_rearrange_keys_out(ENCRYPTO::block128_vector& dst,
                                const ENCRYPTO::block128_vector& src, std::size_t bit_size,
                                const tensor::MaxPoolOp& maxpool_op) {
  using TensorType2C = Eigen::Tensor<const ENCRYPTO::block128_t, 2, Eigen::RowMajor>;
  using TensorType4 = Eigen::Tensor<ENCRYPTO::block128_t, 4, Eigen::RowMajor>;

  const auto out_channels = static_cast<Eigen::Index>(maxpool_op.output_shape_[0]);
  const auto out_rows = static_cast<Eigen::Index>(maxpool_op.output_shape_[1]);
  const auto out_columns = static_cast<Eigen::Index>(maxpool_op.output_shape_[2]);
  const auto num_patches = out_rows * out_columns;
  const auto num_bits = static_cast<Eigen::Index>(bit_size);

  dst.resize(num_bits * out_channels * out_rows * out_columns);

  Eigen::TensorMap<TensorType2C> tensor_src(src.data(), num_bits, num_patches * out_channels);
  Eigen::TensorMap<TensorType4> tensor_dst(dst.data(), num_bits, out_channels, out_rows,
                                           out_columns);
  tensor_dst =
      tensor_src
          .reshape(Eigen::array<Eigen::Index, 4>{num_bits, out_columns, out_rows, out_channels})
          .shuffle(Eigen::array<Eigen::Index, 4>{0, 3, 2, 1});
}

}  // namespace MOTION::proto::yao
