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

#include "tensor_op.h"

#include <cassert>

namespace MOTION::tensor {

bool Conv2DOp::verify() const {
  bool result = true;
  result = result && (output_shape_ == compute_output_shape());
  // maybe add more checks here
  return true;
}

std::array<std::size_t, 3> Conv2DOp::compute_output_shape() const {
  const auto compute_output_dimension = [](auto input_size, auto kernel_size, auto padding_begin,
                                           auto padding_end, auto stride) {
    return (input_size - kernel_size + padding_begin + padding_end + stride) / stride;
  };

  std::array<std::size_t, 3> output_shape;
  output_shape[0] = kernel_shape_[0];
  output_shape[1] =
      compute_output_dimension(input_shape_[1], kernel_shape_[2], pads_[0], pads_[2], strides_[0]);
  output_shape[2] =
      compute_output_dimension(input_shape_[2], kernel_shape_[3], pads_[1], pads_[3], strides_[1]);
  return output_shape;
}

std::size_t Conv2DOp::compute_output_size() const {
  assert(verify());
  auto output_shape = compute_output_shape();
  return output_shape[0] * output_shape[1] * output_shape[2];
}

std::size_t Conv2DOp::compute_input_size() const {
  assert(verify());
  return input_shape_[0] * input_shape_[1] * input_shape_[2];
}

std::size_t Conv2DOp::compute_kernel_size() const {
  assert(verify());
  return kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
}

std::pair<std::size_t, std::size_t> Conv2DOp::compute_input_matrix_shape() const {
  assert(verify());
  std::size_t num_rows = kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
  std::size_t num_columns = output_shape_[1] * output_shape_[2];
  return {num_rows, num_columns};
}

std::pair<std::size_t, std::size_t> Conv2DOp::compute_kernel_matrix_shape() const {
  assert(verify());
  std::size_t num_rows = kernel_shape_[0];
  std::size_t num_columns = kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
  return {num_rows, num_columns};
}

std::pair<std::size_t, std::size_t> Conv2DOp::compute_output_matrix_shape() const {
  assert(verify());
  std::size_t num_rows = kernel_shape_[0];
  std::size_t num_columns = output_shape_[1] * output_shape_[2];
  return {num_rows, num_columns};
}

}  // namespace MOTION::tensor
