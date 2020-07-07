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

#include <boost/functional/hash.hpp>
#include <cassert>

namespace MOTION::tensor {

bool Conv2DOp::verify() const noexcept {
  bool result = true;
  result = result && (output_shape_ == compute_output_shape());
  result = result && strides_[0] > 0 && strides_[1] > 0;
  // maybe add more checks here
  return true;
}

std::array<std::size_t, 3> Conv2DOp::compute_output_shape() const noexcept {
  const auto compute_output_dimension = [](auto input_size, auto kernel_size, auto padding_begin,
                                           auto padding_end, auto stride) {
    assert(stride != 0);
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

std::size_t Conv2DOp::compute_output_size() const noexcept {
  assert(verify());
  auto output_shape = compute_output_shape();
  return output_shape[0] * output_shape[1] * output_shape[2];
}

std::size_t Conv2DOp::compute_input_size() const noexcept {
  assert(verify());
  return input_shape_[0] * input_shape_[1] * input_shape_[2];
}

std::size_t Conv2DOp::compute_kernel_size() const noexcept {
  assert(verify());
  return kernel_shape_[0] * kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
}

std::pair<std::size_t, std::size_t> Conv2DOp::compute_input_matrix_shape() const noexcept {
  assert(verify());
  std::size_t num_rows = kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
  std::size_t num_columns = output_shape_[1] * output_shape_[2];
  return {num_rows, num_columns};
}

std::pair<std::size_t, std::size_t> Conv2DOp::compute_kernel_matrix_shape() const noexcept {
  assert(verify());
  std::size_t num_rows = kernel_shape_[0];
  std::size_t num_columns = kernel_shape_[1] * kernel_shape_[2] * kernel_shape_[3];
  return {num_rows, num_columns};
}

std::pair<std::size_t, std::size_t> Conv2DOp::compute_output_matrix_shape() const noexcept {
  assert(verify());
  std::size_t num_rows = kernel_shape_[0];
  std::size_t num_columns = output_shape_[1] * output_shape_[2];
  return {num_rows, num_columns};
}

TensorDimensions Conv2DOp::get_input_tensor_dims() const noexcept {
  assert(verify());
  return {.batch_size_ = 1,
          .num_channels_ = input_shape_[0],
          .height_ = input_shape_[1],
          .width_ = input_shape_[2]};
}

TensorDimensions Conv2DOp::get_kernel_tensor_dims() const noexcept {
  assert(verify());
  return {.batch_size_ = kernel_shape_[0],
          .num_channels_ = kernel_shape_[1],
          .height_ = kernel_shape_[2],
          .width_ = kernel_shape_[3]};
}

TensorDimensions Conv2DOp::get_output_tensor_dims() const noexcept {
  assert(verify());
  return {.batch_size_ = 1,
          .num_channels_ = output_shape_[0],
          .height_ = output_shape_[1],
          .width_ = output_shape_[2]};
}

bool Conv2DOp::operator==(const Conv2DOp& other) const noexcept {
  assert(verify());
  assert(other.verify());
  bool result = true;
  result = result && kernel_shape_ == other.kernel_shape_;
  result = result && input_shape_ == other.input_shape_;
  result = result && output_shape_ == other.output_shape_;
  result = result && dilations_ == other.dilations_;
  result = result && pads_ == other.pads_;
  result = result && strides_ == other.strides_;
  return result;
}

bool GemmOp::verify() const noexcept {
  bool result = true;
  result = result && (input_A_shape_[1] == input_B_shape_[0]);
  result = result && (input_A_shape_[0] == output_shape_[0]);
  result = result && (input_B_shape_[1] == output_shape_[1]);
  // maybe add more checks here
  return true;
}

std::array<std::size_t, 2> GemmOp::compute_output_shape() const noexcept {
  return {input_A_shape_[0], input_B_shape_[1]};
}

std::size_t GemmOp::compute_output_size() const noexcept {
  assert(verify());
  auto output_shape = compute_output_shape();
  return output_shape[0] * output_shape[1];
}

std::size_t GemmOp::compute_input_A_size() const noexcept {
  assert(verify());
  return input_A_shape_[0] * input_A_shape_[1];
}

std::size_t GemmOp::compute_input_B_size() const noexcept {
  assert(verify());
  return input_B_shape_[0] * input_B_shape_[1];
}

TensorDimensions GemmOp::get_input_A_tensor_dims() const noexcept {
  assert(verify());
  return {.batch_size_ = 1,
          .num_channels_ = 1,
          .height_ = input_A_shape_[0],
          .width_ = input_A_shape_[1]};
}

TensorDimensions GemmOp::get_input_B_tensor_dims() const noexcept {
  assert(verify());
  return {.batch_size_ = 1,
          .num_channels_ = 1,
          .height_ = input_B_shape_[0],
          .width_ = input_B_shape_[1]};
}

TensorDimensions GemmOp::get_output_tensor_dims() const noexcept {
  assert(verify());
  return {.batch_size_ = 1,
          .num_channels_ = 1,
          .height_ = output_shape_[0],
          .width_ = output_shape_[1]};
}

bool GemmOp::operator==(const GemmOp& other) const noexcept {
  assert(verify());
  assert(other.verify());
  bool result = true;
  result = result && input_A_shape_ == other.input_A_shape_;
  result = result && input_B_shape_ == other.input_B_shape_;
  result = result && output_shape_ == other.output_shape_;
  return result;
}

TensorDimensions flatten(const TensorDimensions& dims, std::size_t axis) {
  std::size_t height = 1;
  std::size_t width = 1;
  switch (axis) {
    case 0:
      width *= dims.batch_size_;
      [[fallthrough]];
    case 1:
      width *= dims.num_channels_;
      [[fallthrough]];
    case 2:
      width *= dims.height_;
      [[fallthrough]];
    case 3:
      width *= dims.width_;
  }
  switch (axis) {
    case 4:
      height *= dims.width_;
      [[fallthrough]];
    case 3:
      height *= dims.height_;
      [[fallthrough]];
    case 2:
      height *= dims.num_channels_;
      [[fallthrough]];
    case 1:
      height *= dims.batch_size_;
  }
  return {.batch_size_ = 1, .num_channels_ = 1, .height_ = height, .width_ = width};
}

}  // namespace MOTION::tensor

namespace std {

std::size_t std::hash<MOTION::tensor::Conv2DOp>::operator()(
    const MOTION::tensor::Conv2DOp& op) const noexcept {
  std::size_t seed = 0;
  boost::hash_combine(seed,
                      boost::hash_range(std::begin(op.kernel_shape_), std::end(op.kernel_shape_)));
  boost::hash_combine(seed,
                      boost::hash_range(std::begin(op.input_shape_), std::end(op.input_shape_)));
  boost::hash_combine(seed, boost::hash_range(std::begin(op.dilations_), std::end(op.dilations_)));
  boost::hash_combine(seed, boost::hash_range(std::begin(op.pads_), std::end(op.pads_)));
  boost::hash_combine(seed, boost::hash_range(std::begin(op.strides_), std::end(op.strides_)));
  return seed;
}

std::size_t std::hash<MOTION::tensor::GemmOp>::operator()(
    const MOTION::tensor::GemmOp& op) const noexcept {
  std::size_t seed = 0;
  boost::hash_combine(
      seed, boost::hash_range(std::begin(op.input_A_shape_), std::end(op.input_A_shape_)));
  boost::hash_combine(
      seed, boost::hash_range(std::begin(op.input_B_shape_), std::end(op.input_B_shape_)));
  return seed;
}

}  // namespace std
