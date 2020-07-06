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

#include <array>
#include <cstddef>
#include <functional>
#include <utility>

#include "tensor.h"

namespace MOTION::tensor {

struct TensorOp {
  virtual ~TensorOp() = default;
};

struct Conv2DOp {
  std::array<std::size_t, 4> kernel_shape_;
  std::array<std::size_t, 3> input_shape_;
  std::array<std::size_t, 3> output_shape_;

  std::array<std::size_t, 2> dilations_;
  std::array<std::size_t, 4> pads_;
  std::array<std::size_t, 2> strides_;

  bool verify() const noexcept;
  std::array<std::size_t, 3> compute_output_shape() const noexcept;
  std::size_t compute_output_size() const noexcept;
  std::size_t compute_input_size() const noexcept;
  std::size_t compute_kernel_size() const noexcept;
  std::pair<std::size_t, std::size_t> compute_input_matrix_shape() const noexcept;
  std::pair<std::size_t, std::size_t> compute_kernel_matrix_shape() const noexcept;
  std::pair<std::size_t, std::size_t> compute_output_matrix_shape() const noexcept;
  TensorDimensions get_input_tensor_dims() const noexcept;
  TensorDimensions get_kernel_tensor_dims() const noexcept;
  TensorDimensions get_output_tensor_dims() const noexcept;

  bool operator==(const Conv2DOp&) const noexcept;
};

struct GemmOp {
  std::array<std::size_t, 2> input_A_shape_;
  std::array<std::size_t, 2> input_B_shape_;
  std::array<std::size_t, 2> output_shape_;

  bool verify() const noexcept;
  std::array<std::size_t, 2> compute_output_shape() const noexcept;
  std::size_t compute_output_size() const noexcept;
  std::size_t compute_input_A_size() const noexcept;
  std::size_t compute_input_B_size() const noexcept;
  TensorDimensions get_input_A_tensor_dims() const noexcept;
  TensorDimensions get_input_B_tensor_dims() const noexcept;
  TensorDimensions get_output_tensor_dims() const noexcept;

  bool operator==(const GemmOp&) const noexcept;
};

TensorDimensions flatten(const TensorDimensions& dims, std::size_t axis);

}  // namespace MOTION::tensor

namespace std {

template <>
struct hash<MOTION::tensor::Conv2DOp> {
  std::size_t operator()(const MOTION::tensor::Conv2DOp&) const noexcept;
};

template <>
struct hash<MOTION::tensor::GemmOp> {
  std::size_t operator()(const MOTION::tensor::GemmOp&) const noexcept;
};

}  // namespace std
