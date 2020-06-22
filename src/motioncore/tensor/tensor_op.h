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
#include <utility>

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

  bool verify() const;
  std::array<std::size_t, 3> compute_output_shape() const;
  std::size_t compute_output_size() const;
  std::size_t compute_input_size() const;
  std::size_t compute_kernel_size() const;
  std::pair<std::size_t, std::size_t> compute_input_matrix_shape() const;
  std::pair<std::size_t, std::size_t> compute_kernel_matrix_shape() const;
  std::pair<std::size_t, std::size_t> compute_output_matrix_shape() const;
};

}
