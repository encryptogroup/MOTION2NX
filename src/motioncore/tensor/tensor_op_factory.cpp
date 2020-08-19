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

#include "tensor_op_factory.h"

#include <stdexcept>

#include <fmt/format.h>

namespace MOTION::tensor {

std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint64_t>>, TensorCP>
TensorOpFactory::make_arithmetic_64_tensor_input_my(const TensorDimensions&) {
  throw std::logic_error(
      fmt::format("{} does not support arithmetic 64 bit inputs", get_provider_name()));
}

TensorCP TensorOpFactory::make_arithmetic_64_tensor_input_other(const TensorDimensions&) {
  throw std::logic_error(
      fmt::format("{} does not support arithmetic 64 bit inputs", get_provider_name()));
}

ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint64_t>>
TensorOpFactory::make_arithmetic_64_tensor_output_my(const TensorCP&) {
  throw std::logic_error(
      fmt::format("{} does not support arithmetic 64 bit outputs", get_provider_name()));
}

void TensorOpFactory::make_arithmetic_tensor_output_other(const TensorCP&) {
  throw std::logic_error(
      fmt::format("{} does not support arithmetic outputs", get_provider_name()));
}

tensor::TensorCP TensorOpFactory::make_tensor_conversion(MPCProtocol, const tensor::TensorCP) {
  throw std::logic_error(
      fmt::format("{} does not support conversions to other protocols", get_provider_name()));
}

tensor::TensorCP TensorOpFactory::make_tensor_flatten_op(const tensor::TensorCP, std::size_t) {
  throw std::logic_error(
      fmt::format("{} does not support the Flatten operation", get_provider_name()));
}

tensor::TensorCP TensorOpFactory::make_tensor_conv2d_op(const tensor::Conv2DOp&,
                                                        const tensor::TensorCP,
                                                        const tensor::TensorCP,
                                                        const tensor::TensorCP) {
  throw std::logic_error(
      fmt::format("{} does not support the Conv2D operation", get_provider_name()));
}

tensor::TensorCP TensorOpFactory::make_tensor_conv2d_op(const tensor::Conv2DOp& op,
                                                        const tensor::TensorCP input,
                                                        const tensor::TensorCP kernel) {
  return make_tensor_conv2d_op(op, input, kernel, nullptr);
}

tensor::TensorCP TensorOpFactory::make_tensor_gemm_op(const tensor::GemmOp&, const tensor::TensorCP,
                                                      const tensor::TensorCP) {
  throw std::logic_error(
      fmt::format("{} does not support the Gemm operation", get_provider_name()));
}

tensor::TensorCP TensorOpFactory::make_tensor_sqr_op(const tensor::TensorCP) {
  throw std::logic_error(fmt::format("{} does not support the Sqr operation", get_provider_name()));
}

tensor::TensorCP TensorOpFactory::make_tensor_relu_op(const tensor::TensorCP) {
  throw std::logic_error(
      fmt::format("{} does not support the ReLU operation", get_provider_name()));
}

tensor::TensorCP TensorOpFactory::make_tensor_relu_op(const tensor::TensorCP,
                                                      const tensor::TensorCP) {
  throw std::logic_error(
      fmt::format("{} does not support the ReLU (arith x Bool) operation", get_provider_name()));
}

tensor::TensorCP TensorOpFactory::make_tensor_maxpool_op(const tensor::MaxPoolOp&,
                                                         const tensor::TensorCP) {
  throw std::logic_error(
      fmt::format("{} does not support the MaxPool operation", get_provider_name()));
}

}  // namespace MOTION::tensor
