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

#include <cstdint>
#include <memory>
#include <vector>

#include "tensor_op.h"
#include "utility/reusable_future.h"

namespace MOTION::tensor {

struct TensorDimensions;
struct Conv2DOp;
struct GemmOp;
class Tensor;
using TensorP = std::shared_ptr<Tensor>;
using TensorCP = std::shared_ptr<const Tensor>;

template <typename T>
using IntegerValues = std::vector<T>;

class TensorOpFactory {
 public:
  virtual ~TensorOpFactory() = default;

  virtual std::string get_provider_name() const noexcept = 0;

  // arithmetic inputs
  virtual std::pair<ENCRYPTO::ReusableFiberPromise<IntegerValues<std::uint64_t>>, TensorCP>
  make_arithmetic_64_tensor_input_my(const TensorDimensions&);
  virtual TensorCP make_arithmetic_64_tensor_input_other(const TensorDimensions&);

  // arithmetic outputs
  virtual ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint64_t>>
  make_arithmetic_64_tensor_output_my(const TensorCP&);
  virtual void make_arithmetic_tensor_output_other(const TensorCP&);

  // conversions
  virtual tensor::TensorCP make_tensor_conversion(MPCProtocol, const tensor::TensorCP input);

  // operations
  virtual tensor::TensorCP make_tensor_flatten_op(const tensor::TensorCP input, std::size_t axis);
  virtual tensor::TensorCP make_tensor_conv2d_op(const tensor::Conv2DOp& conv_op,
                                                 const tensor::TensorCP input,
                                                 const tensor::TensorCP kernel,
                                                 const tensor::TensorCP bias,
                                                 std::size_t truncate_bits = 0);
  virtual tensor::TensorCP make_tensor_conv2d_op(const tensor::Conv2DOp& conv_op,
                                                 const tensor::TensorCP input,
                                                 const tensor::TensorCP kernel,
                                                 std::size_t truncate_bits = 0);
  virtual tensor::TensorCP make_tensor_gemm_op(const tensor::GemmOp& gemm_op,
                                               const tensor::TensorCP input_A,
                                               const tensor::TensorCP input_B,
                                               std::size_t truncate_bits = 0);
  virtual tensor::TensorCP make_tensor_sqr_op(const tensor::TensorCP input,
                                              std::size_t truncate_bits = 0);
  virtual tensor::TensorCP make_tensor_relu_op(const tensor::TensorCP input);
  virtual tensor::TensorCP make_tensor_relu_op(const tensor::TensorCP input_bool,
                                               const tensor::TensorCP input_arith);
  virtual tensor::TensorCP make_tensor_maxpool_op(const tensor::MaxPoolOp& maxpool_op,
                                                  const tensor::TensorCP input);
};

}  // namespace MOTION::tensor
