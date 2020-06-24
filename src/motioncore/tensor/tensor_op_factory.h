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

#include "utility/reusable_future.h"

namespace MOTION::tensor {

struct TensorDimensions;
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
  make_arithmetic_64_tensor_input_my(const TensorDimensions&) = 0;

  virtual TensorCP make_arithmetic_64_tensor_input_other(const TensorDimensions&) = 0;

  // arithmetic outputs
  virtual ENCRYPTO::ReusableFiberFuture<IntegerValues<std::uint64_t>>
  make_arithmetic_64_tensor_output_my(const TensorCP&) = 0;

  virtual void make_arithmetic_tensor_output_other(const TensorCP&) = 0;
};

}  // namespace MOTION::tensor
