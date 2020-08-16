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

#include "tensor/tensor.h"
#include "utility/bit_vector.h"
#include "utility/enable_wait.h"
#include "utility/type_traits.hpp"
#include "utility/typedefs.h"

namespace MOTION::proto::gmw {

template <typename T>
class ArithmeticGMWTensor : public tensor::Tensor {
 public:
  using Tensor::Tensor;
  MPCProtocol get_protocol() const noexcept override { return MPCProtocol::ArithmeticGMW; }
  std::size_t get_bit_size() const noexcept override { return ENCRYPTO::bit_size_v<T>; }
  std::vector<T>& get_share() noexcept { return data_; }
  const std::vector<T>& get_share() const noexcept { return data_; }

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::vector<T> data_;
};

template <typename T>
using ArithmeticGMWTensorP = std::shared_ptr<ArithmeticGMWTensor<T>>;

template <typename T>
using ArithmeticGMWTensorCP = std::shared_ptr<const ArithmeticGMWTensor<T>>;

class BooleanGMWTensor : public tensor::Tensor {
 public:
  BooleanGMWTensor(const tensor::TensorDimensions& dims, std::size_t bit_size)
      : Tensor(dims), bit_size_(bit_size), data_(bit_size) {}
  MPCProtocol get_protocol() const noexcept override { return MPCProtocol::BooleanGMW; }
  std::size_t get_bit_size() const noexcept override { return bit_size_; }
  std::vector<ENCRYPTO::BitVector<>>& get_share() noexcept { return data_; }
  const std::vector<ENCRYPTO::BitVector<>>& get_share() const noexcept { return data_; }

 private:
  std::size_t bit_size_;
  std::vector<ENCRYPTO::BitVector<>> data_;
};

using BooleanGMWTensorP = std::shared_ptr<BooleanGMWTensor>;

using BooleanGMWTensorCP = std::shared_ptr<const BooleanGMWTensor>;

}  // namespace MOTION::proto::gmw
