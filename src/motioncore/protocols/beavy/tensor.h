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

namespace MOTION::proto::beavy {

template <typename T>
class ArithmeticBEAVYTensor : public tensor::Tensor, public ENCRYPTO::enable_wait_setup {
 public:
  using Tensor::Tensor;
  MPCProtocol get_protocol() const noexcept override { return MPCProtocol::ArithmeticBEAVY; }
  std::size_t get_bit_size() const noexcept override { return ENCRYPTO::bit_size_v<T>; }
  std::vector<T>& get_public_share() { return public_share_; };
  const std::vector<T>& get_public_share() const { return public_share_; };
  std::vector<T>& get_secret_share() { return secret_share_; };
  const std::vector<T>& get_secret_share() const { return secret_share_; };

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  std::vector<T> public_share_;
  std::vector<T> secret_share_;
};

template <typename T>
using ArithmeticBEAVYTensorP = std::shared_ptr<ArithmeticBEAVYTensor<T>>;

template <typename T>
using ArithmeticBEAVYTensorCP = std::shared_ptr<const ArithmeticBEAVYTensor<T>>;

template <typename T>
std::ostream& operator<<(std::ostream& os, const ArithmeticBEAVYTensor<T>& w) {
  return os << "<ArithmeticBEAVYTensor<T> @ " << &w << ">";
}

class BooleanBEAVYTensor : public tensor::Tensor, public ENCRYPTO::enable_wait_setup {
 public:
  BooleanBEAVYTensor(const tensor::TensorDimensions& dims, std::size_t bit_size)
      : Tensor(dims), bit_size_(bit_size), public_share_(bit_size), secret_share_(bit_size) {}
  MPCProtocol get_protocol() const noexcept override { return MPCProtocol::BooleanBEAVY; }
  std::size_t get_bit_size() const noexcept override { return bit_size_; }
  std::vector<ENCRYPTO::BitVector<>>& get_public_share() noexcept { return public_share_; }
  const std::vector<ENCRYPTO::BitVector<>>& get_public_share() const noexcept {
    return public_share_;
  }
  std::vector<ENCRYPTO::BitVector<>>& get_secret_share() noexcept { return secret_share_; }
  const std::vector<ENCRYPTO::BitVector<>>& get_secret_share() const noexcept {
    return secret_share_;
  }

 private:
  std::size_t bit_size_;
  std::vector<ENCRYPTO::BitVector<>> public_share_;
  std::vector<ENCRYPTO::BitVector<>> secret_share_;
};

using BooleanBEAVYTensorP = std::shared_ptr<BooleanBEAVYTensor>;

using BooleanBEAVYTensorCP = std::shared_ptr<const BooleanBEAVYTensor>;

inline std::ostream& operator<<(std::ostream& os, const BooleanBEAVYTensor& w) {
  return os << "<BooleanBEAVYTensor @ " << &w << ">";
}

}  // namespace MOTION::proto::beavy
