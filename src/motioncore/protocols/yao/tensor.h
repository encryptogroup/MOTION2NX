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

#include <cstddef>
#include <memory>

#include "tensor/tensor.h"
#include "utility/block.h"
#include "utility/typedefs.h"

namespace MOTION::proto::yao {

class YaoTensor : public tensor::Tensor, public ENCRYPTO::enable_wait_setup {
 public:
  YaoTensor(const tensor::TensorDimensions& dims, std::size_t bit_size)
      : Tensor(dims), bit_size_(bit_size) {}
  MPCProtocol get_protocol() const noexcept override { return MPCProtocol::Yao; }
  std::size_t get_bit_size() const noexcept override { return bit_size_; }
  ENCRYPTO::block128_vector& get_keys() noexcept { return keys_; }
  const ENCRYPTO::block128_vector& get_keys() const noexcept { return keys_; }

 private:
  std::size_t bit_size_;
  ENCRYPTO::block128_vector keys_;
};

using YaoTensorP = std::shared_ptr<YaoTensor>;

using YaoTensorCP = std::shared_ptr<const YaoTensor>;

inline std::ostream& operator<<(std::ostream& os, const YaoTensor& w) {
  return os << "<YaoTensor @ " << &w << ">";
}

}  // namespace MOTION::proto::yao
