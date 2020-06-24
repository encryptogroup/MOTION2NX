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

// #include <array>
#include <cstddef>
#include <memory>
// #include <vector>

#include "wire/new_wire.h"

namespace MOTION::tensor {

struct TensorDimensions {
  std::size_t batch_size_;
  std::size_t num_channels_;
  std::size_t height_;
  std::size_t width_;
  std::size_t get_num_dimensions() const noexcept { return 4; }
  std::size_t get_data_size() const noexcept {
    return batch_size_ * num_channels_ * height_ * width_;
  }
  bool operator==(const TensorDimensions& o) const noexcept {
    return batch_size_ == o.batch_size_ && num_channels_ == o.num_channels_ &&
           height_ == o.height_ && width_ == o.width_;
  }
  bool operator!=(const TensorDimensions& o) const noexcept { return !(*this == o); }
};

// enum class CircuitType : unsigned int;
// enum class MPCProtocol : unsigned int;

class Tensor : public NewWire {
 public:
  Tensor(const TensorDimensions& dimensions) : NewWire(1), dimensions_(dimensions) {}
  virtual ~Tensor() = default;
  std::size_t get_num_dimensions() const noexcept { return 4; }
  const TensorDimensions& get_dimensions() const noexcept { return dimensions_; }
  // virtual std::size_t get_dimension(std::size_t) const = 0;
 private:
  const TensorDimensions dimensions_;
};

using TensorP = std::shared_ptr<Tensor>;
using TensorCP = std::shared_ptr<const Tensor>;

}  // namespace MOTION::tensor
