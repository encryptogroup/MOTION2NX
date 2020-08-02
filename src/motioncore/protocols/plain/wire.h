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

#include <memory>
#include <vector>

#include "utility/bit_vector.h"
#include "utility/type_traits.hpp"
#include "utility/typedefs.h"
#include "wire/new_wire.h"

namespace MOTION::proto::plain {

class PlainProvider;

class BooleanPlainWire : public NewWire {
 public:
  BooleanPlainWire(ENCRYPTO::BitVector<> data) : NewWire(data.GetSize()), data_(move(data)) {
    set_online_ready();
  }
  MPCProtocol get_protocol() const noexcept override { return MPCProtocol::BooleanPlain; }
  std::size_t get_bit_size() const noexcept override { return 1; }
  const ENCRYPTO::BitVector<>& get_data() const { return data_; };

 private:
  const ENCRYPTO::BitVector<> data_;
};

using BooleanPlainWireVector = std::vector<std::shared_ptr<BooleanPlainWire>>;

template <typename T>
class ArithmeticPlainWire : public NewWire {
 public:
  ArithmeticPlainWire(std::vector<T> data) : NewWire(data.size()), data_(std::move(data)) {
    set_online_ready();
  }
  MPCProtocol get_protocol() const noexcept override { return MPCProtocol::ArithmeticPlain; }
  std::size_t get_bit_size() const noexcept override { return ENCRYPTO::bit_size_v<T>; }
  const std::vector<T>& get_data() const { return data_; };

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  const std::vector<T> data_;
};

template <typename T>
using ArithmeticPlainWireP = std::shared_ptr<ArithmeticPlainWire<T>>;
template <typename T>
using ArithmeticPlainWireVector = std::vector<std::shared_ptr<ArithmeticPlainWire<T>>>;

}  // namespace MOTION::proto::plain
