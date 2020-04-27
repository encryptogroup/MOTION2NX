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

namespace MOTION::proto::gmw {

class GMWProvider;

class BooleanGMWWire : public NewWire {
 public:
  BooleanGMWWire(std::size_t num_simd) : NewWire(num_simd), share_(num_simd) {}
  MPCProtocol get_protocol() const noexcept override { return MPCProtocol::BooleanGMW; }
  ENCRYPTO::BitVector<>& get_share() { return share_; };
  const ENCRYPTO::BitVector<>& get_share() const { return share_; };

 private:
  // holds this party shares
  ENCRYPTO::BitVector<> share_;
};

using BooleanGMWWireVector = std::vector<std::shared_ptr<BooleanGMWWire>>;

template <typename T>
class ArithmeticGMWWire : public NewWire {
 public:
  ArithmeticGMWWire(std::size_t num_simd) : NewWire(num_simd), share_(num_simd) {}
  MPCProtocol get_protocol() const noexcept override { return MPCProtocol::ArithmeticGMW; }
  std::vector<T>& get_share() { return share_; };
  const std::vector<T>& get_share() const { return share_; };

 private:
  using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;

  // holds this party shares
  std::vector<T> share_;
};

template <typename T>
using ArithmeticGMWWireVector = std::vector<std::shared_ptr<ArithmeticGMWWire<T>>>;

}  // namespace MOTION::proto::gmw
