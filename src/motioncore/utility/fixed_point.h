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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <type_traits>

namespace MOTION::fixed_point {

namespace {

template <typename I, typename F>
constexpr bool check_types_v =
    std::is_unsigned_v<I> && !std::is_same_v<I, bool> && std::is_floating_point_v<F>;

template <typename T>
constexpr std::size_t bit_size_v = 8 * sizeof(T);

template <typename T>
T sar(T x, std::size_t n) {
  constexpr T mask = T(1) << (bit_size_v<T> - 1);
  for (std::size_t i = 0; i < n; ++i) {
    x = (mask & x) | (x >> 1);
  }
  return x;
}

}  // namespace

// Encode a floating point number `x` of type `F` as a fixed point number using
// integer type `I` and `fractional_bits` bits for the fractional part.
template <typename I, typename F, typename = std::enable_if_t<check_types_v<I, F>>>
I encode(F x, std::size_t fractional_bits) {
  assert(fractional_bits <= bit_size_v<I>);
  if (x >= 0) {
    // return std::lround(x * (I(1) << fractional_bits));
    return std::lround(x * std::exp2(fractional_bits));
  } else {
    return std::lround(x * std::exp2(fractional_bits));
  }
}

// Decode a fixed point number `y` encoded in integer type `T` with
// `fractional_bits` bits for the fractional part to a floating point number of
// type `F`.
template <typename I, typename F, typename = std::enable_if_t<check_types_v<I, F>>>
F decode(I y, std::size_t fractional_bits) {
  assert(fractional_bits <= bit_size_v<I>);
  if (y < (I(1) << (bit_size_v<I> - 1))) {
    return double(y) / std::exp2(fractional_bits);
  } else {
    return (double(y) - std::exp2(bit_size_v<I>)) / std::exp2(fractional_bits);
  }
}

template <typename I>
I truncate(I y, std::size_t fractional_bits) {
  return sar(y, fractional_bits);
}

// Encode a range of floating point numbers using above encoding algorithm.
template <typename I, typename F>
void encode(I* dst, const F* src, std::size_t fractional_bits, std::size_t n) {
  std::transform(src, src + n, dst,
                 [fractional_bits](auto x) { return encode(x, fractional_bits); });
}

// Decode a range of floating point numbers using above decoding algorithm.
template <typename I, typename F>
void decode(F* dst, const I* src, std::size_t fractional_bits, std::size_t n) {
  std::transform(src, src + n, dst,
                 [fractional_bits](auto x) { return decode(x, fractional_bits); });
}

// Truncation protocol by Mohassel and Zhang (https://eprint.iacr.org/2017/396).
template <typename T>
void truncate_shared(T* buffer, std::size_t fractional_bits, std::size_t n, bool party_0) {
  if (party_0) {
    std::transform(buffer, buffer + n, buffer,
                   [fractional_bits](auto x) { return x >> fractional_bits; });
  } else {
    std::transform(buffer, buffer + n, buffer,
                   [fractional_bits](auto x) { return -((-x) >> fractional_bits); });
  }
}

}  // namespace MOTION::fixed_point
