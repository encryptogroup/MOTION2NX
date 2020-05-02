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

#include "utility/type_traits.hpp"

#include <cstdint>

static void test_is_unsigned_int() {
  using is_unsigned_8_t = ENCRYPTO::is_unsigned_int_t<std::uint8_t>;
  using is_unsigned_16_t = ENCRYPTO::is_unsigned_int_t<std::uint16_t>;
  using is_unsigned_32_t = ENCRYPTO::is_unsigned_int_t<std::uint32_t>;
  using is_unsigned_64_t = ENCRYPTO::is_unsigned_int_t<std::uint64_t>;
  using is_unsigned_128_t = ENCRYPTO::is_unsigned_int_t<__uint128_t>;
}

static void test_bit_size() {
  static_assert(ENCRYPTO::bit_size_v<std::uint8_t> == 8);
  static_assert(ENCRYPTO::bit_size_v<std::uint16_t> == 16);
  static_assert(ENCRYPTO::bit_size_v<std::uint32_t> == 32);
  static_assert(ENCRYPTO::bit_size_v<std::uint64_t> == 64);
  static_assert(ENCRYPTO::bit_size_v<__uint128_t> == 128);
}
