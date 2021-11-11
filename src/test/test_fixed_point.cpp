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

#include <gtest/gtest.h>
#include <cstdint>

#include "utility/fixed_point.h"

namespace fp = MOTION::fixed_point;


TEST(FixedPoint, EncodeDecodeTruncate) {
  const std::size_t fractional_bits = 24;
  const double max_error = 1.0 / (1 << 10);

  const double f = 13.374247;
  const auto enc_f = fp::encode<std::uint64_t, double>(f, fractional_bits);
  const auto dec_f = fp::decode<std::uint64_t, double>(enc_f, fractional_bits);
  EXPECT_NEAR(dec_f, f, max_error);

  const double g = -48.768676;
  const auto enc_g = fp::encode<std::uint64_t, double>(g, fractional_bits);
  const auto dec_g = fp::decode<std::uint64_t, double>(enc_g, fractional_bits);
  EXPECT_NEAR(dec_g, g, max_error);

  const double h = f * g;
  const auto enc_h = fp::truncate<std::uint64_t>(enc_f * enc_g, fractional_bits);
  const auto dec_h = fp::decode<std::uint64_t, double>(enc_h, fractional_bits);
  EXPECT_NEAR(dec_h, h, max_error);
}

TEST(FixedPoint, TruncateShared) {
  const std::size_t fractional_bits = 24;
  const double max_error = 1.0 / (1 << 10);

  // define factors and encode them as fixed-point numbers
  const double f = 13.374247;
  const double g = -48.768676;
  const auto enc_f = fp::encode<std::uint64_t, double>(f, fractional_bits);
  const auto enc_g = fp::encode<std::uint64_t, double>(g, fractional_bits);

  // compute product in plain as doubles
  const double h = f * g;
  // compute product in plain as fixed-point numbers, but do *not* truncate
  const auto enc_h = enc_f * enc_g;

  // emulate an additive secret sharing (GMW-style)
  const std::uint64_t enc_h_0 = 0x4242424242424242;
  const std::uint64_t enc_h_1 = enc_h - enc_h_0;
  ASSERT_EQ(enc_h, enc_h_0 + enc_h_1);

  // run the truncation protocol on each share separately
  auto enc_h_truncated_0 = enc_h_0;
  fp::truncate_shared(&enc_h_truncated_0, fractional_bits, 1, 0);
  auto enc_h_truncated_1 = enc_h_1;
  fp::truncate_shared(&enc_h_truncated_1, fractional_bits, 1, 1);

  // reconstruct the truncated fixed-point number and decode it into a double
  const auto enc_h_truncated = enc_h_truncated_0 + enc_h_truncated_1;

  // decode the truncated fixed-point number into a double
  const auto dec_h = fp::decode<std::uint64_t, double>(enc_h_truncated, fractional_bits);
  // check that the result matches the product computed on doubles in plaintext
  EXPECT_NEAR(dec_h, h, max_error);
}
