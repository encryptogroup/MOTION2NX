// MIT License
//
// Copyright (c) 2018-2019 Lennart Braun
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

#include <random>

#include "gtest/gtest.h"

#include "test_constants.h"

#include "crypto/garbling/half_gates.h"

using namespace MOTION::Crypto::garbling;

TEST(half_gates, garble_eval) {
  HalfGateGarbler garbler;
  HalfGateEvaluator evaluator(garbler.get_public_data());

  const auto offset = garbler.get_offset();
  const auto key_a = ENCRYPTO::block128_t::make_random();
  const auto key_b = ENCRYPTO::block128_t::make_random();
  const std::size_t index = 42;

  ENCRYPTO::block128_t key_c_original;
  std::array<ENCRYPTO::block128_t, 2> garbled_table;

  garbler.garble_and(key_c_original, garbled_table.data(), index, key_a, key_b);

  ENCRYPTO::block128_t key_c;

  evaluator.evaluate_and(key_c, garbled_table.data(), index, key_a, key_b);
  EXPECT_EQ(key_c, key_c_original);
  evaluator.evaluate_and(key_c, garbled_table.data(), index, key_a ^ offset, key_b);
  EXPECT_EQ(key_c, key_c_original);
  evaluator.evaluate_and(key_c, garbled_table.data(), index, key_a, key_b ^ offset);
  EXPECT_EQ(key_c, key_c_original);

  evaluator.evaluate_and(key_c, garbled_table.data(), index, key_a ^ offset, key_b ^ offset);
  EXPECT_EQ(key_c, key_c_original ^ offset);
}

TEST(half_gates, batch_garble_eval) {
  HalfGateGarbler garbler;
  HalfGateEvaluator evaluator(garbler.get_public_data());

  const std::size_t size = 1024;
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(size);
  auto key_bs = ENCRYPTO::block128_vector::make_random(size);
  const std::size_t index = 42;

  ENCRYPTO::block128_vector key_cs_original(size);
  ENCRYPTO::block128_vector garbled_tables(2 * size);

  garbler.batch_garble_and(key_cs_original, garbled_tables.data(), index, key_as, key_bs);

  ENCRYPTO::block128_vector key_cs(size);

  std::minstd_rand gen_a(0x61);
  std::minstd_rand gen_b(0x62);
  std::uniform_int_distribution dist(0, 1);

  for (std::size_t i = 0; i < size; ++i) {
    if (dist(gen_a) == 1) key_as[i] ^= offset;
    if (dist(gen_b) == 1) key_bs[i] ^= offset;
  }

  evaluator.batch_evaluate_and(key_cs, garbled_tables.data(), index, key_as, key_bs);
  gen_a.seed(0x61);
  gen_b.seed(0x62);

  for (std::size_t i = 0; i < size; ++i) {
    if (dist(gen_a) + dist(gen_b) == 2)
      EXPECT_EQ(key_cs[i], key_cs_original[i] ^ offset);
    else
      EXPECT_EQ(key_cs[i], key_cs_original[i]);
  }
}
