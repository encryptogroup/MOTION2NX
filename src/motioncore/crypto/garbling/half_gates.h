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

#include "crypto/aes/aesni_primitives.h"
#include "utility/block.h"

struct HalfGatePublicData {
  ENCRYPTO::block128_t hash_key;
  ENCRYPTO::block128_t aes_key;
};

using half_gate_t = std::array<ENCRYPTO::block128_t, 2>;

class HalfGateGarbler {
 public:
  HalfGateGarbler();
  HalfGatePublicData get_public_data() const;
  ENCRYPTO::block128_t get_offset() const;
  void garble_and(ENCRYPTO::block128_t& key_c, half_gate_t& garbled_table, std::size_t index,
                  const ENCRYPTO::block128_t& key_a, const ENCRYPTO::block128_t& key_b) const;
  void batch_garble_and(ENCRYPTO::block128_vector& key_c, std::vector<half_gate_t>& garbled_table,
                        std::size_t index, const ENCRYPTO::block128_vector& key_a,
                        const ENCRYPTO::block128_vector& key_b) const;

 private:
  ENCRYPTO::block128_t offset_;
  ENCRYPTO::block128_t hash_key_;
  alignas(aes_block_size) std::array<std::byte, aes_round_keys_size_128> round_keys_;
};

class HalfGateEvaluator {
 public:
  HalfGateEvaluator(const HalfGatePublicData& public_data);

  void evaluate_and(ENCRYPTO::block128_t& key_c, const half_gate_t& garbled_table,
                    std::size_t index, const ENCRYPTO::block128_t& key_a,
                    const ENCRYPTO::block128_t& key_b) const;
  void batch_evaluate_and(ENCRYPTO::block128_vector& key_c,
                          const std::vector<half_gate_t>& garbled_table, std::size_t index,
                          const ENCRYPTO::block128_vector& key_a,
                          const ENCRYPTO::block128_vector& key_b) const;

 private:
  ENCRYPTO::block128_t hash_key_;
  alignas(aes_block_size) std::array<std::byte, aes_round_keys_size_128> round_keys_;
};
