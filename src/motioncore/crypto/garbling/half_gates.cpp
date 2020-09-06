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

#include "half_gates.h"

#include <parallel/algorithm>
#include <numeric>

#include "algorithm/algorithm_description.h"
#include "crypto/aes/aesni_primitives.h"

namespace MOTION::Crypto::garbling {

HalfGateGarbler::HalfGateGarbler()
    : offset_(ENCRYPTO::block128_t::make_random()), hash_key_(ENCRYPTO::block128_t::make_random()) {
  reinterpret_cast<ENCRYPTO::block128_t*>(round_keys_.data())->set_to_random();
  aesni_key_expansion_128(round_keys_.data());
  offset_.byte_array[0] |= std::byte(1);  // LSB needs to be 1 for freeXOR
}

HalfGatePublicData HalfGateGarbler::get_public_data() const noexcept {
  return {hash_key_, *reinterpret_cast<const ENCRYPTO::block128_t*>(round_keys_.data())};
}

ENCRYPTO::block128_t HalfGateGarbler::get_offset() const noexcept { return offset_; }

void HalfGateGarbler::garble_and(ENCRYPTO::block128_t& key_c, ENCRYPTO::block128_t* garbled_table,
                                 std::size_t index, const ENCRYPTO::block128_t& key_a,
                                 const ENCRYPTO::block128_t& key_b) const {
  // TODO: avoid the jumps

  // permutation bits
  bool p_a = static_cast<bool>(key_a.byte_array[0] & std::byte(1));
  bool p_b = static_cast<bool>(key_b.byte_array[0] & std::byte(1));

  std::array<ENCRYPTO::block128_t, 4> hash_inputs = {key_a, key_b, key_a ^ offset_,
                                                     key_b ^ offset_};
  // compute H(W_a^0, j), H(W_b^0, j'), H(W_a^1, j), H(W_b^1, j')
  aesni_fixed_key_for_half_gates_batch_4(round_keys_.data(), hash_key_.data(), index,
                                         hash_inputs.data());

  // T_G <- H(W_a^0, j) ^ H(W_a^1, j) ^ (p_b * R)
  garbled_table[0] = hash_inputs[0] ^ hash_inputs[2];
  if (p_b) garbled_table[0] ^= offset_;

  // T_E <- H(W_b^0, j') ^ H(W_b^1, j') ^ W_a^0
  garbled_table[1] = hash_inputs[1] ^ hash_inputs[3] ^ key_a;

  // W_G^0 <- H(W_a^0, j) ^ (p_b * T_G)
  key_c = hash_inputs[0];
  if (p_a) key_c ^= garbled_table[0];

  // W_E^0 <- H(W_b^0, j') ^ (p_b * (T_E ^ W_a^0))
  key_c ^= hash_inputs[1];
  if (p_b) key_c ^= garbled_table[1] ^ key_a;
}

void HalfGateGarbler::batch_garble_and(ENCRYPTO::block128_t* key_cs,
                                       ENCRYPTO::block128_t* garbled_tables,
                                       std::size_t start_index, const ENCRYPTO::block128_t* key_as,
                                       const ENCRYPTO::block128_t* key_bs,
                                       std::size_t num_gates) const {
  for (std::size_t i = 0; i < num_gates; ++i) {
    garble_and(key_cs[i], &garbled_tables[2 * i], start_index + i, key_as[i], key_bs[i]);
  }
}

void HalfGateGarbler::batch_garble_and(ENCRYPTO::block128_vector& key_cs,
                                       ENCRYPTO::block128_t* garbled_tables,
                                       std::size_t start_index,
                                       const ENCRYPTO::block128_vector& key_as,
                                       const ENCRYPTO::block128_vector& key_bs) const {
  assert(key_as.size() == key_bs.size());
  std::size_t num_gates = key_as.size();
  key_cs.resize(num_gates);
  batch_garble_and(key_cs.data(), garbled_tables, start_index, key_as.data(), key_bs.data(),
                   num_gates);
}

void HalfGateGarbler::batch_garble_and_omp(ENCRYPTO::block128_t* key_cs,
                                           ENCRYPTO::block128_t* garbled_tables,
                                           std::size_t start_index,
                                           const ENCRYPTO::block128_t* key_as,
                                           const ENCRYPTO::block128_t* key_bs,
                                           std::size_t num_gates) const {
#pragma omp parallel for
  for (std::size_t i = 0; i < num_gates; ++i) {
    garble_and(key_cs[i], &garbled_tables[2 * i], start_index + i, key_as[i], key_bs[i]);
  }
}

void HalfGateGarbler::garble_circuit(
    ENCRYPTO::block128_vector& output_keys, ENCRYPTO::block128_vector& garbled_tables,
    std::size_t start_index, const ENCRYPTO::block128_vector& input_keys_a,
    const ENCRYPTO::block128_vector& input_keys_b, std::size_t num_simd,
    const ENCRYPTO::AlgorithmDescription& algo, bool parallel) const {
  assert(input_keys_a.size() == algo.n_input_wires_parent_a_ * num_simd);
  assert((!algo.n_input_wires_parent_b_.has_value()) ||
         (input_keys_b.size() == *algo.n_input_wires_parent_b_ * num_simd));
  output_keys.resize(algo.n_output_wires_ * num_simd);
  std::size_t num_and_gates = std::transform_reduce(
      std::begin(algo.gates_), std::end(algo.gates_), 0, std::plus{}, [](const auto& op) {
        if (op.type_ == ENCRYPTO::PrimitiveOperationType::AND) {
          return 1;
        } else {
          return 0;
        }
      });
  garbled_tables.resize(2 * num_and_gates * num_simd);
  ENCRYPTO::block128_vector wire_keys(algo.n_wires_ * num_simd);
  auto it = std::copy_n(input_keys_a.data(), input_keys_a.size(), wire_keys.data());
  if (algo.n_input_wires_parent_b_.has_value()) {
    std::copy_n(input_keys_b.data(), input_keys_b.size(), it);
  }
  assert(algo.n_gates_ == algo.gates_.size());
  for (std::size_t op_i = 0, and_j = 0; op_i < algo.n_gates_; ++op_i) {
    const auto& op = algo.gates_[op_i];
    const auto* gate_input_keys_a = &wire_keys[op.parent_a_ * num_simd];
    auto* gate_output_keys = &wire_keys[op.output_wire_ * num_simd];
    if (op.parent_b_.has_value()) {
      const auto* gate_input_keys_b = &wire_keys[*op.parent_b_ * num_simd];
      if (op.type_ == ENCRYPTO::PrimitiveOperationType::XOR) {
        if (parallel) {
          __gnu_parallel::transform(gate_input_keys_a, gate_input_keys_a + num_simd,
                                    gate_input_keys_b, gate_output_keys,
                                    [](const auto& ka, const auto& kb) { return ka ^ kb; });
        } else {
          std::transform(gate_input_keys_a, gate_input_keys_a + num_simd, gate_input_keys_b,
                         gate_output_keys, [](const auto& ka, const auto& kb) { return ka ^ kb; });
        }
      } else if (op.type_ == ENCRYPTO::PrimitiveOperationType::AND) {
        if (parallel) {
          batch_garble_and_omp(gate_output_keys, &garbled_tables[and_j * 2 * num_simd], start_index,
                               gate_input_keys_a, gate_input_keys_b, num_simd);
        } else {
          batch_garble_and(gate_output_keys, &garbled_tables[and_j * 2 * num_simd], start_index,
                           gate_input_keys_a, gate_input_keys_b, num_simd);
        }
        ++and_j;
        start_index += num_simd;
      } else {
        throw std::runtime_error("unsupported operation");
      }
    } else {
      if (op.type_ == ENCRYPTO::PrimitiveOperationType::INV) {
        if (parallel) {
          __gnu_parallel::transform(gate_input_keys_a, gate_input_keys_a + num_simd,
                                    gate_output_keys,
                                    [this](const auto& k) { return k ^ offset_; });
        } else {
          std::transform(gate_input_keys_a, gate_input_keys_a + num_simd, gate_output_keys,
                         [this](const auto& k) { return k ^ offset_; });
        }
      } else {
        throw std::runtime_error("unsupported operation");
      }
    }
  }
  std::copy_n(wire_keys.data() + (algo.n_wires_ - algo.n_output_wires_) * num_simd,
              algo.n_output_wires_ * num_simd, output_keys.data());
}

HalfGateEvaluator::HalfGateEvaluator(const HalfGatePublicData& public_data)
    : hash_key_(public_data.hash_key) {
  *reinterpret_cast<ENCRYPTO::block128_t*>(round_keys_.data()) = public_data.aes_key;
  aesni_key_expansion_128(round_keys_.data());
}

void HalfGateEvaluator::evaluate_and(ENCRYPTO::block128_t& key_c,
                                     const ENCRYPTO::block128_t* garbled_table, std::size_t index,
                                     const ENCRYPTO::block128_t& key_a,
                                     const ENCRYPTO::block128_t& key_b) const {
  // TODO: avoid the jumps

  // permutation bits
  bool p_a = static_cast<bool>(key_a.byte_array[0] & std::byte(1));
  bool p_b = static_cast<bool>(key_b.byte_array[0] & std::byte(1));

  std::array<ENCRYPTO::block128_t, 2> hash_inputs = {key_a, key_b};
  aesni_fixed_key_for_half_gates_batch_2(round_keys_.data(), hash_key_.data(), index,
                                         hash_inputs.data());
  key_c = hash_inputs[0] ^ hash_inputs[1];
  if (p_a) key_c ^= garbled_table[0];
  if (p_b) key_c ^= (garbled_table[1] ^ key_a);
}

void HalfGateEvaluator::batch_evaluate_and(ENCRYPTO::block128_t* key_cs,
                                           const ENCRYPTO::block128_t* garbled_tables,
                                           std::size_t start_index,
                                           const ENCRYPTO::block128_t* key_as,
                                           const ENCRYPTO::block128_t* key_bs,
                                           std::size_t num_gates) const {
  for (std::size_t i = 0; i < num_gates; ++i) {
    evaluate_and(key_cs[i], &garbled_tables[2 * i], start_index + i, key_as[i], key_bs[i]);
  }
}

void HalfGateEvaluator::batch_evaluate_and(ENCRYPTO::block128_vector& key_cs,
                                           const ENCRYPTO::block128_t* garbled_tables,
                                           std::size_t start_index,
                                           const ENCRYPTO::block128_vector& key_as,
                                           const ENCRYPTO::block128_vector& key_bs) const {
  std::size_t num_gates = key_as.size();
  assert(key_as.size() == num_gates);
  assert(key_bs.size() == num_gates);
  key_cs.resize(num_gates);
  batch_evaluate_and(key_cs.data(), garbled_tables, start_index, key_as.data(), key_bs.data(),
                     num_gates);
}

void HalfGateEvaluator::batch_evaluate_and_omp(ENCRYPTO::block128_t* key_cs,
                                               const ENCRYPTO::block128_t* garbled_tables,
                                               std::size_t start_index,
                                               const ENCRYPTO::block128_t* key_as,
                                               const ENCRYPTO::block128_t* key_bs,
                                               std::size_t num_gates) const {
#pragma omp parallel for
  for (std::size_t i = 0; i < num_gates; ++i) {
    evaluate_and(key_cs[i], &garbled_tables[2 * i], start_index + i, key_as[i], key_bs[i]);
  }
}

void HalfGateEvaluator::evaluate_circuit(
    ENCRYPTO::block128_vector& output_keys, const ENCRYPTO::block128_vector& garbled_tables,
    std::size_t start_index, const ENCRYPTO::block128_vector& input_keys_a,
    const ENCRYPTO::block128_vector& input_keys_b, std::size_t num_simd,
    const ENCRYPTO::AlgorithmDescription& algo, bool parallel) const {
  assert(input_keys_a.size() == algo.n_input_wires_parent_a_ * num_simd);
  assert((!algo.n_input_wires_parent_b_.has_value()) ||
         (input_keys_b.size() == *algo.n_input_wires_parent_b_ * num_simd));
  output_keys.resize(algo.n_output_wires_ * num_simd);
  ENCRYPTO::block128_vector wire_keys(algo.n_wires_ * num_simd);
  auto it = std::copy_n(input_keys_a.data(), input_keys_a.size(), wire_keys.data());
  if (algo.n_input_wires_parent_b_.has_value()) {
    std::copy_n(input_keys_b.data(), input_keys_b.size(), it);
  }
  assert(algo.n_gates_ == algo.gates_.size());
  for (std::size_t op_i = 0, and_j = 0; op_i < algo.n_gates_; ++op_i) {
    const auto& op = algo.gates_[op_i];
    const ENCRYPTO::block128_t* gate_input_keys_a = &wire_keys[op.parent_a_ * num_simd];
    auto* gate_output_keys = &wire_keys[op.output_wire_ * num_simd];
    if (op.parent_b_.has_value()) {
      const auto* gate_input_keys_b = &wire_keys[*op.parent_b_ * num_simd];
      if (op.type_ == ENCRYPTO::PrimitiveOperationType::XOR) {
        if (parallel) {
          __gnu_parallel::transform(gate_input_keys_a, gate_input_keys_a + num_simd,
                                    gate_input_keys_b, gate_output_keys,
                                    [](const auto& ka, const auto& kb) { return ka ^ kb; });
        } else {
          std::transform(gate_input_keys_a, gate_input_keys_a + num_simd, gate_input_keys_b,
                         gate_output_keys, [](const auto& ka, const auto& kb) { return ka ^ kb; });
        }
      } else if (op.type_ == ENCRYPTO::PrimitiveOperationType::AND) {
        if (parallel) {
          batch_evaluate_and_omp(gate_output_keys, &garbled_tables[and_j * 2 * num_simd],
                                 start_index, gate_input_keys_a, gate_input_keys_b, num_simd);
        } else {
          batch_evaluate_and(gate_output_keys, &garbled_tables[and_j * 2 * num_simd], start_index,
                             gate_input_keys_a, gate_input_keys_b, num_simd);
        }
        ++and_j;
        start_index += num_simd;
      } else {
        throw std::runtime_error("unsupported operation");
      }
    } else {
      if (op.type_ == ENCRYPTO::PrimitiveOperationType::INV) {
        std::copy_n(gate_input_keys_a, num_simd, gate_output_keys);
      } else {
        throw std::runtime_error("unsupported operation");
      }
    }
  }
  std::copy_n(wire_keys.data() + (algo.n_wires_ - algo.n_output_wires_) * num_simd,
              algo.n_output_wires_ * num_simd, output_keys.data());
}

}  // namespace MOTION::Crypto::garbling
