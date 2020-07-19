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

#include "triple_generator.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include <fmt/format.h>

#include "crypto/arithmetic_provider.h"
#include "ot_backend.h"
#include "utility/helpers.h"
#include "utility/linear_algebra.h"
#include "utility/type_traits.hpp"

namespace MOTION {

template <typename T>
void matrix_multiplication_stats(std::size_t m, std::size_t k, std::size_t n) {
  const auto bit_size = ENCRYPTO::bit_size_v<T>;
  const auto num_mults = m * k;
  const auto vector_size = n;
  const auto num_ots = bit_size * num_mults;
  std::cout << fmt::format("multiplication of {0}x{1} and {1}x{2} matrices:\n", m, k, n);
  std::cout << fmt::format("- bit size: {}\n", bit_size);
  std::cout << fmt::format("- {} scalar x vector multiplications with vectors of size {}\n",
                           num_mults, vector_size);
  std::cout << fmt::format("- {} additively correlated OTs\n", num_ots);
  std::cout << fmt::format("- buffer size: {:.3f} GiB\n", num_ots * vector_size * sizeof(T) / double(1 << 30));
  // row-wise:
  std::cout << fmt::format("row-wise: {}x the following:\n", m);
  std::cout << fmt::format("- {} scalar x vector multiplications with vectors of size {}\n",
                           k, vector_size);
  std::cout << fmt::format("- {} additively correlated OTs\n", bit_size * k);
  std::cout << fmt::format("- buffer size: {:.3f} GiB\n", bit_size * k * vector_size * sizeof(T) / double(1 << 30));
}

template <typename T>
std::vector<T> matrix_multiplication_lhs(OTBackend& ot_backend, std::size_t m, std::size_t k,
                                         std::size_t n, const std::vector<T>& input_a) {
  assert(input_a.size() == m * k);
  auto& arith_provider = ot_backend.get_arithmetic_provider();
  std::vector<T> output(m * n);
  // compute matrix product row-wise
  auto mult_lhs = arith_provider.register_matrix_multiplication_lhs<T>(1, k, n);
  for (std::size_t row_i = 0; row_i < m; ++row_i) {
    std::cout << fmt::format("computing row {} of {}\n", row_i + 1, m);
    // run OT setup
    ot_backend.run_setup();
    ot_backend.sync();
    const auto* input_row = input_a.data() + row_i * k;
    auto* output_row = output.data() + row_i * n;
    mult_lhs->set_input(input_row);
    mult_lhs->compute_output();
    const auto output_vector = mult_lhs->get_output();
    std::copy_n(std::begin(output_vector), n, output_row);
    ot_backend.clear();
    ot_backend.sync();
    mult_lhs->clear();
  }
  return output;
}

template <typename T>
std::vector<T> matrix_multiplication_rhs(OTBackend& ot_backend, std::size_t m, std::size_t k,
                                         std::size_t n, const std::vector<T>& input_b) {
  assert(input_b.size() == k * n);
  auto& arith_provider = ot_backend.get_arithmetic_provider();
  std::vector<T> output(m * n);
  // compute matrix product row-wise
  auto mult_rhs = arith_provider.register_matrix_multiplication_rhs<T>(1, k, n);
  for (std::size_t row_i = 0; row_i < m; ++row_i) {
    std::cout << fmt::format("computing row {} of {}\n", row_i + 1, m);
    // run OT setup
    ot_backend.run_setup();
    ot_backend.sync();
    auto* output_row = output.data() + row_i * n;
    mult_rhs->set_input(input_b.data());
    mult_rhs->compute_output();
    const auto output_vector = mult_rhs->get_output();
    std::copy_n(std::begin(output_vector), n, output_row);
    ot_backend.clear();
    ot_backend.sync();
    mult_rhs->clear();
  }
  return output;
}

template <typename T>
void generate_matrix_triple(OTBackend& ot_backend, std::size_t m, std::size_t k, std::size_t n) {
  const auto my_id = ot_backend.get_my_id();
  auto input_a = Helpers::RandomVector<T>(m * k);
  auto input_b = Helpers::RandomVector<T>(k * n);
  auto output = matrix_multiply(m, k, n, input_a, input_b);

  const auto add_to = [](const auto& xs, auto& ys) {
    assert(xs.size() == ys.size());
    std::transform(std::begin(xs), std::end(xs), std::begin(ys), std::begin(ys), std::plus{});
  };

  if (my_id == 0) {
    auto tmp = matrix_multiplication_lhs(ot_backend, m, k, n, input_a);
    add_to(tmp, output);
    tmp = matrix_multiplication_rhs(ot_backend, m, k, n, input_b);
    add_to(tmp, output);
  } else {
    auto tmp = matrix_multiplication_rhs(ot_backend, m, k, n, input_b);
    add_to(tmp, output);
    tmp = matrix_multiplication_lhs(ot_backend, m, k, n, input_a);
    add_to(tmp, output);
  }
}

void generate_triples(OTBackend& ot_backend, std::size_t m, std::size_t k, std::size_t n) {
  using T = std::uint64_t;;
  matrix_multiplication_stats<T>(m, k, n);
  generate_matrix_triple<T>(ot_backend, m, k, n);
}

}  // namespace MOTION
