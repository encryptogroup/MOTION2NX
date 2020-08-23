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

#include <benchmark/benchmark.h>
#include "algorithm/circuit_loader.h"
#include "crypto/garbling/half_gates.h"

static void BM_garble_and(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  const std::size_t num_ands = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(num_ands);
  auto key_bs = ENCRYPTO::block128_vector::make_random(num_ands);
  const std::size_t index = 42;
  ENCRYPTO::block128_vector key_cs(num_ands);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands);

  for (auto _ : state) {
    garbler.batch_garble_and(key_cs, garbled_tables.data(), index, key_as, key_bs);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_ands, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands);
}
BENCHMARK(BM_garble_and)->RangeMultiplier(1 << 2)->Range(1, 1 << 20);

static void BM_evaluate_and(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  MOTION::Crypto::garbling::HalfGateEvaluator evaluator(garbler.get_public_data());
  const std::size_t num_ands = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(num_ands);
  auto key_bs = ENCRYPTO::block128_vector::make_random(num_ands);
  const std::size_t index = 42;
  ENCRYPTO::block128_vector key_cs(num_ands);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands);
  garbler.batch_garble_and(key_cs, garbled_tables.data(), index, key_as, key_bs);

  for (auto _ : state) {
    evaluator.batch_evaluate_and(key_cs, garbled_tables.data(), index, key_as, key_bs);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_ands, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands);
}
BENCHMARK(BM_evaluate_and)->RangeMultiplier(1 << 2)->Range(1, 1 << 20);

static void BM_garble_aes_128_circuit(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  MOTION::CircuitLoader circuit_loader;
  const auto& algo = circuit_loader.load_circuit("aes_128.bristol", MOTION::CircuitFormat::Bristol);
  const std::size_t size = 128;
  const std::size_t num_ands = 6400;
  const std::size_t num_simd = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(size * num_simd);
  auto key_bs = ENCRYPTO::block128_vector::make_random(size * num_simd);
  const std::size_t index = 42;
  ENCRYPTO::block128_vector key_cs(size * num_simd);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands * num_simd);

  for (auto _ : state) {
    garbler.garble_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_simd * num_ands, benchmark::Counter::kIsRate);
  state.counters["circuits_per_second"] =
      benchmark::Counter(state.iterations() * num_simd, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands * num_simd);
}
BENCHMARK(BM_garble_aes_128_circuit)->RangeMultiplier(1 << 2)->Range(1, 1 << 10);

static void BM_evaluate_aes_128_circuit(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  MOTION::Crypto::garbling::HalfGateEvaluator evaluator(garbler.get_public_data());
  MOTION::CircuitLoader circuit_loader;
  const auto& algo = circuit_loader.load_circuit("aes_128.bristol", MOTION::CircuitFormat::Bristol);
  const std::size_t size = 128;
  const std::size_t num_ands = 6400;
  const std::size_t num_simd = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(size * num_simd);
  auto key_bs = ENCRYPTO::block128_vector::make_random(size * num_simd);
  const std::size_t index = 42;
  ENCRYPTO::block128_vector key_cs(size * num_simd);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands * num_simd);
  garbler.garble_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);

  for (auto _ : state) {
    evaluator.evaluate_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_simd * num_ands, benchmark::Counter::kIsRate);
  state.counters["circuits_per_second"] =
      benchmark::Counter(state.iterations() * num_simd, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands * num_simd);
}
BENCHMARK(BM_evaluate_aes_128_circuit)->RangeMultiplier(1 << 2)->Range(1, 1 << 10);

static void BM_garble_sha_256_circuit(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  MOTION::CircuitLoader circuit_loader;
  const auto& algo =
      circuit_loader.load_circuit("sha_256.bristol", MOTION::CircuitFormat::BristolFashion);
  const std::size_t num_ands = 22573;
  const std::size_t num_simd = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(512 * num_simd);
  auto key_bs = ENCRYPTO::block128_vector::make_random(256 * num_simd);
  const std::size_t index = 42;
  ENCRYPTO::block128_vector key_cs(256 * num_simd);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands * num_simd);

  for (auto _ : state) {
    garbler.garble_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_simd * num_ands, benchmark::Counter::kIsRate);
  state.counters["circuits_per_second"] =
      benchmark::Counter(state.iterations() * num_simd, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands * num_simd);
}
BENCHMARK(BM_garble_sha_256_circuit)->RangeMultiplier(1 << 2)->Range(1, 1 << 10);

static void BM_evaluate_sha_256_circuit(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  MOTION::Crypto::garbling::HalfGateEvaluator evaluator(garbler.get_public_data());
  MOTION::CircuitLoader circuit_loader;
  const auto& algo =
      circuit_loader.load_circuit("sha_256.bristol", MOTION::CircuitFormat::BristolFashion);
  const std::size_t num_ands = 22573;
  const std::size_t num_simd = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(512 * num_simd);
  auto key_bs = ENCRYPTO::block128_vector::make_random(256 * num_simd);
  const std::size_t index = 42;
  ENCRYPTO::block128_vector key_cs(256 * num_simd);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands * num_simd);
  garbler.garble_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);

  for (auto _ : state) {
    evaluator.evaluate_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_simd * num_ands, benchmark::Counter::kIsRate);
  state.counters["circuits_per_second"] =
      benchmark::Counter(state.iterations() * num_simd, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands * num_simd);
}
BENCHMARK(BM_evaluate_sha_256_circuit)->RangeMultiplier(1 << 2)->Range(1, 1 << 10);

static void BM_garble_relu_circuit(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  MOTION::CircuitLoader circuit_loader;
  const std::size_t bit_size = 64;
  const auto& algo = circuit_loader.load_relu_circuit(bit_size);
  const std::size_t num_ands = bit_size - 1;
  const std::size_t num_simd = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(bit_size * num_simd);
  ENCRYPTO::block128_vector key_bs;
  ENCRYPTO::block128_vector key_cs(bit_size * num_simd);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands * num_simd);
  const std::size_t index = 42;

  for (auto _ : state) {
    garbler.garble_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_simd * num_ands, benchmark::Counter::kIsRate);
  state.counters["circuits_per_second"] =
      benchmark::Counter(state.iterations() * num_simd, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands * num_simd);
}
BENCHMARK(BM_garble_relu_circuit)->RangeMultiplier(1 << 2)->Range(1, 1 << 10);

static void BM_evaluate_relu_circuit(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  MOTION::Crypto::garbling::HalfGateEvaluator evaluator(garbler.get_public_data());
  MOTION::CircuitLoader circuit_loader;
  const std::size_t bit_size = 64;
  const auto& algo = circuit_loader.load_relu_circuit(bit_size);
  const std::size_t num_ands = bit_size - 1;
  const std::size_t num_simd = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(bit_size * num_simd);
  ENCRYPTO::block128_vector key_bs;
  ENCRYPTO::block128_vector key_cs(bit_size * num_simd);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands * num_simd);
  const std::size_t index = 42;
  garbler.garble_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);

  for (auto _ : state) {
    evaluator.evaluate_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_simd * num_ands, benchmark::Counter::kIsRate);
  state.counters["circuits_per_second"] =
      benchmark::Counter(state.iterations() * num_simd, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands * num_simd);
}
BENCHMARK(BM_evaluate_relu_circuit)->RangeMultiplier(1 << 2)->Range(1, 1 << 10);

static void BM_garble_max4_circuit(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  MOTION::CircuitLoader circuit_loader;
  const std::size_t bit_size = 64;
  const auto& algo = circuit_loader.load_maxpool_circuit(bit_size, 4);
  const std::size_t num_ands = (2 * bit_size) * (4 - 1);
  const std::size_t num_simd = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(4 * bit_size * num_simd);
  ENCRYPTO::block128_vector key_bs;
  ENCRYPTO::block128_vector key_cs(bit_size * num_simd);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands * num_simd);
  const std::size_t index = 42;

  for (auto _ : state) {
    garbler.garble_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_simd * num_ands, benchmark::Counter::kIsRate);
  state.counters["circuits_per_second"] =
      benchmark::Counter(state.iterations() * num_simd, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands * num_simd);
}
BENCHMARK(BM_garble_max4_circuit)->RangeMultiplier(1 << 2)->Range(1, 1 << 10);

static void BM_evaluate_max4_circuit(benchmark::State& state) {
  MOTION::Crypto::garbling::HalfGateGarbler garbler;
  MOTION::Crypto::garbling::HalfGateEvaluator evaluator(garbler.get_public_data());
  MOTION::CircuitLoader circuit_loader;
  const std::size_t bit_size = 64;
  const auto& algo = circuit_loader.load_maxpool_circuit(bit_size, 4);
  const std::size_t num_ands = (2 * bit_size) * (4 - 1);
  const std::size_t num_simd = state.range(0);
  const auto offset = garbler.get_offset();
  auto key_as = ENCRYPTO::block128_vector::make_random(4 * bit_size * num_simd);
  ENCRYPTO::block128_vector key_bs;
  ENCRYPTO::block128_vector key_cs(bit_size * num_simd);
  ENCRYPTO::block128_vector garbled_tables(2 * num_ands * num_simd);
  const std::size_t index = 42;
  garbler.garble_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);

  for (auto _ : state) {
    evaluator.evaluate_circuit(key_cs, garbled_tables, index, key_as, key_bs, num_simd, algo);
  }
  state.counters["ands_per_second"] =
      benchmark::Counter(state.iterations() * num_simd * num_ands, benchmark::Counter::kIsRate);
  state.counters["circuits_per_second"] =
      benchmark::Counter(state.iterations() * num_simd, benchmark::Counter::kIsRate);
  state.SetBytesProcessed(state.iterations() * 32 * num_ands * num_simd);
}
BENCHMARK(BM_evaluate_max4_circuit)->Arg(1)->RangeMultiplier(1 << 2)->Range(1, 1 << 10);
