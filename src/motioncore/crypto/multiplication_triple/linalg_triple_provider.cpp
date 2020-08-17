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

#include "linalg_triple_provider.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <fmt/format.h>

#include "crypto/arithmetic_provider.h"
#include "crypto/oblivious_transfer/ot_flavors.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "tensor/tensor_op.h"
#include "utility/helpers.h"
#include "utility/linear_algebra.h"

namespace MOTION {

template <typename T>
std::size_t LinAlgTripleProvider::register_for_gemm_triple(const tensor::GemmOp& gemm_op) {
  assert(gemm_op.verify());

  const auto record_request = [&gemm_op](auto& count_map, auto& triple_map) -> std::size_t {
    auto [it, inserted] = count_map.try_emplace(gemm_op, 1);
    triple_map.try_emplace(gemm_op, std::vector<LinAlgTriple<T>>{});
    if (inserted) {
      return 0;
    } else {
      return (it->second)++;
    }
  };

  registration_hook(gemm_op, ENCRYPTO::bit_size_v<T>);

  if constexpr (std::is_same_v<T, std::uint8_t>) {
    return record_request(gemm_counts_8_, gemm_triples_8_);
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return record_request(gemm_counts_16_, gemm_triples_16_);
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return record_request(gemm_counts_32_, gemm_triples_32_);
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return record_request(gemm_counts_64_, gemm_triples_64_);
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    return record_request(gemm_counts_128_, gemm_triples_128_);
  }
}

template std::size_t LinAlgTripleProvider::register_for_gemm_triple<std::uint8_t>(
    const tensor::GemmOp&);
template std::size_t LinAlgTripleProvider::register_for_gemm_triple<std::uint16_t>(
    const tensor::GemmOp&);
template std::size_t LinAlgTripleProvider::register_for_gemm_triple<std::uint32_t>(
    const tensor::GemmOp&);
template std::size_t LinAlgTripleProvider::register_for_gemm_triple<std::uint64_t>(
    const tensor::GemmOp&);
template std::size_t LinAlgTripleProvider::register_for_gemm_triple<__uint128_t>(
    const tensor::GemmOp&);

template <typename T>
LinAlgTripleProvider::LinAlgTriple<T> LinAlgTripleProvider::get_gemm_triple(
    const tensor::GemmOp& gemm_op, std::size_t index) {
  assert(gemm_op.verify());
  wait_setup();

  const auto get_triple = [&gemm_op, index](auto& triple_map) -> LinAlgTriple<T> {
    try {
      auto& triple_vector = triple_map.at(gemm_op);
      return std::move(triple_vector.at(index));
    } catch (std::out_of_range& e) {
      throw std::logic_error("could not find gemm triple; did you register and run setup?");
    }
  };

  if constexpr (std::is_same_v<T, std::uint8_t>) {
    return get_triple(gemm_triples_8_);
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return get_triple(gemm_triples_16_);
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return get_triple(gemm_triples_32_);
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return get_triple(gemm_triples_64_);
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    return get_triple(gemm_triples_128_);
  }
}

template LinAlgTripleProvider::LinAlgTriple<std::uint8_t>
LinAlgTripleProvider::get_gemm_triple<std::uint8_t>(const tensor::GemmOp&, std::size_t);
template LinAlgTripleProvider::LinAlgTriple<std::uint16_t>
LinAlgTripleProvider::get_gemm_triple<std::uint16_t>(const tensor::GemmOp&, std::size_t);
template LinAlgTripleProvider::LinAlgTriple<std::uint32_t>
LinAlgTripleProvider::get_gemm_triple<std::uint32_t>(const tensor::GemmOp&, std::size_t);
template LinAlgTripleProvider::LinAlgTriple<std::uint64_t>
LinAlgTripleProvider::get_gemm_triple<std::uint64_t>(const tensor::GemmOp&, std::size_t);
template LinAlgTripleProvider::LinAlgTriple<__uint128_t>
LinAlgTripleProvider::get_gemm_triple<__uint128_t>(const tensor::GemmOp&, std::size_t);

template <typename T>
std::size_t LinAlgTripleProvider::register_for_conv2d_triple(const tensor::Conv2DOp& conv_op) {
  assert(conv_op.verify());

  const auto record_request = [&conv_op](auto& count_map, auto& triple_map) -> std::size_t {
    auto [it, inserted] = count_map.try_emplace(conv_op, 1);
    triple_map.try_emplace(conv_op, std::vector<LinAlgTriple<T>>{});
    if (inserted) {
      return 0;
    } else {
      return (it->second)++;
    }
  };

  registration_hook(conv_op, ENCRYPTO::bit_size_v<T>);

  if constexpr (std::is_same_v<T, std::uint8_t>) {
    return record_request(conv2d_counts_8_, conv2d_triples_8_);
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return record_request(conv2d_counts_16_, conv2d_triples_16_);
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return record_request(conv2d_counts_32_, conv2d_triples_32_);
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return record_request(conv2d_counts_64_, conv2d_triples_64_);
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    return record_request(conv2d_counts_128_, conv2d_triples_128_);
  }
}

template std::size_t LinAlgTripleProvider::register_for_conv2d_triple<std::uint8_t>(
    const tensor::Conv2DOp&);
template std::size_t LinAlgTripleProvider::register_for_conv2d_triple<std::uint16_t>(
    const tensor::Conv2DOp&);
template std::size_t LinAlgTripleProvider::register_for_conv2d_triple<std::uint32_t>(
    const tensor::Conv2DOp&);
template std::size_t LinAlgTripleProvider::register_for_conv2d_triple<std::uint64_t>(
    const tensor::Conv2DOp&);
template std::size_t LinAlgTripleProvider::register_for_conv2d_triple<__uint128_t>(
    const tensor::Conv2DOp&);

template <typename T>
LinAlgTripleProvider::LinAlgTriple<T> LinAlgTripleProvider::get_conv2d_triple(
    const tensor::Conv2DOp& conv_op, std::size_t index) {
  assert(conv_op.verify());
  wait_setup();

  const auto get_triple = [&conv_op, index](auto& triple_map) -> LinAlgTriple<T> {
    try {
      auto& triple_vector = triple_map.at(conv_op);
      return std::move(triple_vector.at(index));
    } catch (std::out_of_range& e) {
      throw std::logic_error("could not find conv2d triple; did you register and run setup?");
    }
  };

  if constexpr (std::is_same_v<T, std::uint8_t>) {
    return get_triple(conv2d_triples_8_);
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return get_triple(conv2d_triples_16_);
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return get_triple(conv2d_triples_32_);
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return get_triple(conv2d_triples_64_);
  } else if constexpr (std::is_same_v<T, __uint128_t>) {
    return get_triple(conv2d_triples_128_);
  }
}

template LinAlgTripleProvider::LinAlgTriple<std::uint8_t>
LinAlgTripleProvider::get_conv2d_triple<std::uint8_t>(const tensor::Conv2DOp&, std::size_t);
template LinAlgTripleProvider::LinAlgTriple<std::uint16_t>
LinAlgTripleProvider::get_conv2d_triple<std::uint16_t>(const tensor::Conv2DOp&, std::size_t);
template LinAlgTripleProvider::LinAlgTriple<std::uint32_t>
LinAlgTripleProvider::get_conv2d_triple<std::uint32_t>(const tensor::Conv2DOp&, std::size_t);
template LinAlgTripleProvider::LinAlgTriple<std::uint64_t>
LinAlgTripleProvider::get_conv2d_triple<std::uint64_t>(const tensor::Conv2DOp&, std::size_t);
template LinAlgTripleProvider::LinAlgTriple<__uint128_t>
LinAlgTripleProvider::get_conv2d_triple<__uint128_t>(const tensor::Conv2DOp&, std::size_t);

std::size_t LinAlgTripleProvider::register_for_relu_triple(std::size_t num_triples,
                                                           std::size_t bit_size) {
  std::size_t index;
  auto [it, inserted] = relu_counts_.try_emplace({num_triples, bit_size}, 1);
  relu_triples_.try_emplace({num_triples, bit_size}, std::vector<BooleanTriple>{});
  if (inserted) {
    index = 0;
  } else {
    index = (it->second)++;
  }
  registration_hook_boolean(num_triples, bit_size);
  return index;
}

LinAlgTripleProvider::BooleanTriple LinAlgTripleProvider::get_relu_triple(std::size_t num_triples,
                                                                          std::size_t bit_size,
                                                                          std::size_t index) {
  wait_setup();
  try {
    auto& triple_vector = relu_triples_.at({num_triples, bit_size});
    return std::move(triple_vector.at(index));
  } catch (std::out_of_range& e) {
    throw std::logic_error("could not find relu triple; did you register and run setup?");
  }
}

// ---------- LinAlgTriplesFromAP ----------

LinAlgTriplesFromAP::LinAlgTriplesFromAP(ArithmeticProvider& arith_provider,
                                         ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider,
                                         std::shared_ptr<Logger> logger)
    : arith_provider_(arith_provider), ot_provider_(ot_provider), logger_(logger) {}

LinAlgTriplesFromAP::~LinAlgTriplesFromAP() = default;

void LinAlgTriplesFromAP::setup() {
  const auto run_setup_gemm_1 = [](const auto& count_map, auto& handle_map, auto& triple_map) {
    for (const auto& [gemm_op, count] : count_map) {
      auto& handle_vec = handle_map.at(gemm_op);
      auto& triple_vec = triple_map.at(gemm_op);
      assert(handle_vec.size() == count);
      triple_vec.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        auto& triple = triple_vec.emplace_back();
        using T = typename decltype(triple.a_)::value_type;
        triple.a_ = Helpers::RandomVector<T>(gemm_op.compute_input_A_size());
        triple.b_ = Helpers::RandomVector<T>(gemm_op.compute_input_B_size());
        triple.c_ = matrix_multiply(gemm_op.input_A_shape_[0], gemm_op.input_A_shape_[1],
                                    gemm_op.output_shape_[1], triple.a_, triple.b_);
        assert(triple.c_.size() == gemm_op.compute_output_size());
        auto& [handle_input, handle_kernel] = handle_vec.at(i);
        handle_input->set_input(triple.a_);
        handle_kernel->set_input(triple.b_);
      }
    }
  };

  const auto run_setup_gemm_2 = [](const auto& count_map, auto& handle_map, auto& triple_map) {
    for (const auto& [gemm_op, count] : count_map) {
      auto& handle_vec = handle_map.at(gemm_op);
      auto& triple_vec = triple_map.at(gemm_op);
      assert(handle_vec.size() == count);
      assert(triple_vec.size() == count);
      for (std::size_t i = 0; i < count; ++i) {
        auto& triple = triple_vec.at(i);
        auto& [handle_input, handle_kernel] = handle_vec.at(i);
        handle_input->compute_output();
        const auto gemm1_output = handle_input->get_output();
        assert(gemm1_output.size() == gemm_op.compute_output_size());
        std::transform(std::begin(triple.c_), std::end(triple.c_), std::begin(gemm1_output),
                       std::begin(triple.c_), std::plus{});
        handle_kernel->compute_output();
        const auto gemm2_output = handle_kernel->get_output();
        assert(gemm1_output.size() == gemm_op.compute_output_size());
        std::transform(std::begin(triple.c_), std::end(triple.c_), std::begin(gemm2_output),
                       std::begin(triple.c_), std::plus{});
      }
    }
  };

  const auto run_setup_conv_1 = [](const auto& count_map, auto& handle_map, auto& triple_map) {
    for (const auto& [conv_op, count] : count_map) {
      auto& handle_vec = handle_map.at(conv_op);
      auto& triple_vec = triple_map.at(conv_op);
      assert(handle_vec.size() == count);
      assert(triple_vec.size() == 0);
      triple_vec.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        auto& triple = triple_vec.emplace_back();
        using T = typename decltype(triple.a_)::value_type;
        triple.a_ = Helpers::RandomVector<T>(conv_op.compute_input_size());
        triple.b_ = Helpers::RandomVector<T>(conv_op.compute_kernel_size());
        triple.c_ = convolution(conv_op, triple.a_, triple.b_);
        auto& [handle_input, handle_kernel] = handle_vec.at(i);
        handle_input->set_input(triple.a_);
        handle_kernel->set_input(triple.b_);
      }
    }
  };

  const auto run_setup_conv_2 = [](const auto& count_map, auto& handle_map, auto& triple_map) {
    for (const auto& [conv_op, count] : count_map) {
      auto& handle_vec = handle_map.at(conv_op);
      auto& triple_vec = triple_map.at(conv_op);
      assert(handle_vec.size() == count);
      assert(triple_vec.size() == count);
      for (std::size_t i = 0; i < count; ++i) {
        auto& triple = triple_vec.at(i);
        auto& [handle_input, handle_kernel] = handle_vec.at(i);
        handle_input->compute_output();
        const auto conv1_output = handle_input->get_output();
        std::transform(std::begin(triple.c_), std::end(triple.c_), std::begin(conv1_output),
                       std::begin(triple.c_), std::plus{});
        handle_kernel->compute_output();
        const auto conv2_output = handle_kernel->get_output();
        std::transform(std::begin(triple.c_), std::end(triple.c_), std::begin(conv2_output),
                       std::begin(triple.c_), std::plus{});
      }
    }
  };

  const auto run_setup_boolean = [](const auto& count_map, auto& handle_map, auto& triple_map) {
    for (const auto& [key, count] : count_map) {
      const auto num_triples = key.first;
      const auto bit_size = key.second;
      auto& handle_vec = handle_map.at(key);
      auto& triple_vec = triple_map.at(key);
      assert(handle_vec.size() == count);
      assert(triple_vec.size() == 0);
      triple_vec.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        auto& triple = triple_vec.emplace_back();
        auto& [handle_sender, handle_receiver] = handle_vec.at(i);
        handle_receiver->ComputeOutputs();
        auto r = handle_receiver->GetChoices();
        auto m_r = handle_receiver->GetOutputs();
        handle_sender->ComputeOutputs();
        auto m_0_and_m_1 = handle_sender->GetOutputs();
        std::vector<ENCRYPTO::BitVector<>> m_0s(bit_size - 1);
        std::vector<ENCRYPTO::BitVector<>> m_1s(bit_size - 1);
        std::vector<ENCRYPTO::BitVector<>> m_rs(bit_size - 1);

        for (auto bvs : {&m_0s, &m_1s, &m_rs}) {
          std::for_each(std::begin(*bvs), std::end(*bvs), [num_triples](auto& bv) {
            bv.Reserve(Helpers::Convert::BitsToBytes(num_triples));
          });
        }
        // probably horrible inefficient, but works ...
        for (std::size_t t_i = 0; t_i < num_triples; ++t_i) {
          for (std::size_t bit_j = 0; bit_j < bit_size - 1; ++bit_j) {
            m_0s[bit_j].Append(m_0_and_m_1.Get(2 * t_i * (bit_size - 1) + bit_j));
            m_1s[bit_j].Append(m_0_and_m_1.Get((2 * t_i + 1) * (bit_size - 1) + bit_j));
            m_rs[bit_j].Append(m_r.Get(t_i * (bit_size - 1) + bit_j));
          }
        }
        // compute triples as follows:
        // a_i = r
        // u_i = m_r = m_0 ^ r * (m_0 ^ m_1)
        // b_i = m_0' ^ m_1'
        // v_i = m_0'
        // c_i = a_i & b_i ^ u_i ^ v_i
        //     = a_i & b_i ^ m_r ^ m_0'
        triple.a_ = std::move(r);
        triple.b_ = std::move(m_1s);
        triple.c_.resize(bit_size - 1);
        for (std::size_t bit_j = 0; bit_j < bit_size - 1; ++bit_j) {
          triple.b_[bit_j] ^= m_0s[bit_j];
          triple.c_[bit_j] = triple.a_ & triple.b_[bit_j];
          triple.c_[bit_j] ^= m_rs[bit_j];
          triple.c_[bit_j] ^= m_0s[bit_j];
        }
      }
    }
  };

  run_setup_gemm_1(gemm_counts_8_, gemm_handles_8_, gemm_triples_8_);
  run_setup_gemm_1(gemm_counts_16_, gemm_handles_16_, gemm_triples_16_);
  run_setup_gemm_1(gemm_counts_32_, gemm_handles_32_, gemm_triples_32_);
  run_setup_gemm_1(gemm_counts_64_, gemm_handles_64_, gemm_triples_64_);
  run_setup_gemm_1(gemm_counts_128_, gemm_handles_128_, gemm_triples_128_);

  run_setup_conv_1(conv2d_counts_8_, conv2d_handles_8_, conv2d_triples_8_);
  run_setup_conv_1(conv2d_counts_16_, conv2d_handles_16_, conv2d_triples_16_);
  run_setup_conv_1(conv2d_counts_32_, conv2d_handles_32_, conv2d_triples_32_);
  run_setup_conv_1(conv2d_counts_64_, conv2d_handles_64_, conv2d_triples_64_);
  run_setup_conv_1(conv2d_counts_128_, conv2d_handles_128_, conv2d_triples_128_);

  run_setup_gemm_2(gemm_counts_8_, gemm_handles_8_, gemm_triples_8_);
  run_setup_gemm_2(gemm_counts_16_, gemm_handles_16_, gemm_triples_16_);
  run_setup_gemm_2(gemm_counts_32_, gemm_handles_32_, gemm_triples_32_);
  run_setup_gemm_2(gemm_counts_64_, gemm_handles_64_, gemm_triples_64_);
  run_setup_gemm_2(gemm_counts_128_, gemm_handles_128_, gemm_triples_128_);

  run_setup_conv_2(conv2d_counts_8_, conv2d_handles_8_, conv2d_triples_8_);
  run_setup_conv_2(conv2d_counts_16_, conv2d_handles_16_, conv2d_triples_16_);
  run_setup_conv_2(conv2d_counts_32_, conv2d_handles_32_, conv2d_triples_32_);
  run_setup_conv_2(conv2d_counts_64_, conv2d_handles_64_, conv2d_triples_64_);
  run_setup_conv_2(conv2d_counts_128_, conv2d_handles_128_, conv2d_triples_128_);

  run_setup_boolean(relu_counts_, relu_handles_, relu_triples_);

  set_setup_ready();
}

void LinAlgTriplesFromAP::registration_hook(const tensor::GemmOp& gemm_op, std::size_t bit_size) {
  assert(gemm_op.verify());

  const auto register_gemms = [this, &gemm_op](auto& gemm_handle_map, auto dummy_arg) {
    using T = decltype(dummy_arg);
    auto matrix_lhs = arith_provider_.register_matrix_multiplication_lhs<T>(
        gemm_op.input_A_shape_[0], gemm_op.input_A_shape_[1], gemm_op.output_shape_[1]);
    auto matrix_rhs = arith_provider_.register_matrix_multiplication_rhs<T>(
        gemm_op.input_A_shape_[0], gemm_op.input_A_shape_[1], gemm_op.output_shape_[1]);
    auto [it, inserted] = gemm_handle_map.try_emplace(gemm_op, gemm_value_type<T>{});
    auto pair = std::make_pair<std::unique_ptr<MatrixMultiplicationLHS<T>>,
                               std::unique_ptr<MatrixMultiplicationRHS<T>>>(std::move(matrix_lhs),
                                                                            std::move(matrix_rhs));
    it->second.emplace_back(std::move(pair));
  };

  switch (bit_size) {
    case 8:
      return register_gemms(gemm_handles_8_, std::uint8_t{});
    case 16:
      return register_gemms(gemm_handles_16_, std::uint16_t{});
    case 32:
      return register_gemms(gemm_handles_32_, std::uint32_t{});
    case 64:
      return register_gemms(gemm_handles_64_, std::uint64_t{});
    case 128:
      return register_gemms(gemm_handles_128_, __uint128_t{});
    default:
      throw std::logic_error("invalid bit size");
  }
}

void LinAlgTriplesFromAP::registration_hook(const tensor::Conv2DOp& conv_op, std::size_t bit_size) {
  assert(conv_op.verify());

  const auto register_convs = [this, &conv_op](auto& conv_handle_map, auto dummy_arg) {
    using T = decltype(dummy_arg);
    auto input_side = arith_provider_.register_convolution_input_side<T>(conv_op);
    auto kernel_side = arith_provider_.register_convolution_kernel_side<T>(conv_op);
    auto [it, inserted] = conv_handle_map.try_emplace(conv_op, conv_value_type<T>{});
    auto pair = std::make_pair<std::unique_ptr<ConvolutionInputSide<T>>,
                               std::unique_ptr<ConvolutionKernelSide<T>>>(std::move(input_side),
                                                                          std::move(kernel_side));
    it->second.emplace_back(std::move(pair));
  };

  switch (bit_size) {
    case 8:
      return register_convs(conv2d_handles_8_, std::uint8_t{});
    case 16:
      return register_convs(conv2d_handles_16_, std::uint16_t{});
    case 32:
      return register_convs(conv2d_handles_32_, std::uint32_t{});
    case 64:
      return register_convs(conv2d_handles_64_, std::uint64_t{});
    case 128:
      return register_convs(conv2d_handles_128_, __uint128_t{});
    default:
      throw std::logic_error("invalid bit size");
  }
}

void LinAlgTriplesFromAP::registration_hook_boolean(std::size_t num_triples, std::size_t bit_size) {
  auto rot_receiver = ot_provider_.RegisterReceiveROT(num_triples, bit_size - 1, true);
  auto rot_sender = ot_provider_.RegisterSendROT(num_triples, bit_size - 1, true);
  auto [it, inserted] =
      relu_handles_.try_emplace(std::make_pair(num_triples, bit_size), relu_value_type{});
  auto pair = std::make_pair<std::unique_ptr<ENCRYPTO::ObliviousTransfer::ROTSender>,
                             std::unique_ptr<ENCRYPTO::ObliviousTransfer::ROTReceiver>>(
      std::move(rot_sender), std::move(rot_receiver));
  it->second.emplace_back(std::move(pair));
}

// ---------- FakeLinAlgTripleProvider ----------

void FakeLinAlgTripleProvider::setup() {
  const auto run_setup_gemm = [](const auto& count_map, auto& triple_map) {
    for (const auto& [gemm_op, count] : count_map) {
      auto& triple_vec = triple_map.at(gemm_op);
      triple_vec.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        auto& triple = triple_vec.emplace_back();
        using T = typename decltype(triple.a_)::value_type;
        triple.a_ = Helpers::RandomVector<T>(gemm_op.compute_input_A_size());
        triple.b_ = Helpers::RandomVector<T>(gemm_op.compute_input_B_size());
        triple.c_ = Helpers::RandomVector<T>(gemm_op.compute_output_size());
      }
    }
  };
  const auto run_setup_conv = [](const auto& count_map, auto& triple_map) {
    for (const auto& [conv_op, count] : count_map) {
      auto& triple_vec = triple_map.at(conv_op);
      triple_vec.reserve(count);
      for (std::size_t i = 0; i < count; ++i) {
        auto& triple = triple_vec.emplace_back();
        using T = typename decltype(triple.a_)::value_type;
        triple.a_ = Helpers::RandomVector<T>(conv_op.compute_input_size());
        triple.b_ = Helpers::RandomVector<T>(conv_op.compute_kernel_size());
        triple.c_ = Helpers::RandomVector<T>(conv_op.compute_output_size());
      }
    }
  };

  run_setup_gemm(gemm_counts_8_, gemm_triples_8_);
  run_setup_gemm(gemm_counts_16_, gemm_triples_16_);
  run_setup_gemm(gemm_counts_32_, gemm_triples_32_);
  run_setup_gemm(gemm_counts_64_, gemm_triples_64_);
  run_setup_gemm(gemm_counts_128_, gemm_triples_128_);

  run_setup_conv(conv2d_counts_8_, conv2d_triples_8_);
  run_setup_conv(conv2d_counts_16_, conv2d_triples_16_);
  run_setup_conv(conv2d_counts_32_, conv2d_triples_32_);
  run_setup_conv(conv2d_counts_64_, conv2d_triples_64_);
  run_setup_conv(conv2d_counts_128_, conv2d_triples_128_);

  set_setup_ready();
}

void FakeLinAlgTripleProvider::registration_hook(const tensor::GemmOp&, std::size_t) {}

void FakeLinAlgTripleProvider::registration_hook(const tensor::Conv2DOp&, std::size_t) {}

void FakeLinAlgTripleProvider::registration_hook_boolean(std::size_t, std::size_t) {}

}  // namespace MOTION
