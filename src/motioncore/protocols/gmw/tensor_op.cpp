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

#include "tensor_op.h"

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "algorithm/circuit_loader.h"
#include "algorithm/make_circuit.h"
#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/linalg_triple_provider.h"
#include "crypto/multiplication_triple/sb_provider.h"
#include "crypto/multiplication_triple/sp_provider.h"
#include "crypto/oblivious_transfer/ot_flavors.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "crypto/sharing_randomness_generator.h"
#include "gmw_provider.h"
#include "utility/bit_vector.h"
#include "utility/constants.h"
#include "utility/fixed_point.h"
#include "utility/linear_algebra.h"
#include "utility/logger.h"
#include "wire.h"

namespace MOTION::proto::gmw {

template <typename T>
ArithmeticGMWTensorInputSender<T>::ArithmeticGMWTensorInputSender(
    std::size_t gate_id, GMWProvider& gmw_provider, const tensor::TensorDimensions& dimensions,
    ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& input_future)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      input_id_(gmw_provider.get_next_input_id(1)),
      input_future_(std::move(input_future)),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(dimensions)) {
  if (gmw_provider_.get_num_parties() != 2) {
    throw std::logic_error("only two parties are currently supported");
  }
  output_->get_share().resize(dimensions.get_data_size());
}

template <typename T>
void ArithmeticGMWTensorInputSender<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWTensorInputSender<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto my_id = gmw_provider_.get_my_id();
  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& rng = mbp.get_my_randomness_generator(1 - my_id);
  rng.GetUnsigned<T>(input_id_, output_->get_dimensions().get_data_size(),
                     output_->get_share().data());

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorInputSender<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticGMWTensorInputSender<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWTensorInputSender<T>::evaluate_online start", gate_id_));
    }
  }

  // wait for input value
  const auto input = input_future_.get();
  if (input.size() != output_->get_dimensions().get_data_size()) {
    throw std::runtime_error("size of input vector != product of expected dimensions");
  }

  // compute my share
  auto& share = output_->get_share();
  std::transform(std::begin(input), std::end(input), std::begin(share), std::begin(share),
                 std::minus{});
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorInputSender<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWTensorInputSender<std::uint32_t>;
template class ArithmeticGMWTensorInputSender<std::uint64_t>;

template <typename T>
ArithmeticGMWTensorInputReceiver<T>::ArithmeticGMWTensorInputReceiver(
    std::size_t gate_id, GMWProvider& gmw_provider, const tensor::TensorDimensions& dimensions)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      input_id_(gmw_provider.get_next_input_id(1)),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(dimensions)) {
  output_->get_share().resize(dimensions.get_data_size());
}

template <typename T>
void ArithmeticGMWTensorInputReceiver<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWTensorInputReceiver<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto my_id = gmw_provider_.get_my_id();
  auto& mbp = gmw_provider_.get_motion_base_provider();
  auto& rng = mbp.get_their_randomness_generator(1 - my_id);
  rng.GetUnsigned<T>(input_id_, output_->get_dimensions().get_data_size(),
                     output_->get_share().data());
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWTensorInputReceiver<T>::evaluate_setup end", gate_id_));
    }
  }
}

template class ArithmeticGMWTensorInputReceiver<std::uint32_t>;
template class ArithmeticGMWTensorInputReceiver<std::uint64_t>;

template <typename T>
ArithmeticGMWTensorOutput<T>::ArithmeticGMWTensorOutput(std::size_t gate_id,
                                                        GMWProvider& gmw_provider,
                                                        ArithmeticGMWTensorCP<T> input,
                                                        std::size_t output_owner)
    : NewGate(gate_id), gmw_provider_(gmw_provider), output_owner_(output_owner), input_(input) {
  auto my_id = gmw_provider_.get_my_id();
  if (output_owner_ == my_id) {
    share_future_ = gmw_provider_.register_for_ints_message<T>(
        1 - my_id, gate_id_, input_->get_dimensions().get_data_size());
  }
}

template <typename T>
void ArithmeticGMWTensorOutput<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorOutput<T>::evaluate_online start", gate_id_));
    }
  }

  auto my_id = gmw_provider_.get_my_id();
  input_->wait_online();
  if (output_owner_ == my_id) {
    auto other_share = share_future_.get();
    assert(other_share.size() == input_->get_dimensions().get_data_size());
    std::transform(std::begin(other_share), std::end(other_share), std::begin(input_->get_share()),
                   std::begin(other_share), std::plus{});
    output_promise_.set_value(std::move(other_share));
  } else {
    gmw_provider_.send_ints_message(1 - my_id, gate_id_, input_->get_share());
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorOutput<T>::evaluate_online end", gate_id_));
    }
  }
}

template <typename T>
ENCRYPTO::ReusableFiberFuture<std::vector<T>> ArithmeticGMWTensorOutput<T>::get_output_future() {
  std::size_t my_id = gmw_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    return output_promise_.get_future();
  } else {
    throw std::logic_error("not this parties output");
  }
}

template class ArithmeticGMWTensorOutput<std::uint32_t>;
template class ArithmeticGMWTensorOutput<std::uint64_t>;

template <typename T>
ArithmeticGMWTensorFlatten<T>::ArithmeticGMWTensorFlatten(std::size_t gate_id,
                                                          GMWProvider& gmw_provider,
                                                          std::size_t axis,
                                                          const ArithmeticGMWTensorCP<T> input)
    : NewGate(gate_id), gmw_provider_(gmw_provider), input_(input) {
  const auto& input_dims = input_->get_dimensions();
  output_ = std::make_shared<ArithmeticGMWTensor<T>>(flatten(input_dims, axis));
}

template <typename T>
void ArithmeticGMWTensorFlatten<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorFlatten<T>::evaluate_online start", gate_id_));
    }
  }

  input_->wait_online();
  output_->get_share() = input_->get_share();
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorFlatten<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWTensorFlatten<std::uint32_t>;
template class ArithmeticGMWTensorFlatten<std::uint64_t>;

template <typename T>
ArithmeticGMWTensorConv2D<T>::ArithmeticGMWTensorConv2D(
    std::size_t gate_id, GMWProvider& gmw_provider, tensor::Conv2DOp conv_op,
    const ArithmeticGMWTensorCP<T> input, const ArithmeticGMWTensorCP<T> kernel,
    const ArithmeticGMWTensorCP<T> bias, std::size_t fractional_bits)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      conv_op_(conv_op),
      fractional_bits_(fractional_bits),
      input_(input),
      kernel_(kernel),
      bias_(bias),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(conv_op.get_output_tensor_dims())),
      triple_index_(
          gmw_provider.get_linalg_triple_provider().register_for_conv2d_triple<T>(conv_op)),
      share_future_(gmw_provider_.register_for_ints_message<T>(
          1 - gmw_provider.get_my_id(), gate_id_,
          conv_op.compute_input_size() + conv_op.compute_kernel_size())) {}

template <typename T>
void ArithmeticGMWTensorConv2D<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorConv2D<T>::evaluate_online start", gate_id_));
    }
  }

  auto& ltp = gmw_provider_.get_linalg_triple_provider();
  auto triple = ltp.get_conv2d_triple<T>(conv_op_, triple_index_);
  input_->wait_online();
  kernel_->wait_online();
  const auto& input_buffer = input_->get_share();
  const auto& kernel_buffer = kernel_->get_share();
  const auto input_size = conv_op_.compute_input_size();
  const auto kernel_size = conv_op_.compute_kernel_size();
  assert(input_buffer.size() == input_size);
  assert(kernel_buffer.size() == kernel_size);

  const auto my_id = gmw_provider_.get_my_id();

  //  mask inputs
  std::vector<T> de(input_size + kernel_size);
  auto it = std::transform(std::begin(input_buffer), std::end(input_buffer), std::begin(triple.a_),
                           std::begin(de), std::minus{});
  std::transform(std::begin(kernel_buffer), std::end(kernel_buffer), std::begin(triple.b_), it,
                 std::minus{});
  gmw_provider_.send_ints_message(1 - my_id, gate_id_, de);

  // compute d, e
  auto other_share = share_future_.get();
  std::transform(std::begin(de), std::end(de), std::begin(other_share), std::begin(de),
                 std::plus{});

  // result = c ...
  std::vector<T> result(std::move(triple.c_));
  std::vector<T> tmp(result.size());
  // ... - d * e ...
  if (gmw_provider_.is_my_job(gate_id_)) {
    convolution(conv_op_, de.data(), de.data() + input_size, tmp.data());
    std::transform(std::begin(result), std::end(result), std::begin(tmp), std::begin(result),
                   std::minus{});
  }
  // ... + e * x + d * y
  convolution(conv_op_, input_buffer.data(), de.data() + input_size, tmp.data());
  std::transform(std::begin(result), std::end(result), std::begin(tmp), std::begin(result),
                 std::plus{});
  convolution(conv_op_, de.data(), kernel_buffer.data(), tmp.data());
  std::transform(std::begin(result), std::end(result), std::begin(tmp), std::begin(result),
                 std::plus{});
  if (fractional_bits_ > 0) {
    fixed_point::truncate_shared<T>(result.data(), fractional_bits_, result.size(),
                                    gmw_provider_.is_my_job(gate_id_));
  }
  output_->get_share() = std::move(result);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorConv2D<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWTensorConv2D<std::uint32_t>;
template class ArithmeticGMWTensorConv2D<std::uint64_t>;

template <typename T>
ArithmeticGMWTensorGemm<T>::ArithmeticGMWTensorGemm(std::size_t gate_id, GMWProvider& gmw_provider,
                                                    tensor::GemmOp gemm_op,
                                                    const ArithmeticGMWTensorCP<T> input_A,
                                                    const ArithmeticGMWTensorCP<T> input_B,
                                                    std::size_t fractional_bits)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      gemm_op_(gemm_op),
      fractional_bits_(fractional_bits),
      input_A_(input_A),
      input_B_(input_B),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(gemm_op.get_output_tensor_dims())),
      triple_index_(gmw_provider.get_linalg_triple_provider().register_for_gemm_triple<T>(gemm_op)),
      share_future_(gmw_provider_.register_for_ints_message<T>(
          1 - gmw_provider.get_my_id(), gate_id_,
          gemm_op.compute_input_A_size() + gemm_op.compute_input_B_size())) {
  assert(input_A_->get_dimensions() == gemm_op.get_input_A_tensor_dims());
  assert(input_B_->get_dimensions() == gemm_op.get_input_B_tensor_dims());
  assert(gemm_op.verify());
}

template <typename T>
void ArithmeticGMWTensorGemm<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorGemm<T>::evaluate_online start", gate_id_));
    }
  }

  auto& ltp = gmw_provider_.get_linalg_triple_provider();
  auto triple = ltp.get_gemm_triple<T>(gemm_op_, triple_index_);
  input_A_->wait_online();
  input_B_->wait_online();
  const auto& input_A_buffer = input_A_->get_share();
  const auto& input_B_buffer = input_B_->get_share();
  const auto input_A_size = gemm_op_.compute_input_A_size();
  const auto input_B_size = gemm_op_.compute_input_B_size();
  assert(input_A_buffer.size() == input_A_size);
  assert(input_B_buffer.size() == input_B_size);

  const auto my_id = gmw_provider_.get_my_id();

  //  mask inputs
  std::vector<T> de(input_A_size + input_B_size);
  auto it = std::transform(std::begin(input_A_buffer), std::end(input_A_buffer),
                           std::begin(triple.a_), std::begin(de), std::minus{});
  std::transform(std::begin(input_B_buffer), std::end(input_B_buffer), std::begin(triple.b_), it,
                 std::minus{});
  gmw_provider_.send_ints_message(1 - my_id, gate_id_, de);

  // compute d, e
  auto other_share = share_future_.get();
  std::transform(std::begin(de), std::end(de), std::begin(other_share), std::begin(de),
                 std::plus{});

  // result = c ...
  std::vector<T> result(std::move(triple.c_));
  std::vector<T> tmp(result.size());

  // ... - d * e ...
  if (gmw_provider_.is_my_job(gate_id_)) {
    matrix_multiply(gemm_op_, de.data(), de.data() + input_A_size, tmp.data());
    std::transform(std::begin(result), std::end(result), std::begin(tmp), std::begin(result),
                   std::minus{});
  }
  // ... + e * x + d * y
  matrix_multiply(gemm_op_, input_A_buffer.data(), de.data() + input_A_size, tmp.data());
  std::transform(std::begin(result), std::end(result), std::begin(tmp), std::begin(result),
                 std::plus{});
  matrix_multiply(gemm_op_, de.data(), input_B_buffer.data(), tmp.data());
  std::transform(std::begin(result), std::end(result), std::begin(tmp), std::begin(result),
                 std::plus{});
  if (fractional_bits_ > 0) {
    fixed_point::truncate_shared<T>(result.data(), fractional_bits_, result.size(),
                                    gmw_provider_.is_my_job(gate_id_));
  }
  output_->get_share() = std::move(result);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorGemm<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWTensorGemm<std::uint32_t>;
template class ArithmeticGMWTensorGemm<std::uint64_t>;

template <typename T>
ArithmeticGMWTensorSqr<T>::ArithmeticGMWTensorSqr(std::size_t gate_id, GMWProvider& gmw_provider,
                                                  const ArithmeticGMWTensorCP<T> input,
                                                  std::size_t fractional_bits)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      data_size_(input->get_dimensions().get_data_size()),
      fractional_bits_(fractional_bits),
      input_(input),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(input_->get_dimensions())),
      triple_index_(gmw_provider.get_sp_provider().RequestSPs<T>(data_size_)),
      share_future_(gmw_provider_.register_for_ints_message<T>(1 - gmw_provider.get_my_id(),
                                                               gate_id_, data_size_)) {}

template <typename T>
void ArithmeticGMWTensorSqr<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorSqr<T>::evaluate_online start", gate_id_));
    }
  }

  auto& spp = gmw_provider_.get_sp_provider();
  const auto& all_triples = spp.GetSPsAll<T>();

  input_->wait_online();
  const auto& input_buffer = input_->get_share();
  assert(input_buffer.size() == data_size_);

  const auto my_id = gmw_provider_.get_my_id();

  //  mask inputs
  std::vector<T> d(data_size_);
  std::transform(std::begin(input_buffer), std::end(input_buffer), &all_triples.a[triple_index_],
                 std::begin(d), std::minus{});
  gmw_provider_.send_ints_message(1 - my_id, gate_id_, d);

  // compute d
  auto other_share = share_future_.get();
  std::transform(std::begin(d), std::end(d), std::begin(other_share), std::begin(d), std::plus{});

  std::vector<T> result(data_size_);

  // result = 2 * d * x
  std::transform(std::begin(d), std::end(d), std::begin(input_buffer), std::begin(result),
                 [](auto d, auto x) { return 2 * d * x; });

  // ... + c ...
  std::transform(std::begin(result), std::end(result), &all_triples.c[triple_index_],
                 std::begin(result), std::plus{});

  // ... - d^2
  if (gmw_provider_.is_my_job(gate_id_)) {
    std::transform(std::begin(result), std::end(result), std::begin(d), std::begin(result),
                   [](auto res, auto d) { return res - d * d; });
  }
  if (fractional_bits_ > 0) {
    fixed_point::truncate_shared<T>(result.data(), fractional_bits_, result.size(),
                                    gmw_provider_.is_my_job(gate_id_));
  }
  output_->get_share() = std::move(result);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticGMWTensorSqr<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWTensorSqr<std::uint32_t>;
template class ArithmeticGMWTensorSqr<std::uint64_t>;

template <typename T>
BooleanToArithmeticGMWTensorConversion<T>::BooleanToArithmeticGMWTensorConversion(
    std::size_t gate_id, GMWProvider& gmw_provider, const BooleanGMWTensorCP input)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(std::move(input)),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(input_->get_dimensions())) {
  const auto my_id = gmw_provider_.get_my_id();
  auto& sb_provider = gmw_provider_.get_sb_provider();
  sb_offset_ = sb_provider.RequestSBs<T>(bit_size_ * data_size_);
  t_share_future_ =
      gmw_provider_.register_for_bits_message(1 - my_id, gate_id_, bit_size_ * data_size_);
  output_->get_share().resize(data_size_);
}

template <typename T>
void BooleanToArithmeticGMWTensorConversion<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanToArithmeticGMWTensorConversion<T>::evaluate_online start", gate_id_));
    }
  }

  // let `sbs` point to our SBs
  const auto& all_sbs = gmw_provider_.get_sb_provider().GetSBsAll<T>();
  const auto* sbs = &all_sbs[sb_offset_];

  ENCRYPTO::BitVector<> t;
  t.Reserve(Helpers::Convert::BitsToBytes(bit_size_ * data_size_));

  // collect all shares into a single buffer
  input_->wait_online();
  const auto& input_share = input_->get_share();
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    const auto& share = input_share[bit_j];
    t.Append(share);
  }

  // indexing function
  const auto idx = [this](auto bit_j, auto int_i) { return bit_j * data_size_ + int_i; };

  // mask them with the shared bits
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      auto x = t.Get(idx(bit_j, int_i));
      auto r = bool(sbs[idx(bit_j, int_i)] & 1);
      t.Set(x ^ r, idx(bit_j, int_i));
    }
  }

  // reconstruct masked values
  gmw_provider_.broadcast_bits_message(gate_id_, t);
  t ^= t_share_future_.get();

  const auto is_my_job = gmw_provider_.is_my_job(gate_id_);

  // remove mask in arithmetic sharing
  auto& output = output_->get_share();
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      auto t_ij = T(t.Get(idx(bit_j, int_i)));
      auto r_ij = sbs[idx(bit_j, int_i)];
      T value = r_ij - 2 * t_ij * r_ij;
      if (is_my_job) {
        value += t_ij;
      }
      output[int_i] += value << bit_j;
    }
  }

  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanToArithmeticGMWTensorConversion<T>::evaluate_online end", gate_id_));
    }
  }
}

template class BooleanToArithmeticGMWTensorConversion<std::uint32_t>;
template class BooleanToArithmeticGMWTensorConversion<std::uint64_t>;

BooleanGMWTensorRelu::BooleanGMWTensorRelu(std::size_t gate_id, GMWProvider& gmw_provider,
                                           const BooleanGMWTensorCP input)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(std::move(input)),
      output_(std::make_shared<BooleanGMWTensor>(input_->get_dimensions(), bit_size_)),
      triple_index_(gmw_provider_.get_linalg_triple_provider().register_for_relu_triple(
          data_size_, bit_size_)) {
  const auto my_id = gmw_provider_.get_my_id();
  share_future_ =
      gmw_provider_.register_for_bits_message(1 - my_id, gate_id_, data_size_ * bit_size_);
}

void BooleanGMWTensorRelu::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWTensorRelu::evaluate_online start", gate_id_));
    }
  }

  auto triple = gmw_provider_.get_linalg_triple_provider().get_relu_triple(data_size_, bit_size_,
                                                                           triple_index_);
  assert(triple.a_.GetSize() == data_size_);
  assert(triple.b_.size() == bit_size_ - 1);
  assert(triple.c_.size() == bit_size_ - 1);
  assert(std::all_of(std::begin(triple.b_), std::end(triple.b_),
                     [this](const auto& bv) { return bv.GetSize() == data_size_; }));
  assert(std::all_of(std::begin(triple.c_), std::end(triple.c_),
                     [this](const auto& bv) { return bv.GetSize() == data_size_; }));

  input_->wait_online();
  const auto& input_share = input_->get_share();
  assert(input_share.size() == bit_size_);
  assert(std::all_of(std::begin(input_share), std::end(input_share),
                     [this](const auto& bv) { return bv.GetSize() == data_size_; }));

  // we compute a scalar x vector product of the inverted msb with the remaining bits
  // using Beaver multiplication

  // get the inverted msb
  auto inv_msb_share = input_share[bit_size_ - 1];
  if (gmw_provider_.is_my_job(gate_id_)) {
    inv_msb_share.Invert();
  }

  // masked x, y: d || e == x ^ a || y ^ b
  ENCRYPTO::BitVector<> de;
  de.Reserve(Helpers::Convert::BitsToBytes(bit_size_ * data_size_));

  // mask with x, y with a, b to get d, e
  for (std::size_t bit_j = 0; bit_j < bit_size_ - 1; ++bit_j) {
    de.Append(input_share[bit_j] ^ triple.b_[bit_j]);
  }
  de.Append(inv_msb_share ^ triple.a_);

  // reconstruct d, e
  const auto my_id = gmw_provider_.get_my_id();
  gmw_provider_.send_bits_message(1 - my_id, gate_id_, de);
  de ^= share_future_.get();
  assert(de.GetSize() == bit_size_ * data_size_);

  auto& output_share = output_->get_share();
  assert(output_share.size() == bit_size_);

  // helper function to get the ith subvector of size data_size_ from de
  const auto get_de_subset = [this, de](auto i) {
    return de.Subset(i * data_size_, (i + 1) * data_size_);
  };

  auto masked_inv_msb = get_de_subset(bit_size_ - 1);
  for (std::size_t bit_j = 0; bit_j < bit_size_ - 1; ++bit_j) {
    output_share[bit_j] = triple.c_[bit_j];
    output_share[bit_j] ^= inv_msb_share & get_de_subset(bit_j);
    output_share[bit_j] ^= masked_inv_msb & input_share[bit_j];
  }
  if (gmw_provider_.is_my_job(gate_id_)) {
    for (std::size_t bit_j = 0; bit_j < bit_size_ - 1; ++bit_j) {
      output_share[bit_j] ^= masked_inv_msb & get_de_subset(bit_j);
    }
  }
  // set the share of the msb to 0 since the plain value is always 0
  output_share[bit_size_ - 1] = ENCRYPTO::BitVector<>(data_size_, false);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: BooleanGMWTensorRelu::evaluate_online end", gate_id_));
    }
  }
}

template <typename T>
BooleanXArithmeticGMWTensorRelu<T>::BooleanXArithmeticGMWTensorRelu(
    std::size_t gate_id, GMWProvider& gmw_provider, const BooleanGMWTensorCP input_bool,
    const ArithmeticGMWTensorCP<T> input_arith)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      data_size_(input_bool->get_dimensions().get_data_size()),
      input_bool_(std::move(input_bool)),
      input_arith_(std::move(input_arith)),
      output_(std::make_shared<ArithmeticGMWTensor<T>>(input_arith_->get_dimensions())) {
  if (input_bool_->get_dimensions() != input_arith_->get_dimensions()) {
    throw std::invalid_argument("dimension mismatch");
  }
  if (input_bool_->get_bit_size() != input_arith_->get_bit_size()) {
    throw std::invalid_argument("bit size mismatch");
  }
  const auto my_id = gmw_provider_.get_my_id();
  auto& otp = gmw_provider_.get_ot_manager().get_provider(1 - my_id);
  ot_sender_ = otp.RegisterSendACOT<T>(data_size_);
  ot_receiver_ = otp.RegisterReceiveACOT<T>(data_size_);
}

template <typename T>
BooleanXArithmeticGMWTensorRelu<T>::~BooleanXArithmeticGMWTensorRelu() = default;

template <typename T>
void BooleanXArithmeticGMWTensorRelu<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanXArithmeticGMWTensorRelu::evaluate_online start", gate_id_));
    }
  }

  input_bool_->wait_online();
  input_arith_->wait_online();
  const auto& ashare = input_arith_->get_share();
  auto inv_msb_share = input_bool_->get_share()[bit_size_ - 1];
  auto& out_share = output_->get_share();

  if (gmw_provider_.is_my_job(gate_id_)) {
    inv_msb_share.Invert();
  }

  ot_receiver_->SetChoices(inv_msb_share);
  ot_receiver_->SendCorrections();
  {
    std::vector<T> ot_inputs(data_size_);
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      if (inv_msb_share.Get(int_i)) {
        ot_inputs[int_i] = -ashare[int_i];
      } else {
        ot_inputs[int_i] = ashare[int_i];
      }
    }
    ot_sender_->SetCorrelations(std::move(ot_inputs));
  }
  ot_sender_->SendMessages();

  ot_receiver_->ComputeOutputs();
  ot_sender_->ComputeOutputs();
  out_share = ot_receiver_->GetOutputs();
  const auto sender_outputs = ot_sender_->GetOutputs();
  for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
    out_share[int_i] -= sender_outputs[int_i];
    if (inv_msb_share.Get(int_i)) {
      out_share[int_i] += ashare[int_i];
    }
  }
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanXArithmeticGMWTensorRelu::evaluate_online end", gate_id_));
    }
  }
}

template class BooleanXArithmeticGMWTensorRelu<std::uint32_t>;
template class BooleanXArithmeticGMWTensorRelu<std::uint64_t>;

BooleanGMWTensorMaxPool::BooleanGMWTensorMaxPool(std::size_t gate_id, GMWProvider& gmw_provider,
                                                 tensor::MaxPoolOp maxpool_op,
                                                 const BooleanGMWTensorCP input)
    : NewGate(gate_id),
      gmw_provider_(gmw_provider),
      maxpool_op_(maxpool_op),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<BooleanGMWTensor>(maxpool_op_.get_output_tensor_dims(), bit_size_)),
      // XXX: use depth-optimized circuit here
      maxpool_algo_(gmw_provider_.get_circuit_loader().load_maxpool_circuit(
          bit_size_, maxpool_op_.compute_kernel_size(), true)) {
  if (!maxpool_op_.verify()) {
    throw std::invalid_argument("invalid MaxPoolOp");
  }
  const auto kernel_size = maxpool_op_.compute_kernel_size();
  const auto output_size = maxpool_op_.compute_output_size();
  input_wires_.resize(bit_size_ * kernel_size);
  std::generate(std::begin(input_wires_), std::end(input_wires_), [output_size] {
    auto w = std::make_shared<BooleanGMWWire>(output_size);
    w->get_share().Resize(output_size);
    return w;
  });
  {
    WireVector in(bit_size_ * kernel_size);
    std::transform(std::begin(input_wires_), std::end(input_wires_), std::begin(in),
                   [](auto w) { return std::dynamic_pointer_cast<BooleanGMWWire>(w); });
    auto [gates, out] = construct_circuit(gmw_provider_, maxpool_algo_, in);
    gates_ = std::move(gates);
    assert(out.size() == bit_size_);
    output_wires_.resize(bit_size_);
    std::transform(std::begin(out), std::end(out), std::begin(output_wires_),
                   [](auto w) { return std::dynamic_pointer_cast<BooleanGMWWire>(w); });
  }
}

static void prepare_wires(std::size_t bit_size, const tensor::MaxPoolOp& maxpool_op,
                          BooleanGMWWireVector& circuit_wires,
                          const std::vector<ENCRYPTO::BitVector<>>& input_shares) {
  const auto& input_shape = maxpool_op.input_shape_;
  const auto& output_shape = maxpool_op.output_shape_;
  const auto& kernel_shape = maxpool_op.kernel_shape_;
  const auto& strides = maxpool_op.strides_;

  // compute the index in the (tensor) input shares
  const auto in_idx = [input_shape](auto channel, auto row, auto column) {
    assert(channel < input_shape[0]);
    assert(row < input_shape[1]);
    assert(column < input_shape[2]);
    return channel * (input_shape[1] * input_shape[2]) + row * input_shape[2] + column;
  };

  // compute the index of the input wire of the circuit
  const auto mpin_wires_idx = [bit_size, &kernel_shape](auto bit_j, auto k_row, auto k_column) {
    assert(bit_j < bit_size);
    assert(k_row < kernel_shape[0]);
    assert(k_column < kernel_shape[1]);
    return (k_row * kernel_shape[1] + k_column) * bit_size + bit_j;
  };

  // compute the index in the output shares and circuit input shares
  const auto out_idx = [&output_shape](auto channel, auto row, auto column) {
    assert(channel < output_shape[0]);
    assert(row < output_shape[1]);
    assert(column < output_shape[2]);
    return channel * (output_shape[1] * output_shape[2]) + row * output_shape[2] + column;
  };

  for (std::size_t bit_j = 0; bit_j < bit_size; ++bit_j) {
    const auto& in_share = input_shares[bit_j];
    for (std::size_t channel_i = 0; channel_i < output_shape[0]; ++channel_i) {
      std::size_t i_row = 0;
      for (std::size_t o_row = 0; o_row < output_shape[1]; ++o_row) {
        std::size_t i_col = 0;
        for (std::size_t o_col = 0; o_col < output_shape[2]; ++o_col) {
          for (std::size_t k_row = 0; k_row < kernel_shape[0]; ++k_row) {
            for (std::size_t k_col = 0; k_col < kernel_shape[0]; ++k_col) {
              auto bit = in_share.Get(in_idx(channel_i, i_row + k_row, i_col + k_col));
              auto& bv = circuit_wires[mpin_wires_idx(bit_j, k_row, k_col)]->get_share();
              bv.Set(bit, out_idx(channel_i, o_row, o_col));
            }
          }

          i_col += strides[1];
        }
        i_row += strides[0];
      }
    }
  }
  for (auto& wire : circuit_wires) {
    wire->set_online_ready();
  }
}

void BooleanGMWTensorMaxPool::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWTensorMaxPool::evaluate_online start", gate_id_));
    }
  }

  input_->wait_online();

  prepare_wires(bit_size_, maxpool_op_, input_wires_, input_->get_share());

  for (auto& gate : gates_) {
    // should work since its a Boolean circuit consisting of AND, XOR, INV gates
    gate->evaluate_online();
  }

  auto& output_shares = output_->get_share();
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    auto& wire = output_wires_[bit_j];
    wire->wait_online();
    output_shares[bit_j] = std::move(wire->get_share());
  }
  output_->set_online_ready();

  std::cerr << "complete\n";

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanGMWTensorMaxPool::evaluate_online end", gate_id_));
    }
  }
}

}  // namespace MOTION::proto::gmw
