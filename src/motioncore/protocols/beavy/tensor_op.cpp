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

#include <parallel/algorithm>
#include <stdexcept>

#include "algorithm/circuit_loader.h"
#include "algorithm/make_circuit.h"
#include "beavy_provider.h"
#include "crypto/arithmetic_provider.h"
#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/linalg_triple_provider.h"
#include "crypto/multiplication_triple/sp_provider.h"
#include "crypto/oblivious_transfer/ot_flavors.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "crypto/sharing_randomness_generator.h"
#include "executor/execution_context.h"
#include "utility/constants.h"
#include "utility/fiber_thread_pool/fiber_thread_pool.hpp"
#include "utility/fixed_point.h"
#include "utility/helpers.h"
#include "utility/linear_algebra.h"
#include "utility/logger.h"
#include "wire.h"

namespace MOTION::proto::beavy {

template <typename T>
ArithmeticBEAVYTensorInputSender<T>::ArithmeticBEAVYTensorInputSender(
    std::size_t gate_id, BEAVYProvider& beavy_provider, const tensor::TensorDimensions& dimensions,
    ENCRYPTO::ReusableFiberFuture<std::vector<T>>&& input_future)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      dimensions_(dimensions),
      input_id_(beavy_provider.get_next_input_id(1)),
      input_future_(std::move(input_future)),
      output_(std::make_shared<ArithmeticBEAVYTensor<T>>(dimensions)) {
  if (beavy_provider_.get_num_parties() != 2) {
    throw std::logic_error("only two parties are currently supported");
  }
  output_->get_public_share().resize(dimensions.get_data_size());
  output_->get_secret_share().resize(dimensions.get_data_size());

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorInputSender<T> created", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorInputSender<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYTensorInputSender<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto my_id = beavy_provider_.get_my_id();
  const auto data_size = dimensions_.get_data_size();
  auto& my_secret_share = output_->get_secret_share();
  auto& my_public_share = output_->get_public_share();
  my_secret_share = Helpers::RandomVector<T>(data_size);
  output_->set_setup_ready();
  auto& mbp = beavy_provider_.get_motion_base_provider();
  auto& rng = mbp.get_my_randomness_generator(1 - my_id);
  rng.GetUnsigned<T>(input_id_, data_size, my_public_share.data());
  __gnu_parallel::transform(std::begin(my_public_share), std::end(my_public_share),
                            std::begin(my_secret_share), std::begin(my_public_share), std::plus{});

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYTensorInputSender<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorInputSender<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYTensorInputSender<T>::evaluate_online start", gate_id_));
    }
  }

  // wait for input value
  const auto input = input_future_.get();
  if (input.size() != output_->get_dimensions().get_data_size()) {
    throw std::runtime_error("size of input vector != product of expected dimensions");
  }

  // compute public share
  auto& my_public_share = output_->get_public_share();
  __gnu_parallel::transform(std::begin(input), std::end(input), std::begin(my_public_share),
                            std::begin(my_public_share), std::plus{});
  output_->set_online_ready();
  beavy_provider_.broadcast_ints_message(gate_id_, my_public_share);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYTensorInputSender<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYTensorInputSender<std::uint32_t>;
template class ArithmeticBEAVYTensorInputSender<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorInputReceiver<T>::ArithmeticBEAVYTensorInputReceiver(
    std::size_t gate_id, BEAVYProvider& beavy_provider, const tensor::TensorDimensions& dimensions)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      dimensions_(dimensions),
      input_id_(beavy_provider.get_next_input_id(1)),
      output_(std::make_shared<ArithmeticBEAVYTensor<T>>(dimensions)) {
  const auto my_id = beavy_provider_.get_my_id();
  public_share_future_ =
      beavy_provider_.register_for_ints_message<T>(1 - my_id, gate_id_, dimensions.get_data_size());
  output_->get_secret_share().resize(dimensions.get_data_size());

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorInputReceiver<T> created", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorInputReceiver<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYTensorInputReceiver<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto my_id = beavy_provider_.get_my_id();
  auto& mbp = beavy_provider_.get_motion_base_provider();
  auto& rng = mbp.get_their_randomness_generator(1 - my_id);
  rng.GetUnsigned<T>(input_id_, output_->get_dimensions().get_data_size(),
                     output_->get_secret_share().data());
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYTensorInputReceiver<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorInputReceiver<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYTensorInputReceiver<T>::evaluate_online start", gate_id_));
    }
  }

  output_->get_public_share() = public_share_future_.get();
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYTensorInputReceiver<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYTensorInputReceiver<std::uint32_t>;
template class ArithmeticBEAVYTensorInputReceiver<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorOutput<T>::ArithmeticBEAVYTensorOutput(std::size_t gate_id,
                                                            BEAVYProvider& beavy_provider,
                                                            ArithmeticBEAVYTensorCP<T> input,
                                                            std::size_t output_owner)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      output_owner_(output_owner),
      input_(input) {
  auto my_id = beavy_provider_.get_my_id();
  if (output_owner_ == my_id) {
    secret_share_future_ = beavy_provider_.register_for_ints_message<T>(
        1 - my_id, gate_id_, input_->get_dimensions().get_data_size());
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYTensorOutput<T> created", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorOutput<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorOutput<T>::evaluate_setup start", gate_id_));
    }
  }

  auto my_id = beavy_provider_.get_my_id();
  input_->wait_setup();
  const auto& my_secret_share = input_->get_secret_share();
  if (output_owner_ == my_id) {
    secret_shares_ = secret_share_future_.get();
    assert(my_secret_share.size() == input_->get_dimensions().get_data_size());
    assert(secret_shares_.size() == input_->get_dimensions().get_data_size());
    __gnu_parallel::transform(std::begin(secret_shares_), std::end(secret_shares_),
                              std::begin(my_secret_share), std::begin(secret_shares_), std::plus{});
  } else {
    beavy_provider_.send_ints_message<T>(1 - my_id, gate_id_, my_secret_share);
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorOutput<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorOutput<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorOutput<T>::evaluate_online start", gate_id_));
    }
  }

  auto my_id = beavy_provider_.get_my_id();
  if (output_owner_ == my_id) {
    input_->wait_online();
    const auto& public_share = input_->get_public_share();
    assert(public_share.size() == input_->get_dimensions().get_data_size());
    assert(secret_shares_.size() == input_->get_dimensions().get_data_size());
    __gnu_parallel::transform(std::begin(public_share), std::end(public_share),
                              std::begin(secret_shares_), std::begin(secret_shares_), std::minus{});
    output_promise_.set_value(std::move(secret_shares_));
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorOutput<T>::evaluate_online end", gate_id_));
    }
  }
}

template <typename T>
ENCRYPTO::ReusableFiberFuture<std::vector<T>> ArithmeticBEAVYTensorOutput<T>::get_output_future() {
  std::size_t my_id = beavy_provider_.get_my_id();
  if (output_owner_ == ALL_PARTIES || output_owner_ == my_id) {
    return output_promise_.get_future();
  } else {
    throw std::logic_error("not this parties output");
  }
}

template class ArithmeticBEAVYTensorOutput<std::uint32_t>;
template class ArithmeticBEAVYTensorOutput<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorFlatten<T>::ArithmeticBEAVYTensorFlatten(
    std::size_t gate_id, BEAVYProvider& beavy_provider, std::size_t axis,
    const ArithmeticBEAVYTensorCP<T> input)
    : NewGate(gate_id), beavy_provider_(beavy_provider), input_(input) {
  const auto& input_dims = input_->get_dimensions();
  output_ = std::make_shared<ArithmeticBEAVYTensor<T>>(flatten(input_dims, axis));

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYTensorFlatten<T> created", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorFlatten<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorFlatten<T>::evaluate_setup start", gate_id_));
    }
  }

  input_->wait_setup();
  output_->get_secret_share() = input_->get_secret_share();
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorFlatten<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorFlatten<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorFlatten<T>::evaluate_online start", gate_id_));
    }
  }

  input_->wait_online();
  output_->get_public_share() = input_->get_public_share();
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorFlatten<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYTensorFlatten<std::uint32_t>;
template class ArithmeticBEAVYTensorFlatten<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorConv2D<T>::ArithmeticBEAVYTensorConv2D(
    std::size_t gate_id, BEAVYProvider& beavy_provider, tensor::Conv2DOp conv_op,
    const ArithmeticBEAVYTensorCP<T> input, const ArithmeticBEAVYTensorCP<T> kernel,
    const ArithmeticBEAVYTensorCP<T> bias, std::size_t fractional_bits)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      conv_op_(conv_op),
      fractional_bits_(fractional_bits),
      input_(input),
      kernel_(kernel),
      bias_(bias),
      output_(std::make_shared<ArithmeticBEAVYTensor<T>>(conv_op.get_output_tensor_dims())) {
  const auto my_id = beavy_provider_.get_my_id();
  const auto output_size = conv_op_.compute_output_size();
  share_future_ = beavy_provider_.register_for_ints_message<T>(1 - my_id, gate_id_, output_size);
  auto& ap = beavy_provider_.get_arith_manager().get_provider(1 - my_id);
  conv_input_side_ = ap.template register_convolution_input_side<T>(conv_op);
  conv_kernel_side_ = ap.template register_convolution_kernel_side<T>(conv_op);
  Delta_y_share_.resize(output_size);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYTensorConv2D<T> created", gate_id_));
    }
  }
}

template <typename T>
ArithmeticBEAVYTensorConv2D<T>::~ArithmeticBEAVYTensorConv2D() = default;

template <typename T>
void ArithmeticBEAVYTensorConv2D<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorConv2D<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto output_size = conv_op_.compute_output_size();

  output_->get_secret_share() = Helpers::RandomVector<T>(output_size);
  output_->set_setup_ready();

  input_->wait_setup();
  kernel_->wait_setup();

  const auto& delta_a_share = input_->get_secret_share();
  const auto& delta_b_share = kernel_->get_secret_share();
  const auto& delta_y_share = output_->get_secret_share();

  conv_input_side_->set_input(delta_a_share);
  conv_kernel_side_->set_input(delta_b_share);

  // [Delta_y]_i = [delta_a]_i * [delta_b]_i
  convolution(conv_op_, delta_a_share.data(), delta_b_share.data(), Delta_y_share_.data());

  if (fractional_bits_ == 0) {
    // [Delta_y]_i += [delta_y]_i
    __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                              std::begin(delta_y_share), std::begin(Delta_y_share_), std::plus{});
    // NB: happens after truncation if that is requested
  }

  conv_input_side_->compute_output();
  conv_kernel_side_->compute_output();
  // [[delta_a]_i * [delta_b]_(1-i)]_i
  auto delta_ab_share1 = conv_input_side_->get_output();
  // [[delta_b]_i * [delta_a]_(1-i)]_i
  auto delta_ab_share2 = conv_kernel_side_->get_output();
  // [Delta_y]_i += [[delta_a]_i * [delta_b]_(1-i)]_i
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                            std::begin(delta_ab_share1), std::begin(Delta_y_share_), std::plus{});
  // [Delta_y]_i += [[delta_b]_i * [delta_a]_(1-i)]_i
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                            std::begin(delta_ab_share2), std::begin(Delta_y_share_), std::plus{});

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorConv2D<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorConv2D<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorConv2D<T>::evaluate_online start", gate_id_));
    }
  }

  const auto output_size = conv_op_.compute_output_size();
  input_->wait_online();
  kernel_->wait_online();
  const auto& Delta_a = input_->get_public_share();
  const auto& Delta_b = kernel_->get_public_share();
  const auto& delta_a_share = input_->get_secret_share();
  const auto& delta_b_share = kernel_->get_secret_share();
  std::vector<T> tmp(output_size);

  // after setup phase, `Delta_y_share_` contains [delta_y]_i + [delta_ab]_i

  // [Delta_y]_i -= Delta_a * [delta_b]_i
  convolution(conv_op_, Delta_a.data(), delta_b_share.data(), tmp.data());
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                            std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i -= Delta_b * [delta_a]_i
  convolution(conv_op_, delta_a_share.data(), Delta_b.data(), tmp.data());
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                            std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i += Delta_ab (== Delta_a * Delta_b)
  if (beavy_provider_.is_my_job(gate_id_)) {
    convolution(conv_op_, Delta_a.data(), Delta_b.data(), tmp.data());
    __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                              std::begin(Delta_y_share_), std::plus{});
  }

  if (fractional_bits_ > 0) {
    fixed_point::truncate_shared<T>(Delta_y_share_.data(), fractional_bits_, Delta_y_share_.size(),
                                    beavy_provider_.is_my_job(gate_id_));
    // [Delta_y]_i += [delta_y]_i
    __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                              std::begin(output_->get_secret_share()), std::begin(Delta_y_share_),
                              std::plus{});
    // NB: happens in setup phase if no truncation is requested
  }

  // broadcast [Delta_y]_i
  beavy_provider_.broadcast_ints_message(gate_id_, Delta_y_share_);
  // Delta_y = [Delta_y]_i + [Delta_y]_(1-i)
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                            std::begin(share_future_.get()), std::begin(Delta_y_share_),
                            std::plus{});
  output_->get_public_share() = std::move(Delta_y_share_);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorConv2D<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYTensorConv2D<std::uint32_t>;
template class ArithmeticBEAVYTensorConv2D<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorGemm<T>::ArithmeticBEAVYTensorGemm(std::size_t gate_id,
                                                        BEAVYProvider& beavy_provider,
                                                        tensor::GemmOp gemm_op,
                                                        const ArithmeticBEAVYTensorCP<T> input_A,
                                                        const ArithmeticBEAVYTensorCP<T> input_B,
                                                        std::size_t fractional_bits)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      gemm_op_(gemm_op),
      fractional_bits_(fractional_bits),
      input_A_(input_A),
      input_B_(input_B),
      output_(std::make_shared<ArithmeticBEAVYTensor<T>>(gemm_op.get_output_tensor_dims())) {
  const auto my_id = beavy_provider_.get_my_id();
  const auto output_size = gemm_op_.compute_output_size();
  share_future_ = beavy_provider_.register_for_ints_message<T>(1 - my_id, gate_id_, output_size);
  auto& ap = beavy_provider_.get_arith_manager().get_provider(1 - my_id);
  const auto dim_l = gemm_op_.input_A_shape_[0];
  const auto dim_m = gemm_op_.input_A_shape_[1];
  const auto dim_n = gemm_op_.input_B_shape_[1];
  mm_lhs_side_ = ap.template register_matrix_multiplication_lhs<T>(dim_l, dim_m, dim_n);
  mm_rhs_side_ = ap.template register_matrix_multiplication_rhs<T>(dim_l, dim_m, dim_n);
  Delta_y_share_.resize(output_size);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYTensorGemm<T> created", gate_id_));
    }
  }
}

template <typename T>
ArithmeticBEAVYTensorGemm<T>::~ArithmeticBEAVYTensorGemm() = default;

template <typename T>
void ArithmeticBEAVYTensorGemm<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorGemm<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto output_size = gemm_op_.compute_output_size();

  output_->get_secret_share() = Helpers::RandomVector<T>(output_size);
  output_->set_setup_ready();

  input_A_->wait_setup();
  input_B_->wait_setup();

  const auto& delta_a_share = input_A_->get_secret_share();
  const auto& delta_b_share = input_B_->get_secret_share();
  const auto& delta_y_share = output_->get_secret_share();

  mm_lhs_side_->set_input(delta_a_share);
  mm_rhs_side_->set_input(delta_b_share);

  // [Delta_y]_i = [delta_a]_i * [delta_b]_i
  matrix_multiply(gemm_op_, delta_a_share.data(), delta_b_share.data(), Delta_y_share_.data());

  if (fractional_bits_ == 0) {
    // [Delta_y]_i += [delta_y]_i
    __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                              std::begin(delta_y_share), std::begin(Delta_y_share_), std::plus{});
    // NB: happens after truncation if that is requested
  }

  mm_lhs_side_->compute_output();
  mm_rhs_side_->compute_output();
  // [[delta_a]_i * [delta_b]_(1-i)]_i
  auto delta_ab_share1 = mm_lhs_side_->get_output();
  // [[delta_b]_i * [delta_a]_(1-i)]_i
  auto delta_ab_share2 = mm_rhs_side_->get_output();
  // [Delta_y]_i += [[delta_a]_i * [delta_b]_(1-i)]_i
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                            std::begin(delta_ab_share1), std::begin(Delta_y_share_), std::plus{});
  // [Delta_y]_i += [[delta_b]_i * [delta_a]_(1-i)]_i
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                            std::begin(delta_ab_share2), std::begin(Delta_y_share_), std::plus{});

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorGemm<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorGemm<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorGemm<T>::evaluate_online start", gate_id_));
    }
  }

  const auto output_size = gemm_op_.compute_output_size();
  input_A_->wait_online();
  input_B_->wait_online();
  const auto& Delta_a = input_A_->get_public_share();
  const auto& Delta_b = input_B_->get_public_share();
  const auto& delta_a_share = input_A_->get_secret_share();
  const auto& delta_b_share = input_B_->get_secret_share();
  std::vector<T> tmp(output_size);

  // after setup phase, `Delta_y_share_` contains [delta_y]_i + [delta_ab]_i

  // [Delta_y]_i -= Delta_a * [delta_b]_i
  matrix_multiply(gemm_op_, Delta_a.data(), delta_b_share.data(), tmp.data());
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                            std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i -= Delta_b * [delta_a]_i
  matrix_multiply(gemm_op_, delta_a_share.data(), Delta_b.data(), tmp.data());
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                            std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i += Delta_ab (== Delta_a * Delta_b)
  if (beavy_provider_.is_my_job(gate_id_)) {
    matrix_multiply(gemm_op_, Delta_a.data(), Delta_b.data(), tmp.data());
    __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                              std::begin(Delta_y_share_), std::plus{});
  }

  if (fractional_bits_ > 0) {
    fixed_point::truncate_shared<T>(Delta_y_share_.data(), fractional_bits_, Delta_y_share_.size(),
                                    beavy_provider_.is_my_job(gate_id_));
    // [Delta_y]_i += [delta_y]_i
    __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                              std::begin(output_->get_secret_share()), std::begin(Delta_y_share_),
                              std::plus{});
    // NB: happens in setup phase if no truncation is requested
  }

  // broadcast [Delta_y]_i
  beavy_provider_.broadcast_ints_message(gate_id_, Delta_y_share_);
  // Delta_y = [Delta_y]_i + [Delta_y]_(1-i)
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                            std::begin(share_future_.get()), std::begin(Delta_y_share_),
                            std::plus{});
  output_->get_public_share() = std::move(Delta_y_share_);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorGemm<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYTensorGemm<std::uint32_t>;
template class ArithmeticBEAVYTensorGemm<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorMul<T>::ArithmeticBEAVYTensorMul(std::size_t gate_id,
                                                      BEAVYProvider& beavy_provider,
                                                      const ArithmeticBEAVYTensorCP<T> input_A,
                                                      const ArithmeticBEAVYTensorCP<T> input_B,
                                                      std::size_t fractional_bits)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      fractional_bits_(fractional_bits),
      input_A_(input_A),
      input_B_(input_B),
      output_(std::make_shared<ArithmeticBEAVYTensor<T>>(input_A_->get_dimensions())) {
  if (input_A_->get_dimensions() != input_B_->get_dimensions()) {
    throw std::logic_error("mismatch of dimensions");
  }
  const auto my_id = beavy_provider_.get_my_id();
  const auto data_size = input_A_->get_dimensions().get_data_size();
  share_future_ = beavy_provider_.register_for_ints_message<T>(1 - my_id, gate_id_, data_size);
  auto& ap = beavy_provider_.get_arith_manager().get_provider(1 - my_id);
  mult_sender_ = ap.template register_integer_multiplication_send<T>(data_size);
  mult_receiver_ = ap.template register_integer_multiplication_receive<T>(data_size);
  Delta_y_share_.resize(data_size);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: ArithmeticBEAVYTensorMul<T> created", gate_id_));
    }
  }
}

template <typename T>
ArithmeticBEAVYTensorMul<T>::~ArithmeticBEAVYTensorMul() = default;

template <typename T>
void ArithmeticBEAVYTensorMul<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorMul<T>::evaluate_setup start", gate_id_));
    }
  }

  const auto data_size = input_A_->get_dimensions().get_data_size();

  output_->get_secret_share() = Helpers::RandomVector<T>(data_size);
  output_->set_setup_ready();

  const auto& delta_a_share = input_A_->get_secret_share();
  const auto& delta_b_share = input_B_->get_secret_share();
  const auto& delta_y_share = output_->get_secret_share();

  mult_receiver_->set_inputs(delta_a_share);
  mult_sender_->set_inputs(delta_b_share);

  // [Delta_y]_i = [delta_a]_i * [delta_b]_i
  __gnu_parallel::transform(std::begin(delta_a_share), std::end(delta_a_share),
                            std::begin(delta_b_share), std::begin(Delta_y_share_),
                            std::multiplies{});

  if (fractional_bits_ == 0) {
    // [Delta_y]_i += [delta_y]_i
    __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                              std::begin(delta_y_share), std::begin(Delta_y_share_), std::plus{});
    // NB: happens after truncation if that is requested
  }

  mult_receiver_->compute_outputs();
  mult_sender_->compute_outputs();
  // [[delta_a]_i * [delta_b]_(1-i)]_i
  auto delta_ab_share1 = mult_receiver_->get_outputs();
  // [[delta_b]_i * [delta_a]_(1-i)]_i
  auto delta_ab_share2 = mult_sender_->get_outputs();
  // [Delta_y]_i += [[delta_a]_i * [delta_b]_(1-i)]_i
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                            std::begin(delta_ab_share1), std::begin(Delta_y_share_), std::plus{});
  // [Delta_y]_i += [[delta_b]_i * [delta_a]_(1-i)]_i
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                            std::begin(delta_ab_share2), std::begin(Delta_y_share_), std::plus{});

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorMul<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorMul<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorMul<T>::evaluate_online start", gate_id_));
    }
  }

  const auto data_size = input_A_->get_dimensions().get_data_size();
  input_A_->wait_online();
  input_B_->wait_online();
  const auto& Delta_a = input_A_->get_public_share();
  const auto& Delta_b = input_B_->get_public_share();
  const auto& delta_a_share = input_A_->get_secret_share();
  const auto& delta_b_share = input_B_->get_secret_share();
  std::vector<T> tmp(data_size);

  // after setup phase, `Delta_y_share_` contains [delta_y]_i + [delta_ab]_i

  // [Delta_y]_i -= Delta_a * [delta_b]_i
  __gnu_parallel::transform(std::begin(Delta_a), std::end(Delta_a), std::begin(delta_b_share),
                            std::begin(tmp), std::multiplies{});
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                            std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i -= Delta_b * [delta_a]_i
  __gnu_parallel::transform(std::begin(Delta_b), std::end(Delta_b), std::begin(delta_a_share),
                            std::begin(tmp), std::multiplies{});
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                            std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i += Delta_ab (== Delta_a * Delta_b)
  if (beavy_provider_.is_my_job(gate_id_)) {
    __gnu_parallel::transform(std::begin(Delta_a), std::end(Delta_a), std::begin(Delta_b),
                              std::begin(tmp), std::multiplies{});
    __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                              std::begin(Delta_y_share_), std::plus{});
  }

  if (fractional_bits_ > 0) {
    fixed_point::truncate_shared<T>(Delta_y_share_.data(), fractional_bits_, Delta_y_share_.size(),
                                    beavy_provider_.is_my_job(gate_id_));
    // [Delta_y]_i += [delta_y]_i
    __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                              std::begin(output_->get_secret_share()), std::begin(Delta_y_share_),
                              std::plus{});
    // NB: happens in setup phase if no truncation is requested
  }

  // broadcast [Delta_y]_i
  beavy_provider_.broadcast_ints_message(gate_id_, Delta_y_share_);
  // Delta_y = [Delta_y]_i + [Delta_y]_(1-i)
  __gnu_parallel::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                            std::begin(share_future_.get()), std::begin(Delta_y_share_),
                            std::plus{});
  output_->get_public_share() = std::move(Delta_y_share_);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorMul<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYTensorMul<std::uint32_t>;
template class ArithmeticBEAVYTensorMul<std::uint64_t>;

template <typename T>
BooleanToArithmeticBEAVYTensorConversion<T>::BooleanToArithmeticBEAVYTensorConversion(
    std::size_t gate_id, BEAVYProvider& beavy_provider, const BooleanBEAVYTensorCP input)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(std::move(input)),
      output_(std::make_shared<ArithmeticBEAVYTensor<T>>(input_->get_dimensions())) {
  const auto my_id = beavy_provider_.get_my_id();

  auto& ot_provider = beavy_provider_.get_ot_manager().get_provider(1 - my_id);
  if (my_id == 0) {
    ot_sender_ = ot_provider.RegisterSendACOT<T>(bit_size_ * data_size_);
  } else {
    assert(my_id == 1);
    ot_receiver_ = ot_provider.RegisterReceiveACOT<T>(bit_size_ * data_size_);
  }
  share_future_ = beavy_provider_.register_for_ints_message<T>(1 - my_id, gate_id_, data_size_);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanToArithmeticBEAVYTensorConversion<T> created", gate_id_));
    }
  }
}

template <typename T>
BooleanToArithmeticBEAVYTensorConversion<T>::~BooleanToArithmeticBEAVYTensorConversion() = default;

template <typename T>
void BooleanToArithmeticBEAVYTensorConversion<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanToArithmeticBEAVYTensorConversion<T>::evaluate_setup start", gate_id_));
    }
  }

  output_->get_secret_share() = Helpers::RandomVector<T>(data_size_);
  output_->set_setup_ready();

  input_->wait_setup();
  const auto& sshares = input_->get_secret_share();
  assert(sshares.size() == bit_size_);

  std::vector<T> ot_output;
  if (ot_sender_ != nullptr) {
    std::vector<T> correlations(bit_size_ * data_size_);
#pragma omp parallel for
    for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
      for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
        if (sshares[bit_j].Get(int_i)) {
          correlations[bit_j * data_size_ + int_i] = 1;
        }
      }
    }
    ot_sender_->SetCorrelations(std::move(correlations));
    ot_sender_->SendMessages();
    ot_sender_->ComputeOutputs();
    ot_output = ot_sender_->GetOutputs();
#pragma omp parallel for
    for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
      for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
        T bit = sshares[bit_j].Get(int_i);
        ot_output[bit_j * data_size_ + int_i] = bit + 2 * ot_output[bit_j * data_size_ + int_i];
      }
    }
  } else {
    assert(ot_receiver_ != nullptr);
    ENCRYPTO::BitVector<> choices;
    choices.Reserve(Helpers::Convert::BitsToBytes(bit_size_ * data_size_));
    for (const auto& sshare : sshares) {
      choices.Append(sshare);
    }
    ot_receiver_->SetChoices(std::move(choices));
    ot_receiver_->SendCorrections();
    ot_receiver_->ComputeOutputs();
    ot_output = ot_receiver_->GetOutputs();
#pragma omp parallel for
    for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
      for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
        T bit = sshares[bit_j].Get(int_i);
        ot_output[bit_j * data_size_ + int_i] = bit - 2 * ot_output[bit_j * data_size_ + int_i];
      }
    }
  }
  arithmetized_secret_share_ = std::move(ot_output);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanToArithmeticBEAVYTensorConversion<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void BooleanToArithmeticBEAVYTensorConversion<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanToArithmeticBEAVYTensorConversion<T>::evaluate_online start", gate_id_));
    }
  }

  const auto my_id = beavy_provider_.get_my_id();
  std::vector<T> arithmetized_public_share(bit_size_ * data_size_);

  input_->wait_online();
  const auto& pshares = input_->get_public_share();

#pragma omp parallel for
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      if (pshares[bit_j].Get(int_i)) {
        arithmetized_public_share[bit_j * data_size_ + int_i] = 1;
      }
    }
  }

  auto tmp = output_->get_secret_share();
  if (beavy_provider_.is_my_job(gate_id_)) {
#pragma omp parallel for
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
        const auto p = arithmetized_public_share[bit_j * data_size_ + int_i];
        const auto s = arithmetized_secret_share_[bit_j * data_size_ + int_i];
        tmp[int_i] += (p + (1 - 2 * p) * s) << bit_j;
      }
    }
  } else {
#pragma omp parallel for
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
        const auto p = arithmetized_public_share[bit_j * data_size_ + int_i];
        const auto s = arithmetized_secret_share_[bit_j * data_size_ + int_i];
        tmp[int_i] += ((1 - 2 * p) * s) << bit_j;
      }
    }
  }
  beavy_provider_.send_ints_message(1 - my_id, gate_id_, tmp);
  const auto other_share = share_future_.get();
  __gnu_parallel::transform(std::begin(tmp), std::end(tmp), std::begin(other_share),
                            std::begin(tmp), std::plus{});
  output_->get_public_share() = std::move(tmp);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanToArithmeticBEAVYTensorConversion<T>::evaluate_online end", gate_id_));
    }
  }
}

template class BooleanToArithmeticBEAVYTensorConversion<std::uint32_t>;
template class BooleanToArithmeticBEAVYTensorConversion<std::uint64_t>;

BooleanBEAVYTensorRelu::BooleanBEAVYTensorRelu(std::size_t gate_id, BEAVYProvider& beavy_provider,
                                               const BooleanBEAVYTensorCP input)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(std::move(input)),
      output_(std::make_shared<BooleanBEAVYTensor>(input_->get_dimensions(), bit_size_)) {
  const auto my_id = beavy_provider_.get_my_id();
  share_future_ =
      beavy_provider_.register_for_bits_message(1 - my_id, gate_id_, data_size_ * (bit_size_ - 1));
  Delta_y_share_.Reserve(Helpers::Convert::BitsToBytes((bit_size_ - 1) * data_size_));
  auto& otp = beavy_provider_.get_ot_manager().get_provider(1 - my_id);
  ot_sender_ = otp.RegisterSendXCOTBit(data_size_, bit_size_ - 1);
  ot_receiver_ = otp.RegisterReceiveXCOTBit(data_size_, bit_size_ - 1);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: BooleanGMWTensorRelu created", gate_id_));
    }
  }
}

BooleanBEAVYTensorRelu::~BooleanBEAVYTensorRelu() = default;

void BooleanBEAVYTensorRelu::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYTensorRelu::evaluate_setup start", gate_id_));
    }
  }

  // generate the secret shares
  auto& out_sshares = output_->get_secret_share();
  assert(out_sshares.size() == bit_size_);
#pragma omp parallel for
  for (std::size_t bit_j = 0; bit_j < bit_size_ - 1; ++bit_j) {
    out_sshares[bit_j] = ENCRYPTO::BitVector<>::Random(data_size_);
  }
  // the last bit is always 0
  out_sshares[bit_size_ - 1] = ENCRYPTO::BitVector<>(data_size_, false);
  output_->set_setup_ready();

  input_->wait_setup();
  const auto& sshares = input_->get_secret_share();
  const auto& my_msb_sshare = sshares[bit_size_ - 1];

  ot_receiver_->SetChoices(my_msb_sshare);
  ot_receiver_->SendCorrections();
  ENCRYPTO::BitVector<> ot_inputs((bit_size_ - 1) * data_size_);
  // inefficient, but works ...
  for (std::size_t bit_j = 0; bit_j < bit_size_ - 1; ++bit_j) {
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      ot_inputs.Set(sshares[bit_j].Get(int_i), int_i * (bit_size_ - 1) + bit_j);
    }
  }
  ot_sender_->SetCorrelations(std::move(ot_inputs));
  ot_sender_->SendMessages();
  ot_sender_->ComputeOutputs();
  ot_receiver_->ComputeOutputs();

  // compute the products of the msb_mask with all other masks
  for (std::size_t bit_j = 0; bit_j < bit_size_ - 1; ++bit_j) {
    // local part
    auto tmp = my_msb_sshare & sshares[bit_j];
    // output mask
    tmp ^= out_sshares[bit_j];
    Delta_y_share_.Append(tmp);
  }
  const auto ot_snd_out = ot_sender_->GetOutputs();
  const auto ot_rcv_out = ot_receiver_->GetOutputs();
  // inefficient, but works ...
  for (std::size_t bit_j = 0; bit_j < bit_size_ - 1; ++bit_j) {
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      bool tmp = Delta_y_share_.Get(bit_j * data_size_ + int_i);
      // product of other msb_mask with my sshare masks
      tmp ^= ot_snd_out.Get(int_i * (bit_size_ - 1) + bit_j);
      // product of my_msb_mask with other sshare masks
      tmp ^= ot_rcv_out.Get(int_i * (bit_size_ - 1) + bit_j);
      Delta_y_share_.Set(tmp, bit_j * data_size_ + int_i);
    }
  }
  // => Delta_y_share_ contains now [delta_ab]_i ^ [delta_y]_i

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYTensorRelu::evaluate_setup end", gate_id_));
    }
  }
}

void BooleanBEAVYTensorRelu::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYTensorRelu::evaluate_online start", gate_id_));
    }
  }

  input_->wait_online();
  const auto& pshares = input_->get_public_share();
  const auto my_msb_pshare = ~pshares[bit_size_ - 1];
  const auto& sshares = input_->get_secret_share();
  const auto& my_msb_sshare = sshares[bit_size_ - 1];

  ENCRYPTO::BitVector tmp;
  tmp.Reserve(Helpers::Convert::BitsToBytes((bit_size_ - 1) * data_size_));
  const bool my_job = beavy_provider_.is_my_job(gate_id_);
  for (std::size_t bit_j = 0; bit_j < bit_size_ - 1; ++bit_j) {
    // Delta_y_share_ ^= Delta_a & [delta_b]_i
    auto tmp2 = my_msb_pshare & sshares[bit_j];
    // Delta_y_share_ ^= [delta_a]_i & Delta_b
    tmp2 ^= my_msb_sshare & pshares[bit_j];
    if (my_job) {
      // Delta_y_share_ ^= Delta_a & Delta_b
      tmp2 ^= my_msb_pshare & pshares[bit_j];
    }
    tmp.Append(tmp2);
  }
  Delta_y_share_ ^= tmp;
  const auto my_id = beavy_provider_.get_my_id();
  beavy_provider_.send_bits_message(1 - my_id, gate_id_, Delta_y_share_);
  Delta_y_share_ ^= share_future_.get();

  auto& out_pshares = output_->get_public_share();
#pragma omp parallel for
  for (std::size_t bit_j = 0; bit_j < bit_size_ - 1; ++bit_j) {
    out_pshares[bit_j] = Delta_y_share_.Subset(bit_j * data_size_, (bit_j + 1) * data_size_);
  }
  out_pshares[bit_size_ - 1].Resize(data_size_, true);  // fill with zeros
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYTensorRelu::evaluate_online end", gate_id_));
    }
  }
}

template <typename T>
BooleanXArithmeticBEAVYTensorRelu<T>::BooleanXArithmeticBEAVYTensorRelu(
    std::size_t gate_id, BEAVYProvider& beavy_provider, const BooleanBEAVYTensorCP input_bool,
    const ArithmeticBEAVYTensorCP<T> input_arith)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      data_size_(input_bool->get_dimensions().get_data_size()),
      input_bool_(std::move(input_bool)),
      input_arith_(std::move(input_arith)),
      output_(std::make_shared<ArithmeticBEAVYTensor<T>>(input_arith_->get_dimensions())) {
  if (input_bool_->get_dimensions() != input_arith_->get_dimensions()) {
    throw std::invalid_argument("dimension mismatch");
  }
  if (input_bool_->get_bit_size() != input_arith_->get_bit_size()) {
    throw std::invalid_argument("bit size mismatch");
  }
  const auto my_id = beavy_provider_.get_my_id();
  auto& ap = beavy_provider_.get_arith_manager().get_provider(1 - my_id);
  if (beavy_provider_.is_my_job(gate_id_)) {
    mult_int_side_ = ap.register_bit_integer_multiplication_int_side<T>(data_size_, 2);
    mult_bit_side_ = ap.register_bit_integer_multiplication_bit_side<T>(data_size_, 1);
  } else {
    mult_int_side_ = ap.register_bit_integer_multiplication_int_side<T>(data_size_, 1);
    mult_bit_side_ = ap.register_bit_integer_multiplication_bit_side<T>(data_size_, 2);
  }
  delta_b_share_.resize(data_size_);
  delta_b_x_delta_n_share_.resize(data_size_);
  share_future_ = beavy_provider_.register_for_ints_message<T>(1 - my_id, gate_id_, data_size_);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: BooleanXArithmeticBEAVYTensorRelu created", gate_id_));
    }
  }
}

template <typename T>
BooleanXArithmeticBEAVYTensorRelu<T>::~BooleanXArithmeticBEAVYTensorRelu() = default;

template <typename T>
void BooleanXArithmeticBEAVYTensorRelu<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanXArithmeticBEAVYTensorRelu::evaluate_setup start", gate_id_));
    }
  }

  output_->get_secret_share() = Helpers::RandomVector<T>(data_size_);
  output_->set_setup_ready();

  input_bool_->wait_setup();
  input_arith_->wait_setup();
  const auto& int_sshare = input_arith_->get_secret_share();
  assert(int_sshare.size() == data_size_);
  const auto& msb_sshare = input_bool_->get_secret_share()[bit_size_ - 1];
  assert(msb_sshare.GetSize() == data_size_);

  std::vector<T> msb_sshare_as_ints(data_size_);
  for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
    msb_sshare_as_ints[int_i] = msb_sshare.Get(int_i);
  }

  mult_bit_side_->set_inputs(msb_sshare);

  if (beavy_provider_.is_my_job(gate_id_)) {
    std::vector<T> mult_inputs(2 * data_size_);
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      mult_inputs[2 * int_i] = msb_sshare_as_ints[int_i];
      mult_inputs[2 * int_i + 1] =
          int_sshare[int_i] - 2 * msb_sshare_as_ints[int_i] * int_sshare[int_i];
    }
    mult_int_side_->set_inputs(std::move(mult_inputs));
  } else {
    std::vector<T> mult_inputs(data_size_);
    std::transform(std::begin(int_sshare), std::end(int_sshare), std::begin(msb_sshare_as_ints),
                   std::begin(mult_inputs), [](auto n, auto b) { return n - 2 * b * n; });
    mult_int_side_->set_inputs(std::move(mult_inputs));
  }

  mult_bit_side_->compute_outputs();
  mult_int_side_->compute_outputs();
  auto mult_bit_side_out = mult_bit_side_->get_outputs();
  auto mult_int_side_out = mult_int_side_->get_outputs();

  // compute [delta_b]^A and [delta_b * delta_n]^A
  if (beavy_provider_.is_my_job(gate_id_)) {
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      delta_b_share_[int_i] = msb_sshare_as_ints[int_i] - 2 * mult_int_side_out[2 * int_i];
      delta_b_x_delta_n_share_[int_i] = msb_sshare_as_ints[int_i] * int_sshare[int_i] +
                                        mult_int_side_out[2 * int_i + 1] + mult_bit_side_out[int_i];
    }
  } else {
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      delta_b_share_[int_i] = msb_sshare_as_ints[int_i] - 2 * mult_bit_side_out[2 * int_i];
      delta_b_x_delta_n_share_[int_i] = msb_sshare_as_ints[int_i] * int_sshare[int_i] +
                                        mult_bit_side_out[2 * int_i + 1] + mult_int_side_out[int_i];
    }
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanXArithmeticBEAVYTensorRelu::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void BooleanXArithmeticBEAVYTensorRelu<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanXArithmeticBEAVYTensorRelu::evaluate_online start", gate_id_));
    }
  }

  input_bool_->wait_online();
  input_arith_->wait_online();
  const auto& int_sshare = input_arith_->get_secret_share();
  const auto& int_pshare = input_arith_->get_public_share();
  assert(int_pshare.size() == data_size_);
  const auto& msb_pshare = input_bool_->get_public_share()[bit_size_ - 1];
  assert(msb_pshare.GetSize() == data_size_);

  const auto& sshare = output_->get_secret_share();
  std::vector<T> pshare(data_size_);

#pragma omp parallel for
  for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
    T Delta_b = !msb_pshare.Get(int_i);
    auto Delta_n = int_pshare[int_i];
    pshare[int_i] = delta_b_share_[int_i] * (Delta_n - 2 * Delta_b * Delta_n) -
                    Delta_b * int_sshare[int_i] -
                    delta_b_x_delta_n_share_[int_i] * (1 - 2 * Delta_b) + sshare[int_i];
    if (beavy_provider_.is_my_job(gate_id_)) {
      pshare[int_i] += Delta_b * Delta_n;
    }
  }

  beavy_provider_.broadcast_ints_message(gate_id_, pshare);
  const auto other_pshare = share_future_.get();
  __gnu_parallel::transform(std::begin(pshare), std::end(pshare), std::begin(other_pshare),
                            std::begin(pshare), std::plus{});

  output_->get_public_share() = std::move(pshare);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanXArithmeticBEAVYTensorRelu::evaluate_online end", gate_id_));
    }
  }
}

template class BooleanXArithmeticBEAVYTensorRelu<std::uint32_t>;
template class BooleanXArithmeticBEAVYTensorRelu<std::uint64_t>;

BooleanBEAVYTensorMaxPool::BooleanBEAVYTensorMaxPool(std::size_t gate_id,
                                                     BEAVYProvider& beavy_provider,
                                                     tensor::MaxPoolOp maxpool_op,
                                                     const BooleanBEAVYTensorCP input)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      maxpool_op_(maxpool_op),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(
          std::make_shared<BooleanBEAVYTensor>(maxpool_op_.get_output_tensor_dims(), bit_size_)),
      // XXX: use depth-optimized circuit here
      maxpool_algo_(beavy_provider_.get_circuit_loader().load_maxpool_circuit(
          bit_size_, maxpool_op_.compute_kernel_size(), true)) {
  if (!maxpool_op_.verify()) {
    throw std::invalid_argument("invalid MaxPoolOp");
  }
  const auto kernel_size = maxpool_op_.compute_kernel_size();
  const auto output_size = maxpool_op_.compute_output_size();
  input_wires_.resize(bit_size_ * kernel_size);
  std::generate(std::begin(input_wires_), std::end(input_wires_), [output_size] {
    auto w = std::make_shared<BooleanBEAVYWire>(output_size);
    w->get_secret_share().Resize(output_size);
    w->get_public_share().Resize(output_size);
    return w;
  });
  {
    WireVector in(bit_size_ * kernel_size);
    std::transform(std::begin(input_wires_), std::end(input_wires_), std::begin(in),
                   [](auto w) { return std::dynamic_pointer_cast<BooleanBEAVYWire>(w); });
    auto [gates, out] = construct_circuit(beavy_provider_, maxpool_algo_, in);
    gates_ = std::move(gates);
    assert(out.size() == bit_size_);
    output_wires_.resize(bit_size_);
    std::transform(std::begin(out), std::end(out), std::begin(output_wires_),
                   [](auto w) { return std::dynamic_pointer_cast<BooleanBEAVYWire>(w); });
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: BooleanBEAVYTensorMaxPool created", gate_id_));
    }
  }
}

template <bool setup>
static void prepare_wires(std::size_t bit_size, const tensor::MaxPoolOp& maxpool_op,
                          BooleanBEAVYWireVector& circuit_wires,
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

#pragma omp parallel for
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
              if constexpr (setup) {
                auto& bv = circuit_wires[mpin_wires_idx(bit_j, k_row, k_col)]->get_secret_share();
                bv.Set(bit, out_idx(channel_i, o_row, o_col));
              } else {
                auto& bv = circuit_wires[mpin_wires_idx(bit_j, k_row, k_col)]->get_public_share();
                bv.Set(bit, out_idx(channel_i, o_row, o_col));
              }
            }
          }

          i_col += strides[1];
        }
        i_row += strides[0];
      }
    }
  }
  for (auto& wire : circuit_wires) {
    if constexpr (setup) {
      wire->set_setup_ready();
    } else {
      wire->set_online_ready();
    }
  }
}

void BooleanBEAVYTensorMaxPool::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYTensorMaxPool::evaluate_setup start", gate_id_));
    }
  }

  input_->wait_setup();

  prepare_wires<true>(bit_size_, maxpool_op_, input_wires_, input_->get_secret_share());

  for (auto& gate : gates_) {
    // should work since its a Boolean circuit consisting of AND, XOR, INV gates
    gate->evaluate_setup();
  }

  auto& output_shares = output_->get_secret_share();
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    auto& wire = output_wires_[bit_j];
    wire->wait_setup();
    output_shares[bit_j] = std::move(wire->get_secret_share());
  }
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYTensorMaxPool::evaluate_setup end", gate_id_));
    }
  }
}

void BooleanBEAVYTensorMaxPool::evaluate_setup_with_context(ExecutionContext& exec_ctx) {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanBEAVYTensorMaxPool::evaluate_setup_with_context start", gate_id_));
    }
  }

  input_->wait_setup();

  prepare_wires<true>(bit_size_, maxpool_op_, input_wires_, input_->get_secret_share());

  for (auto& gate : gates_) {
    exec_ctx.fpool_->post([&] { gate->evaluate_online(); });
  }

  auto& output_shares = output_->get_secret_share();
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    auto& wire = output_wires_[bit_j];
    wire->wait_setup();
    output_shares[bit_j] = std::move(wire->get_secret_share());
  }
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanBEAVYTensorMaxPool::evaluate_setup_with_context end", gate_id_));
    }
  }
}

void BooleanBEAVYTensorMaxPool::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYTensorMaxPool::evaluate_online start", gate_id_));
    }
  }

  input_->wait_online();

  prepare_wires<false>(bit_size_, maxpool_op_, input_wires_, input_->get_public_share());

  for (auto& gate : gates_) {
    // should work since its a Boolean circuit consisting of AND, XOR, INV gates
    gate->evaluate_online();
  }

  auto& output_shares = output_->get_public_share();
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    auto& wire = output_wires_[bit_j];
    wire->wait_online();
    output_shares[bit_j] = std::move(wire->get_public_share());
  }
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: BooleanBEAVYTensorMaxPool::evaluate_online end", gate_id_));
    }
  }
}

void BooleanBEAVYTensorMaxPool::evaluate_online_with_context(ExecutionContext& exec_ctx) {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanBEAVYTensorMaxPool::evaluate_online_with_context start", gate_id_));
    }
  }

  input_->wait_online();

  prepare_wires<false>(bit_size_, maxpool_op_, input_wires_, input_->get_public_share());

  for (auto& gate : gates_) {
    exec_ctx.fpool_->post([&] { gate->evaluate_online(); });
  }

  auto& output_shares = output_->get_public_share();
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    auto& wire = output_wires_[bit_j];
    wire->wait_online();
    output_shares[bit_j] = std::move(wire->get_public_share());
  }
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = beavy_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: BooleanBEAVYTensorMaxPool::evaluate_online_with_context end", gate_id_));
    }
  }
}

}  // namespace MOTION::proto::beavy
