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

#include <stdexcept>

#include "beavy_provider.h"
#include "crypto/arithmetic_provider.h"
#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/linalg_triple_provider.h"
#include "crypto/multiplication_triple/sp_provider.h"
#include "crypto/oblivious_transfer/ot_flavors.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "crypto/sharing_randomness_generator.h"
#include "utility/constants.h"
#include "utility/linear_algebra.h"
#include "utility/logger.h"

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
  std::transform(std::begin(my_public_share), std::end(my_public_share),
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
  std::transform(std::begin(input), std::end(input), std::begin(my_public_share),
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
    std::transform(std::begin(secret_shares_), std::end(secret_shares_),
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
    std::transform(std::begin(public_share), std::end(public_share), std::begin(secret_shares_),
                   std::begin(secret_shares_), std::minus{});
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

template class ArithmeticBEAVYTensorOutput<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorFlatten<T>::ArithmeticBEAVYTensorFlatten(
    std::size_t gate_id, BEAVYProvider& gmw_provider, std::size_t axis,
    const ArithmeticBEAVYTensorCP<T> input)
    : NewGate(gate_id), gmw_provider_(gmw_provider), input_(input) {
  const auto& input_dims = input_->get_dimensions();
  output_ = std::make_shared<ArithmeticBEAVYTensor<T>>(flatten(input_dims, axis));
}

template <typename T>
void ArithmeticBEAVYTensorFlatten<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorFlatten<T>::evaluate_setup start", gate_id_));
    }
  }

  input_->wait_setup();
  output_->get_secret_share() = input_->get_secret_share();
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorFlatten<T>::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYTensorFlatten<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorFlatten<T>::evaluate_online start", gate_id_));
    }
  }

  input_->wait_online();
  output_->get_public_share() = input_->get_public_share();
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = gmw_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: ArithmeticBEAVYTensorFlatten<T>::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYTensorFlatten<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorConv2D<T>::ArithmeticBEAVYTensorConv2D(std::size_t gate_id,
                                                            BEAVYProvider& beavy_provider,
                                                            tensor::Conv2DOp conv_op,
                                                            const ArithmeticBEAVYTensorCP<T> input,
                                                            const ArithmeticBEAVYTensorCP<T> kernel,
                                                            const ArithmeticBEAVYTensorCP<T> bias)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      conv_op_(conv_op),
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
  // [Delta_y]_i += [delta_y]_i
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(delta_y_share),
                 std::begin(Delta_y_share_), std::plus{});

  conv_input_side_->compute_output();
  conv_kernel_side_->compute_output();
  // [[delta_a]_i * [delta_b]_(1-i)]_i
  auto delta_ab_share1 = conv_input_side_->get_output();
  // [[delta_b]_i * [delta_a]_(1-i)]_i
  auto delta_ab_share2 = conv_kernel_side_->get_output();
  // [Delta_y]_i += [[delta_a]_i * [delta_b]_(1-i)]_i
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(delta_ab_share1),
                 std::begin(Delta_y_share_), std::plus{});
  // [Delta_y]_i += [[delta_b]_i * [delta_a]_(1-i)]_i
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(delta_ab_share2),
                 std::begin(Delta_y_share_), std::plus{});

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
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                 std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i -= Delta_b * [delta_a]_i
  convolution(conv_op_, delta_a_share.data(), Delta_b.data(), tmp.data());
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                 std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i += Delta_ab (== Delta_a * Delta_b)
  if (beavy_provider_.is_my_job(gate_id_)) {
    convolution(conv_op_, Delta_a.data(), Delta_b.data(), tmp.data());
    std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                   std::begin(Delta_y_share_), std::plus{});
  }
  // broadcast [Delta_y]_i
  beavy_provider_.broadcast_ints_message(gate_id_, Delta_y_share_);
  // Delta_y = [Delta_y]_i + [Delta_y]_(1-i)
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                 std::begin(share_future_.get()), std::begin(Delta_y_share_), std::plus{});
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

template class ArithmeticBEAVYTensorConv2D<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorGemm<T>::ArithmeticBEAVYTensorGemm(std::size_t gate_id,
                                                        BEAVYProvider& beavy_provider,
                                                        tensor::GemmOp gemm_op,
                                                        const ArithmeticBEAVYTensorCP<T> input_A,
                                                        const ArithmeticBEAVYTensorCP<T> input_B)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
      gemm_op_(gemm_op),
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
  matrix_multiply(gemm_op_, delta_a_share.data(), delta_b_share.data(),
                  Delta_y_share_.data());
  // [Delta_y]_i += [delta_y]_i
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(delta_y_share),
                 std::begin(Delta_y_share_), std::plus{});

  mm_lhs_side_->compute_output();
  mm_rhs_side_->compute_output();
  // [[delta_a]_i * [delta_b]_(1-i)]_i
  auto delta_ab_share1 = mm_lhs_side_->get_output();
  // [[delta_b]_i * [delta_a]_(1-i)]_i
  auto delta_ab_share2 = mm_rhs_side_->get_output();
  // [Delta_y]_i += [[delta_a]_i * [delta_b]_(1-i)]_i
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(delta_ab_share1),
                 std::begin(Delta_y_share_), std::plus{});
  // [Delta_y]_i += [[delta_b]_i * [delta_a]_(1-i)]_i
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(delta_ab_share2),
                 std::begin(Delta_y_share_), std::plus{});

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
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                 std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i -= Delta_b * [delta_a]_i
  matrix_multiply(gemm_op_, delta_a_share.data(), Delta_b.data(), tmp.data());
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                 std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i += Delta_ab (== Delta_a * Delta_b)
  if (beavy_provider_.is_my_job(gate_id_)) {
    matrix_multiply(gemm_op_, Delta_a.data(), Delta_b.data(), tmp.data());
    std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                   std::begin(Delta_y_share_), std::plus{});
  }
  // broadcast [Delta_y]_i
  beavy_provider_.broadcast_ints_message(gate_id_, Delta_y_share_);
  // Delta_y = [Delta_y]_i + [Delta_y]_(1-i)
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                 std::begin(share_future_.get()), std::begin(Delta_y_share_), std::plus{});
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

template class ArithmeticBEAVYTensorGemm<std::uint64_t>;

template <typename T>
ArithmeticBEAVYTensorMul<T>::ArithmeticBEAVYTensorMul(std::size_t gate_id,
                                                      BEAVYProvider& beavy_provider,
                                                      const ArithmeticBEAVYTensorCP<T> input_A,
                                                      const ArithmeticBEAVYTensorCP<T> input_B)
    : NewGate(gate_id),
      beavy_provider_(beavy_provider),
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
  std::transform(std::begin(delta_a_share), std::end(delta_a_share), std::begin(delta_b_share),
                 std::begin(Delta_y_share_), std::multiplies{});
  // [Delta_y]_i += [delta_y]_i
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(delta_y_share),
                 std::begin(Delta_y_share_), std::plus{});

  mult_receiver_->compute_outputs();
  mult_sender_->compute_outputs();
  // [[delta_a]_i * [delta_b]_(1-i)]_i
  auto delta_ab_share1 = mult_receiver_->get_outputs();
  // [[delta_b]_i * [delta_a]_(1-i)]_i
  auto delta_ab_share2 = mult_sender_->get_outputs();
  // [Delta_y]_i += [[delta_a]_i * [delta_b]_(1-i)]_i
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(delta_ab_share1),
                 std::begin(Delta_y_share_), std::plus{});
  // [Delta_y]_i += [[delta_b]_i * [delta_a]_(1-i)]_i
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(delta_ab_share2),
                 std::begin(Delta_y_share_), std::plus{});

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
  std::transform(std::begin(Delta_a), std::end(Delta_a), std::begin(delta_b_share), std::begin(tmp),
                 std::multiplies{});
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                 std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i -= Delta_b * [delta_a]_i
  std::transform(std::begin(Delta_b), std::end(Delta_b), std::begin(delta_a_share), std::begin(tmp),
                 std::multiplies{});
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                 std::begin(Delta_y_share_), std::minus{});

  // [Delta_y]_i += Delta_ab (== Delta_a * Delta_b)
  if (beavy_provider_.is_my_job(gate_id_)) {
    std::transform(std::begin(Delta_a), std::end(Delta_a), std::begin(Delta_b), std::begin(tmp),
                   std::multiplies{});
    std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_), std::begin(tmp),
                   std::begin(Delta_y_share_), std::plus{});
  }
  // broadcast [Delta_y]_i
  beavy_provider_.broadcast_ints_message(gate_id_, Delta_y_share_);
  // Delta_y = [Delta_y]_i + [Delta_y]_(1-i)
  std::transform(std::begin(Delta_y_share_), std::end(Delta_y_share_),
                 std::begin(share_future_.get()), std::begin(Delta_y_share_), std::plus{});
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

  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      if (pshares[bit_j].Get(int_i)) {
        arithmetized_public_share[bit_j * data_size_ + int_i] = 1;
      }
    }
  }

  auto tmp = output_->get_secret_share();
  if (beavy_provider_.is_my_job(gate_id_)) {
    for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
      for (std::size_t simd_j = 0; simd_j < data_size_; ++simd_j) {
        const auto p = arithmetized_public_share[bit_j * data_size_ + simd_j];
        const auto s = arithmetized_secret_share_[bit_j * data_size_ + simd_j];
        tmp[simd_j] += (p + (1 - 2 * p) * s) << bit_j;
      }
    }
  } else {
    for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
      for (std::size_t simd_j = 0; simd_j < data_size_; ++simd_j) {
        const auto p = arithmetized_public_share[bit_j * data_size_ + simd_j];
        const auto s = arithmetized_secret_share_[bit_j * data_size_ + simd_j];
        tmp[simd_j] += ((1 - 2 * p) * s) << bit_j;
      }
    }
  }
  beavy_provider_.send_ints_message(1 - my_id, gate_id_, tmp);
  const auto other_share = share_future_.get();
  std::transform(std::begin(tmp), std::end(tmp), std::begin(other_share), std::begin(tmp),
                 std::plus{});
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

template class BooleanToArithmeticBEAVYTensorConversion<std::uint64_t>;

}  // namespace MOTION::proto::beavy
