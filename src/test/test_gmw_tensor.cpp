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

#include <array>
#include <iterator>
#include <memory>

#include <gtest/gtest.h>

#include "base/gate_register.h"
#include "communication/communication_layer.h"
#include "crypto/arithmetic_provider.h"
#include "crypto/base_ots/base_ot_provider.h"
#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/linalg_triple_provider.h"
#include "crypto/multiplication_triple/mt_provider.h"
#include "crypto/multiplication_triple/sb_provider.h"
#include "crypto/multiplication_triple/sp_provider.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "gate/new_gate.h"
#include "protocols/gmw/gmw_provider.h"
#include "protocols/gmw/tensor_op.h"
#include "protocols/gmw/wire.h"
#include "statistics/run_time_stats.h"
#include "tensor/tensor.h"
#include "utility/helpers.h"
#include "utility/linear_algebra.h"
#include "utility/logger.h"

using namespace MOTION::proto::gmw;

class GMWTensorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    comm_layers_ = MOTION::Communication::make_dummy_communication_layers(2);
    for (std::size_t i = 0; i < 2; ++i) {
      loggers_[i] = std::make_shared<MOTION::Logger>(i, boost::log::trivial::severity_level::trace);
      comm_layers_[i]->set_logger(loggers_[i]);
      base_ot_providers_[i] =
          std::make_unique<MOTION::BaseOTProvider>(*comm_layers_[i], nullptr, nullptr);
      motion_base_providers_[i] =
          std::make_unique<MOTION::Crypto::MotionBaseProvider>(*comm_layers_[i], nullptr);
      ot_provider_managers_[i] = std::make_unique<ENCRYPTO::ObliviousTransfer::OTProviderManager>(
          *comm_layers_[i], *base_ot_providers_[i], *motion_base_providers_[i], nullptr, nullptr);
      arithmetic_provider_managers_[i] = std::make_unique<MOTION::ArithmeticProviderManager>(
          *comm_layers_[i], *ot_provider_managers_[i], nullptr);
      linalg_triple_providers_[i] = std::make_shared<MOTION::LinAlgTriplesFromAP>(
          arithmetic_provider_managers_[i]->get_provider(1 - i),
          ot_provider_managers_[i]->get_provider(1 - i), stats_[i], nullptr);
      mt_providers_[i] = std::make_unique<MOTION::MTProviderFromOTs>(
          i, 2, *arithmetic_provider_managers_[i], *ot_provider_managers_[i], stats_[i], nullptr);
      sp_providers_[i] = std::make_unique<MOTION::SPProviderFromOTs>(
          ot_provider_managers_[i]->get_providers(), i, stats_[i], nullptr);
      sb_providers_[i] = std::make_unique<MOTION::TwoPartySBProvider>(
          *comm_layers_[i], ot_provider_managers_[i]->get_provider(1 - i), stats_[i], nullptr);
      gate_registers_[i] = std::make_unique<MOTION::GateRegister>();
      gmw_providers_[i] = std::make_unique<GMWProvider>(
          *comm_layers_[i], *gate_registers_[i], *motion_base_providers_[i],
          *ot_provider_managers_[i], *mt_providers_[i], *sp_providers_[i], *sb_providers_[i],
          loggers_[i]);
      gmw_providers_[i]->set_linalg_triple_provider(linalg_triple_providers_[i]);
    }
  }

  void TearDown() override {
    std::vector<std::future<void>> futs;
    for (std::size_t i = 0; i < 2; ++i) {
      futs.emplace_back(std::async(std::launch::async, [this, i] { comm_layers_[i]->shutdown(); }));
    }
    std::for_each(std::begin(futs), std::end(futs), [](auto& f) { f.get(); });
  }

  const std::size_t garbler_i_ = 0;
  const std::size_t evaluator_i_ = 1;
  ENCRYPTO::ObliviousTransfer::OTProvider& get_garbler_ot_provider() {
    return ot_provider_managers_[garbler_i_]->get_provider(evaluator_i_);
  }
  ENCRYPTO::ObliviousTransfer::OTProvider& get_evaluator_ot_provider() {
    return ot_provider_managers_[evaluator_i_]->get_provider(garbler_i_);
  }

  void run_setup() {
    std::vector<std::future<void>> futs;
    for (std::size_t i = 0; i < 2; ++i) {
      futs.emplace_back(std::async(std::launch::async, [this, i] {
        comm_layers_[i]->start();
        motion_base_providers_[i]->setup();
        base_ot_providers_[i]->ComputeBaseOTs();
        mt_providers_[i]->PreSetup();
        sp_providers_[i]->PreSetup();
        sb_providers_[i]->PreSetup();
        auto f = std::async(std::launch::async, [this, i] {
          ot_provider_managers_[i]->get_provider(1 - i).SendSetup();
        });
        ot_provider_managers_[i]->get_provider(1 - i).ReceiveSetup();
        f.get();
        linalg_triple_providers_[i]->setup();
        mt_providers_[i]->Setup();
        sp_providers_[i]->Setup();
        sb_providers_[i]->Setup();
        gmw_providers_[i]->setup();
      }));
    }
    std::for_each(std::begin(futs), std::end(futs), [](auto& f) { f.get(); });
  }

  void run_gates_setup() {
    auto& gates_g = gate_registers_[garbler_i_]->get_gates();
    auto& gates_e = gate_registers_[evaluator_i_]->get_gates();
    for (auto& gate : gates_g) {
      gate->evaluate_setup();
    }
    for (auto& gate : gates_e) {
      gate->evaluate_setup();
    }
  }

  void run_gates_online() {
    auto& gates_g = gate_registers_[garbler_i_]->get_gates();
    auto& gates_e = gate_registers_[evaluator_i_]->get_gates();
    auto fut_g = std::async(std::launch::async, [&gates_g] {
      for (auto& gate : gates_g) {
        gate->evaluate_online();
      }
    });
    auto fut_e = std::async(std::launch::async, [&gates_e] {
      for (auto& gate : gates_e) {
        gate->evaluate_online();
      }
    });
    fut_g.get();
    fut_e.get();
  }

  std::vector<std::unique_ptr<MOTION::Communication::CommunicationLayer>> comm_layers_;
  std::array<std::unique_ptr<MOTION::BaseOTProvider>, 2> base_ot_providers_;
  std::array<std::unique_ptr<MOTION::Crypto::MotionBaseProvider>, 2> motion_base_providers_;
  std::array<std::unique_ptr<ENCRYPTO::ObliviousTransfer::OTProviderManager>, 2>
      ot_provider_managers_;
  std::array<std::unique_ptr<MOTION::ArithmeticProviderManager>, 2> arithmetic_provider_managers_;
  std::array<std::shared_ptr<MOTION::LinAlgTripleProvider>, 2> linalg_triple_providers_;
  std::array<std::unique_ptr<MOTION::MTProvider>, 2> mt_providers_;
  std::array<std::unique_ptr<MOTION::SPProvider>, 2> sp_providers_;
  std::array<std::unique_ptr<MOTION::SBProvider>, 2> sb_providers_;
  std::array<std::unique_ptr<MOTION::GateRegister>, 2> gate_registers_;
  std::array<std::unique_ptr<GMWProvider>, 2> gmw_providers_;
  std::array<std::shared_ptr<MOTION::Logger>, 2> loggers_;
  std::array<MOTION::Statistics::RunTimeStats, 2> stats_;
};

template <typename T>
class ArithmeticGMWTensorTest : public GMWTensorTest {
 public:
  static std::vector<T> generate_inputs(const MOTION::tensor::TensorDimensions dims) {
    return MOTION::Helpers::RandomVector<T>(dims.get_data_size());
  }
  std::pair<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<T>>, MOTION::tensor::TensorCP>
  make_arithmetic_T_tensor_input_my(std::size_t party_id,
                                    const MOTION::tensor::TensorDimensions& dims) {
    auto& gp = *gmw_providers_.at(party_id);
    static_assert(ENCRYPTO::bit_size_v<T> == 64);
    return gp.make_arithmetic_64_tensor_input_my(dims);
  }
  MOTION::tensor::TensorCP make_arithmetic_T_tensor_input_other(
      std::size_t party_id, const MOTION::tensor::TensorDimensions& dims) {
    auto& gp = *gmw_providers_.at(party_id);
    static_assert(ENCRYPTO::bit_size_v<T> == 64);
    return gp.make_arithmetic_64_tensor_input_other(dims);
  }
  ENCRYPTO::ReusableFiberFuture<MOTION::IntegerValues<T>> make_arithmetic_T_tensor_output_my(
      std::size_t party_id, const MOTION::tensor::TensorCP& in) {
    auto& gp = *gmw_providers_.at(party_id);
    static_assert(ENCRYPTO::bit_size_v<T> == 64);
    return gp.make_arithmetic_64_tensor_output_my(in);
  }
};

using integer_types = ::testing::Types<std::uint64_t>;
TYPED_TEST_SUITE(ArithmeticGMWTensorTest, integer_types);

TYPED_TEST(ArithmeticGMWTensorTest, Input) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input_a = this->generate_inputs(dims);
  const auto input_b = this->generate_inputs(dims);

  auto [input_a_promise, tensor_a_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_a_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);
  auto tensor_b_in_0 = this->make_arithmetic_T_tensor_input_other(0, dims);
  auto [input_b_promise, tensor_b_in_1] = this->make_arithmetic_T_tensor_input_my(1, dims);

  ASSERT_EQ(tensor_a_in_0->get_dimensions(), dims);
  ASSERT_EQ(tensor_a_in_1->get_dimensions(), dims);
  ASSERT_EQ(tensor_b_in_0->get_dimensions(), dims);
  ASSERT_EQ(tensor_b_in_1->get_dimensions(), dims);

  this->run_setup();
  this->run_gates_setup();
  input_a_promise.set_value(input_a);
  input_b_promise.set_value(input_b);
  this->run_gates_online();

  const auto tensor_a0 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_a_in_0);
  const auto tensor_a1 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_a_in_1);
  const auto tensor_b0 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_b_in_0);
  const auto tensor_b1 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_b_in_1);

  ASSERT_NE(tensor_a0, nullptr);
  ASSERT_NE(tensor_a1, nullptr);
  ASSERT_NE(tensor_b0, nullptr);
  ASSERT_NE(tensor_b1, nullptr);

  tensor_a0->wait_online();
  tensor_a1->wait_online();
  tensor_b0->wait_online();
  tensor_b1->wait_online();

  const auto& share_a0 = tensor_a0->get_share();
  const auto& share_a1 = tensor_a1->get_share();
  const auto& share_b0 = tensor_b0->get_share();
  const auto& share_b1 = tensor_b1->get_share();

  ASSERT_EQ(share_a0.size(), input_a.size());
  ASSERT_EQ(share_a1.size(), input_a.size());
  ASSERT_EQ(share_b0.size(), input_b.size());
  ASSERT_EQ(share_b1.size(), input_b.size());

  for (std::size_t i = 0; i < input_a.size(); ++i) {
    ASSERT_EQ(input_a[i], TypeParam(share_a0[i] + share_a1[i]));
    ASSERT_EQ(input_b[i], TypeParam(share_b0[i] + share_b1[i]));
  }
}

TYPED_TEST(ArithmeticGMWTensorTest, OutputSingle) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input_a = this->generate_inputs(dims);

  auto [input_a_promise, tensor_a_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_a_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);
  this->gmw_providers_[0]->make_arithmetic_tensor_output_other(tensor_a_in_0);
  auto output_future = this->make_arithmetic_T_tensor_output_my(1, tensor_a_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_a_promise.set_value(input_a);
  this->run_gates_online();

  auto output = output_future.get();

  ASSERT_EQ(output.size(), dims.get_data_size());
  ASSERT_EQ(input_a, output);
}

TYPED_TEST(ArithmeticGMWTensorTest, Convolution) {
  // Convolution from CryptoNets
  const MOTION::tensor::Conv2DOp conv_op = {.kernel_shape_ = {5, 1, 5, 5},
                                            .input_shape_ = {1, 28, 28},
                                            .output_shape_ = {5, 13, 13},
                                            .dilations_ = {1, 1},
                                            .pads_ = {1, 1, 0, 0},
                                            .strides_ = {2, 2}};
  ASSERT_TRUE(conv_op.verify());
  const auto input_dims = conv_op.get_input_tensor_dims();
  const auto kernel_dims = conv_op.get_kernel_tensor_dims();
  const auto output_dims = conv_op.get_output_tensor_dims();
  const auto input = this->generate_inputs(input_dims);
  const auto kernel = this->generate_inputs(kernel_dims);

  auto [input_promise, tensor_input_0] = this->make_arithmetic_T_tensor_input_my(0, input_dims);
  auto tensor_input_1 = this->make_arithmetic_T_tensor_input_other(1, input_dims);
  auto tensor_kernel_0 = this->make_arithmetic_T_tensor_input_other(0, kernel_dims);
  auto [kernel_promise, tensor_kernel_1] = this->make_arithmetic_T_tensor_input_my(1, kernel_dims);

  ASSERT_EQ(tensor_input_0->get_dimensions(), input_dims);
  ASSERT_EQ(tensor_input_1->get_dimensions(), input_dims);
  ASSERT_EQ(tensor_kernel_0->get_dimensions(), kernel_dims);
  ASSERT_EQ(tensor_kernel_1->get_dimensions(), kernel_dims);

  auto tensor_output_0 =
      this->gmw_providers_[0]->make_tensor_conv2d_op(conv_op, tensor_input_0, tensor_kernel_0);
  auto tensor_output_1 =
      this->gmw_providers_[1]->make_tensor_conv2d_op(conv_op, tensor_input_1, tensor_kernel_1);

  ASSERT_EQ(tensor_output_0->get_dimensions(), output_dims);
  ASSERT_EQ(tensor_output_1->get_dimensions(), output_dims);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  kernel_promise.set_value(kernel);
  this->run_gates_online();

  const auto output_gmw_tensor_0 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_output_0);
  const auto output_gmw_tensor_1 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_output_1);

  const auto& output_share_0 = output_gmw_tensor_0->get_share();
  const auto& output_share_1 = output_gmw_tensor_1->get_share();

  ASSERT_EQ(output_share_0.size(), output_dims.get_data_size());
  ASSERT_EQ(output_share_1.size(), output_dims.get_data_size());

  const auto expected_output = MOTION::convolution(conv_op, input, kernel);
  const auto plain_output = MOTION::Helpers::AddVectors(output_share_0, output_share_1);

  ASSERT_EQ(plain_output, expected_output);
}

TYPED_TEST(ArithmeticGMWTensorTest, Gemm) {
  const MOTION::tensor::GemmOp gemm_op = {
      .input_A_shape_ = {1, 100}, .input_B_shape_ = {100, 10}, .output_shape_ = {1, 10}};
  ASSERT_TRUE(gemm_op.verify());
  const auto input_A_dims = gemm_op.get_input_A_tensor_dims();
  const auto input_B_dims = gemm_op.get_input_B_tensor_dims();
  const auto output_dims = gemm_op.get_output_tensor_dims();
  const auto input_A = this->generate_inputs(input_A_dims);
  const auto input_B = this->generate_inputs(input_B_dims);

  auto [input_A_promise, tensor_input_A_0] =
      this->make_arithmetic_T_tensor_input_my(0, input_A_dims);
  auto tensor_input_A_1 = this->make_arithmetic_T_tensor_input_other(1, input_A_dims);
  auto tensor_input_B_0 = this->make_arithmetic_T_tensor_input_other(0, input_B_dims);
  auto [input_B_promise, tensor_input_B_1] =
      this->make_arithmetic_T_tensor_input_my(1, input_B_dims);

  ASSERT_EQ(tensor_input_A_0->get_dimensions(), input_A_dims);
  ASSERT_EQ(tensor_input_A_1->get_dimensions(), input_A_dims);
  ASSERT_EQ(tensor_input_B_0->get_dimensions(), input_B_dims);
  ASSERT_EQ(tensor_input_B_1->get_dimensions(), input_B_dims);

  auto tensor_output_0 =
      this->gmw_providers_[0]->make_tensor_gemm_op(gemm_op, tensor_input_A_0, tensor_input_B_0);
  auto tensor_output_1 =
      this->gmw_providers_[1]->make_tensor_gemm_op(gemm_op, tensor_input_A_1, tensor_input_B_1);

  ASSERT_EQ(tensor_output_0->get_dimensions(), output_dims);
  ASSERT_EQ(tensor_output_1->get_dimensions(), output_dims);

  this->run_setup();
  this->run_gates_setup();
  input_A_promise.set_value(input_A);
  input_B_promise.set_value(input_B);
  this->run_gates_online();

  const auto output_gmw_tensor_0 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_output_0);
  const auto output_gmw_tensor_1 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_output_1);

  const auto& output_share_0 = output_gmw_tensor_0->get_share();
  const auto& output_share_1 = output_gmw_tensor_1->get_share();

  ASSERT_EQ(output_share_0.size(), output_dims.get_data_size());
  ASSERT_EQ(output_share_1.size(), output_dims.get_data_size());

  const auto expected_output =
      MOTION::matrix_multiply(gemm_op.input_A_shape_[0], gemm_op.input_A_shape_[1],
                              gemm_op.input_B_shape_[1], input_A, input_B);
  const auto plain_output = MOTION::Helpers::AddVectors(output_share_0, output_share_1);

  ASSERT_EQ(plain_output, expected_output);
}

TYPED_TEST(ArithmeticGMWTensorTest, Sqr) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_out_0 = this->gmw_providers_[0]->make_tensor_sqr_op(tensor_in_0);
  auto tensor_out_1 = this->gmw_providers_[1]->make_tensor_sqr_op(tensor_in_1);

  ASSERT_EQ(tensor_out_0->get_dimensions(), dims);
  ASSERT_EQ(tensor_out_1->get_dimensions(), dims);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto tensor_output_0 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_out_0);
  const auto tensor_output_1 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(tensor_out_1);

  ASSERT_NE(tensor_output_0, nullptr);
  ASSERT_NE(tensor_output_1, nullptr);

  tensor_output_0->wait_online();
  tensor_output_1->wait_online();

  const auto& share_0 = tensor_output_0->get_share();
  const auto& share_1 = tensor_output_1->get_share();

  ASSERT_EQ(share_0.size(), input.size());
  ASSERT_EQ(share_1.size(), input.size());

  for (std::size_t i = 0; i < input.size(); ++i) {
    ASSERT_EQ(TypeParam(input[i] * input[i]), TypeParam(share_0[i] + share_1[i]));
  }
}
