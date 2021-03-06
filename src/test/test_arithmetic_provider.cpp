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

#include <gtest/gtest.h>

#include "communication/communication_layer.h"
#include "crypto/arithmetic_provider.h"
#include "crypto/base_ots/base_ot_provider.h"
#include "crypto/motion_base_provider.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "utility/linear_algebra.h"

template <typename T>
class ArithmeticProviderTest : public ::testing::Test {
  using is_enabled_t_ = ENCRYPTO::is_unsigned_int_t<T>;

 protected:
  void SetUp() override {
    comm_layers_ = MOTION::Communication::make_dummy_communication_layers(2);
    base_ot_providers_.resize(2);
    motion_base_providers_.resize(2);
    ot_provider_managers_.resize(2);
    arithmetic_provider_managers_.resize(2);
    for (std::size_t i = 0; i < 2; ++i) {
      base_ot_providers_[i] =
          std::make_unique<MOTION::BaseOTProvider>(*comm_layers_[i], nullptr, nullptr);
      motion_base_providers_[i] =
          std::make_unique<MOTION::Crypto::MotionBaseProvider>(*comm_layers_[i], nullptr);
      ot_provider_managers_[i] = std::make_unique<ENCRYPTO::ObliviousTransfer::OTProviderManager>(
          *comm_layers_[i], *base_ot_providers_[i], *motion_base_providers_[i], nullptr, nullptr);
      arithmetic_provider_managers_[i] = std::make_unique<MOTION::ArithmeticProviderManager>(
          *comm_layers_[i], *ot_provider_managers_[i], nullptr);
    }

    std::vector<std::future<void>> futs;
    for (std::size_t i = 0; i < 2; ++i) {
      futs.emplace_back(std::async(std::launch::async, [this, i] {
        comm_layers_[i]->start();
        motion_base_providers_[i]->setup();
        base_ot_providers_[i]->ComputeBaseOTs();
      }));
    }
    std::for_each(std::begin(futs), std::end(futs), [](auto& f) { f.get(); });
  }

  void TearDown() override {
    std::vector<std::future<void>> futs;
    for (std::size_t i = 0; i < 2; ++i) {
      futs.emplace_back(std::async(std::launch::async, [this, i] { comm_layers_[i]->shutdown(); }));
    }
    std::for_each(std::begin(futs), std::end(futs), [](auto& f) { f.get(); });
  }

  const std::size_t sender_i_ = 0;
  const std::size_t receiver_i_ = 1;
  MOTION::ArithmeticProvider& get_sender_provider() {
    return arithmetic_provider_managers_[sender_i_]->get_provider(receiver_i_);
  }
  MOTION::ArithmeticProvider& get_receiver_provider() {
    return arithmetic_provider_managers_[receiver_i_]->get_provider(sender_i_);
  }

  void run_setup() {
    std::vector<std::future<void>> futs;
    for (std::size_t i = 0; i < 2; ++i) {
      futs.emplace_back(std::async(std::launch::async, [this, i] {
        ot_provider_managers_[i]->get_provider(1 - i).SendSetup();
      }));
      futs.emplace_back(std::async(std::launch::async, [this, i] {
        ot_provider_managers_[i]->get_provider(1 - i).ReceiveSetup();
      }));
    }
    std::for_each(std::begin(futs), std::end(futs), [](auto& f) { f.get(); });
  }

  std::vector<std::unique_ptr<MOTION::Communication::CommunicationLayer>> comm_layers_;
  std::vector<std::unique_ptr<MOTION::BaseOTProvider>> base_ot_providers_;
  std::vector<std::unique_ptr<MOTION::Crypto::MotionBaseProvider>> motion_base_providers_;
  std::vector<std::unique_ptr<ENCRYPTO::ObliviousTransfer::OTProviderManager>>
      ot_provider_managers_;
  std::vector<std::unique_ptr<MOTION::ArithmeticProviderManager>> arithmetic_provider_managers_;
};

using integer_types =
    ::testing::Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, __uint128_t>;
TYPED_TEST_SUITE(ArithmeticProviderTest, integer_types);

TYPED_TEST(ArithmeticProviderTest, BitIntegerMultiplication) {
  const std::size_t batch_size = 10;

  const auto input_int_side = MOTION::Helpers::RandomVector<TypeParam>(batch_size);
  const auto input_bit_side = ENCRYPTO::BitVector<>::Random(batch_size);
  auto mult_int_side =
      this->get_sender_provider().template register_bit_integer_multiplication_int_side<TypeParam>(
          batch_size);
  auto mult_bit_side =
      this->get_receiver_provider()
          .template register_bit_integer_multiplication_bit_side<TypeParam>(batch_size);

  this->run_setup();

  mult_int_side->set_inputs(input_int_side);
  mult_bit_side->set_inputs(input_bit_side);
  mult_int_side->compute_outputs();
  mult_bit_side->compute_outputs();
  const auto output_int_side = mult_int_side->get_outputs();
  const auto output_bit_side = mult_bit_side->get_outputs();

  ASSERT_EQ(output_int_side.size(), batch_size);
  ASSERT_EQ(output_bit_side.size(), batch_size);

  for (std::size_t i = 0; i < batch_size; ++i) {
    if (input_bit_side.Get(i)) {
      ASSERT_EQ(TypeParam(output_int_side[i] + output_bit_side[i]), input_int_side[i]);
    } else {
      ASSERT_EQ(TypeParam(output_int_side[i] + output_bit_side[i]), TypeParam(0));
    }
  }
}

TYPED_TEST(ArithmeticProviderTest, BitIntegerVectorMultiplication) {
  const std::size_t batch_size = 10;
  const std::size_t vector_size = 8;

  const auto input_int_side = MOTION::Helpers::RandomVector<TypeParam>(batch_size * vector_size);
  const auto input_bit_side = ENCRYPTO::BitVector<>::Random(batch_size);
  auto mult_int_side =
      this->get_sender_provider().template register_bit_integer_multiplication_int_side<TypeParam>(
          batch_size, vector_size);
  auto mult_bit_side = this->get_receiver_provider()
                           .template register_bit_integer_multiplication_bit_side<TypeParam>(
                               batch_size, vector_size);

  this->run_setup();

  mult_int_side->set_inputs(input_int_side);
  mult_bit_side->set_inputs(input_bit_side);
  mult_int_side->compute_outputs();
  mult_bit_side->compute_outputs();
  const auto output_int_side = mult_int_side->get_outputs();
  const auto output_bit_side = mult_bit_side->get_outputs();

  ASSERT_EQ(output_int_side.size(), batch_size * vector_size);
  ASSERT_EQ(output_bit_side.size(), batch_size * vector_size);

  for (std::size_t i = 0; i < batch_size; ++i) {
    if (input_bit_side.Get(i)) {
      for (std::size_t j = 0; j < vector_size; ++j) {
        ASSERT_EQ(
            TypeParam(output_int_side[i * vector_size + j] + output_bit_side[i * vector_size + j]),
            input_int_side[i * vector_size + j]);
      }
    } else {
      for (std::size_t j = 0; j < vector_size; ++j) {
        ASSERT_EQ(
            TypeParam(output_int_side[i * vector_size + j] + output_bit_side[i * vector_size + j]),
            TypeParam(0));
      }
    }
  }
}

TYPED_TEST(ArithmeticProviderTest, IntegerMultiplication) {
  const std::size_t batch_size = 10;

  const auto input_sender = MOTION::Helpers::RandomVector<TypeParam>(batch_size);
  const auto input_receiver = MOTION::Helpers::RandomVector<TypeParam>(batch_size);
  auto mult_sender =
      this->get_sender_provider().template register_integer_multiplication_send<TypeParam>(
          batch_size);
  auto mult_receiver =
      this->get_receiver_provider().template register_integer_multiplication_receive<TypeParam>(
          batch_size);

  this->run_setup();

  mult_sender->set_inputs(input_sender);
  mult_receiver->set_inputs(input_receiver);
  mult_sender->compute_outputs();
  mult_receiver->compute_outputs();
  const auto output_sender = mult_sender->get_outputs();
  const auto output_receiver = mult_receiver->get_outputs();

  ASSERT_EQ(output_sender.size(), batch_size);
  ASSERT_EQ(output_receiver.size(), batch_size);

  for (std::size_t i = 0; i < batch_size; ++i) {
    ASSERT_EQ(TypeParam(output_sender[i] + output_receiver[i]),
              TypeParam(input_sender[i] * input_receiver[i]));
  }
}

TYPED_TEST(ArithmeticProviderTest, IntegerVectorMultiplication) {
  const std::size_t batch_size = 10;
  const std::size_t vector_size = 8;

  const auto input_sender = MOTION::Helpers::RandomVector<TypeParam>(batch_size * vector_size);
  const auto input_receiver = MOTION::Helpers::RandomVector<TypeParam>(batch_size);
  auto mult_sender =
      this->get_sender_provider().template register_integer_multiplication_send<TypeParam>(
          batch_size, vector_size);
  auto mult_receiver =
      this->get_receiver_provider().template register_integer_multiplication_receive<TypeParam>(
          batch_size, vector_size);

  this->run_setup();

  mult_sender->set_inputs(input_sender);
  mult_receiver->set_inputs(input_receiver);
  mult_sender->compute_outputs();
  mult_receiver->compute_outputs();
  const auto output_sender = mult_sender->get_outputs();
  const auto output_receiver = mult_receiver->get_outputs();

  ASSERT_EQ(output_sender.size(), batch_size * vector_size);
  ASSERT_EQ(output_receiver.size(), batch_size * vector_size);

  for (std::size_t i = 0; i < batch_size; ++i) {
    for (std::size_t j = 0; j < vector_size; ++j) {
      ASSERT_EQ(
          TypeParam(output_sender[i * vector_size + j] + output_receiver[i * vector_size + j]),
          TypeParam(input_sender[i * vector_size + j] * input_receiver[i]));
    }
  }
}

TYPED_TEST(ArithmeticProviderTest, MatrixMultiplication) {
  const std::size_t dim_l = 7;
  const std::size_t dim_m = 13;
  const std::size_t dim_n = 11;

  const auto input_rhs = MOTION::Helpers::RandomVector<TypeParam>(dim_m * dim_n);
  const auto input_lhs = MOTION::Helpers::RandomVector<TypeParam>(dim_l * dim_m);
  std::vector<TypeParam> expected_output =
      MOTION::matrix_multiply(dim_l, dim_m, dim_n, input_lhs, input_rhs);
  ASSERT_EQ(expected_output.size(), dim_l * dim_n);

  auto mm_rhs = this->get_sender_provider().template register_matrix_multiplication_rhs<TypeParam>(
      dim_l, dim_m, dim_n);
  auto mm_lhs =
      this->get_receiver_provider().template register_matrix_multiplication_lhs<TypeParam>(
          dim_l, dim_m, dim_n);

  this->run_setup();

  mm_rhs->set_input(input_rhs);
  mm_lhs->set_input(input_lhs);
  mm_rhs->compute_output();
  mm_lhs->compute_output();
  const auto output_sender = mm_rhs->get_output();
  const auto output_receiver = mm_lhs->get_output();

  ASSERT_EQ(output_sender.size(), dim_l * dim_n);
  ASSERT_EQ(output_receiver.size(), dim_l * dim_n);

  for (std::size_t row_i = 0; row_i < dim_l; ++row_i) {
    for (std::size_t col_j = 0; col_j < dim_n; ++col_j) {
      ASSERT_EQ(
          TypeParam(output_sender[row_i * dim_n + col_j] + output_receiver[row_i * dim_n + col_j]),
          TypeParam(expected_output[row_i * dim_n + col_j]));
    }
  }
}

TYPED_TEST(ArithmeticProviderTest, Convolution) {
  // Convolution from CryptoNets
  const MOTION::tensor::Conv2DOp conv_op = {.kernel_shape_ = {5, 1, 5, 5},
                                            .input_shape_ = {1, 28, 28},
                                            .output_shape_ = {5, 13, 13},
                                            .dilations_ = {1, 1},
                                            .pads_ = {1, 1, 0, 0},
                                            .strides_ = {2, 2}};
  ASSERT_TRUE(conv_op.verify());

  const auto input = MOTION::Helpers::RandomVector<TypeParam>(conv_op.compute_input_size());
  const auto kernel = MOTION::Helpers::RandomVector<TypeParam>(conv_op.compute_kernel_size());
  std::vector<TypeParam> expected_output = MOTION::convolution(conv_op, input, kernel);
  ASSERT_EQ(expected_output.size(), conv_op.compute_output_size());

  auto conv_input_side =
      this->get_sender_provider().template register_convolution_input_side<TypeParam>(conv_op);
  auto conv_kernel_side =
      this->get_receiver_provider().template register_convolution_kernel_side<TypeParam>(conv_op);

  this->run_setup();

  conv_input_side->set_input(input);
  conv_kernel_side->set_input(kernel);
  conv_input_side->compute_output();
  conv_kernel_side->compute_output();
  const auto output_is = conv_input_side->get_output();
  const auto output_ks = conv_kernel_side->get_output();

  ASSERT_EQ(output_is.size(), conv_op.compute_output_size());
  ASSERT_EQ(output_ks.size(), conv_op.compute_output_size());

  for (std::size_t i = 0; i < expected_output.size(); ++i) {
    ASSERT_EQ(TypeParam(output_is[i] + output_ks[i]), TypeParam(expected_output[i]));
  }
}
