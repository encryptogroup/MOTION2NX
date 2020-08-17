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
#include <iterator>
#include <memory>

#include "communication/communication_layer.h"
#include "crypto/arithmetic_provider.h"
#include "crypto/base_ots/base_ot_provider.h"
#include "crypto/motion_base_provider.h"
#include "crypto/multiplication_triple/linalg_triple_provider.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "utility/linear_algebra.h"

template <typename T>
class LinAlgTripleProviderTest : public ::testing::Test {
  using is_enabled_t_ = ENCRYPTO::is_unsigned_int_t<T>;

 protected:
  void SetUp() override {
    comm_layers_ = MOTION::Communication::make_dummy_communication_layers(2);
    base_ot_providers_.resize(2);
    motion_base_providers_.resize(2);
    ot_provider_managers_.resize(2);
    arithmetic_provider_managers_.resize(2);
    linalg_triple_providers_.resize(2);
    for (std::size_t i = 0; i < 2; ++i) {
      base_ot_providers_[i] =
          std::make_unique<MOTION::BaseOTProvider>(*comm_layers_[i], nullptr, nullptr);
      motion_base_providers_[i] =
          std::make_unique<MOTION::Crypto::MotionBaseProvider>(*comm_layers_[i], nullptr);
      ot_provider_managers_[i] = std::make_unique<ENCRYPTO::ObliviousTransfer::OTProviderManager>(
          *comm_layers_[i], *base_ot_providers_[i], *motion_base_providers_[i], nullptr, nullptr);
      arithmetic_provider_managers_[i] = std::make_unique<MOTION::ArithmeticProviderManager>(
          *comm_layers_[i], *ot_provider_managers_[i], nullptr);
      linalg_triple_providers_[i] = std::make_unique<MOTION::LinAlgTriplesFromAP>(
          arithmetic_provider_managers_[i]->get_provider(1 - i),
          ot_provider_managers_[i]->get_provider(1 - i), nullptr);
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
    futs.clear();
    for (std::size_t i = 0; i < 2; ++i) {
      futs.emplace_back(
          std::async(std::launch::async, [this, i] { linalg_triple_providers_[i]->setup(); }));
    }
    std::for_each(std::begin(futs), std::end(futs), [](auto& f) { f.get(); });
  }

  std::vector<std::unique_ptr<MOTION::Communication::CommunicationLayer>> comm_layers_;
  std::vector<std::unique_ptr<MOTION::BaseOTProvider>> base_ot_providers_;
  std::vector<std::unique_ptr<MOTION::Crypto::MotionBaseProvider>> motion_base_providers_;
  std::vector<std::unique_ptr<ENCRYPTO::ObliviousTransfer::OTProviderManager>>
      ot_provider_managers_;
  std::vector<std::unique_ptr<MOTION::ArithmeticProviderManager>> arithmetic_provider_managers_;
  std::vector<std::unique_ptr<MOTION::LinAlgTripleProvider>> linalg_triple_providers_;
};

using integer_types =
    ::testing::Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, __uint128_t>;
TYPED_TEST_SUITE(LinAlgTripleProviderTest, integer_types);

TYPED_TEST(LinAlgTripleProviderTest, Gemm) {
  const MOTION::tensor::GemmOp gemm_op = {
      .input_A_shape_ = {7, 11}, .input_B_shape_ = {11, 13}, .output_shape_ = {7, 13}};
  ASSERT_TRUE(gemm_op.verify());

  auto index_0 =
      this->linalg_triple_providers_[0]->template register_for_gemm_triple<TypeParam>(gemm_op);
  auto index_1 =
      this->linalg_triple_providers_[1]->template register_for_gemm_triple<TypeParam>(gemm_op);

  this->run_setup();

  auto triple_0 =
      this->linalg_triple_providers_[0]->template get_gemm_triple<TypeParam>(gemm_op, index_0);
  auto triple_1 =
      this->linalg_triple_providers_[1]->template get_gemm_triple<TypeParam>(gemm_op, index_1);

  ASSERT_EQ(triple_0.a_.size(), gemm_op.compute_input_A_size());
  ASSERT_EQ(triple_0.b_.size(), gemm_op.compute_input_B_size());
  ASSERT_EQ(triple_0.c_.size(), gemm_op.compute_output_size());

  ASSERT_EQ(triple_0.a_.size(), triple_1.a_.size());
  ASSERT_EQ(triple_0.b_.size(), triple_1.b_.size());
  ASSERT_EQ(triple_0.c_.size(), triple_1.c_.size());

  MOTION::LinAlgTripleProvider::LinAlgTriple<TypeParam> plain_triple;

  plain_triple.a_ = MOTION::Helpers::AddVectors(triple_0.a_, triple_1.a_);
  plain_triple.b_ = MOTION::Helpers::AddVectors(triple_0.b_, triple_1.b_);
  plain_triple.c_ = MOTION::Helpers::AddVectors(triple_0.c_, triple_1.c_);

  auto expected_c =
      MOTION::matrix_multiply(gemm_op.input_A_shape_[0], gemm_op.input_A_shape_[1],
                              gemm_op.output_shape_[1], plain_triple.a_, plain_triple.b_);
  ASSERT_EQ(plain_triple.c_, expected_c);
}

TYPED_TEST(LinAlgTripleProviderTest, Convolution) {
  // Convolution from CryptoNets
  const MOTION::tensor::Conv2DOp conv_op = {.kernel_shape_ = {5, 1, 5, 5},
                                            .input_shape_ = {1, 28, 28},
                                            .output_shape_ = {5, 13, 13},
                                            .dilations_ = {1, 1},
                                            .pads_ = {1, 1, 0, 0},
                                            .strides_ = {2, 2}};
  ASSERT_TRUE(conv_op.verify());

  auto index_0 =
      this->linalg_triple_providers_[0]->template register_for_conv2d_triple<TypeParam>(conv_op);
  auto index_1 =
      this->linalg_triple_providers_[1]->template register_for_conv2d_triple<TypeParam>(conv_op);

  this->run_setup();

  auto triple_0 =
      this->linalg_triple_providers_[0]->template get_conv2d_triple<TypeParam>(conv_op, index_0);
  auto triple_1 =
      this->linalg_triple_providers_[1]->template get_conv2d_triple<TypeParam>(conv_op, index_1);

  ASSERT_EQ(triple_0.a_.size(), conv_op.compute_input_size());
  ASSERT_EQ(triple_0.b_.size(), conv_op.compute_kernel_size());
  ASSERT_EQ(triple_0.c_.size(), conv_op.compute_output_size());

  ASSERT_EQ(triple_0.a_.size(), triple_1.a_.size());
  ASSERT_EQ(triple_0.b_.size(), triple_1.b_.size());
  ASSERT_EQ(triple_0.c_.size(), triple_1.c_.size());

  MOTION::LinAlgTripleProvider::LinAlgTriple<TypeParam> plain_triple;

  plain_triple.a_ = MOTION::Helpers::AddVectors(triple_0.a_, triple_1.a_);
  plain_triple.b_ = MOTION::Helpers::AddVectors(triple_0.b_, triple_1.b_);
  plain_triple.c_ = MOTION::Helpers::AddVectors(triple_0.c_, triple_1.c_);

  auto expected_c = MOTION::convolution(conv_op, plain_triple.a_, plain_triple.b_);
  ASSERT_EQ(plain_triple.c_, expected_c);
}

TYPED_TEST(LinAlgTripleProviderTest, ReLU) {
  std::size_t num_triples = 100;
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;

  auto index_0 = this->linalg_triple_providers_[0]->register_for_relu_triple(num_triples, bit_size);
  auto index_1 = this->linalg_triple_providers_[1]->register_for_relu_triple(num_triples, bit_size);

  this->run_setup();

  auto triple_0 =
      this->linalg_triple_providers_[0]->get_relu_triple(num_triples, bit_size, index_0);
  auto triple_1 =
      this->linalg_triple_providers_[1]->get_relu_triple(num_triples, bit_size, index_1);

  ASSERT_EQ(triple_0.a_.GetSize(), num_triples);
  ASSERT_EQ(triple_0.b_.size(), bit_size - 1);
  ASSERT_EQ(triple_0.c_.size(), bit_size - 1);
  ASSERT_EQ(triple_0.a_.GetSize(), triple_1.a_.GetSize());
  ASSERT_EQ(triple_0.b_.size(), triple_1.b_.size());
  ASSERT_EQ(triple_0.c_.size(), triple_1.c_.size());
  for (std::size_t bit_j = 0; bit_j < bit_size - 1; ++bit_j) {
    ASSERT_EQ(triple_0.b_.at(bit_j).GetSize(), num_triples);
    ASSERT_EQ(triple_0.c_.at(bit_j).GetSize(), num_triples);
    ASSERT_EQ(triple_0.b_.at(bit_j).GetSize(), triple_1.b_.at(bit_j).GetSize());
    ASSERT_EQ(triple_0.c_.at(bit_j).GetSize(), triple_1.c_.at(bit_j).GetSize());
  }

  MOTION::LinAlgTripleProvider::BooleanTriple plain_triple;
  plain_triple.a_ = triple_0.a_ ^ triple_1.a_;
  plain_triple.b_.resize(bit_size - 1);
  plain_triple.c_.resize(bit_size - 1);
  for (std::size_t bit_j = 0; bit_j < bit_size - 1; ++bit_j) {
    plain_triple.b_.at(bit_j) = triple_0.b_.at(bit_j) ^ triple_1.b_.at(bit_j);
    plain_triple.c_.at(bit_j) = triple_0.c_.at(bit_j) ^ triple_1.c_.at(bit_j);
  }

  std::vector<ENCRYPTO::BitVector<>> expected_c(bit_size - 1);
  for (std::size_t bit_j = 0; bit_j < bit_size - 1; ++bit_j) {
    expected_c.at(bit_j) = plain_triple.a_ & plain_triple.b_.at(bit_j);
  }
  ASSERT_EQ(plain_triple.c_, expected_c);
}
