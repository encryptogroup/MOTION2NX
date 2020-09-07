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

#include "algorithm/circuit_loader.h"
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
#include "protocols/beavy/beavy_provider.h"
#include "protocols/beavy/tensor_op.h"
#include "protocols/gmw/gmw_provider.h"
#include "protocols/gmw/tensor_op.h"
#include "protocols/yao/tensor.h"
#include "protocols/yao/yao_provider.h"
#include "statistics/run_time_stats.h"
#include "tensor/tensor.h"
#include "utility/helpers.h"
#include "utility/linear_algebra.h"
#include "utility/logger.h"

using namespace MOTION::proto::beavy;
using namespace MOTION::proto::gmw;
using namespace MOTION::proto::yao;

class YaoTensorTest : public ::testing::Test {
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
      yao_providers_[i] = std::make_unique<YaoProvider>(
          *comm_layers_[i], *gate_registers_[i], circuit_loader_, *motion_base_providers_[i],
          ot_provider_managers_[i]->get_provider(1 - i), loggers_[i]);
      beavy_providers_[i] = std::make_unique<BEAVYProvider>(
          *comm_layers_[i], *gate_registers_[i], *motion_base_providers_[i],
          *ot_provider_managers_[i], *arithmetic_provider_managers_[i], loggers_[i]);
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
        beavy_providers_[i]->setup();
        gmw_providers_[i]->setup();
        yao_providers_[i]->setup();
      }));
    }
    std::for_each(std::begin(futs), std::end(futs), [](auto& f) { f.get(); });
  }

  void run_gates_setup() {
    auto& gates_g = gate_registers_[garbler_i_]->get_gates();
    auto& gates_e = gate_registers_[evaluator_i_]->get_gates();
    auto fut_g = std::async(std::launch::async, [&gates_g] {
      for (auto& gate : gates_g) {
        if (gate->need_setup()) {
          gate->evaluate_setup();
        }
      }
    });
    auto fut_e = std::async(std::launch::async, [&gates_e] {
      for (auto& gate : gates_e) {
        if (gate->need_setup()) {
          gate->evaluate_setup();
        }
      }
    });
    fut_g.get();
    fut_e.get();
  }

  void run_gates_online() {
    auto& gates_g = gate_registers_[garbler_i_]->get_gates();
    auto& gates_e = gate_registers_[evaluator_i_]->get_gates();
    auto fut_g = std::async(std::launch::async, [&gates_g] {
      for (auto& gate : gates_g) {
        if (gate->need_online()) {
          gate->evaluate_online();
        }
      }
    });
    auto fut_e = std::async(std::launch::async, [&gates_e] {
      for (auto& gate : gates_e) {
        if (gate->need_online()) {
          gate->evaluate_online();
        }
      }
    });
    fut_g.get();
    fut_e.get();
  }

  MOTION::CircuitLoader circuit_loader_;
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
  std::array<std::unique_ptr<BEAVYProvider>, 2> beavy_providers_;
  std::array<std::unique_ptr<GMWProvider>, 2> gmw_providers_;
  std::array<std::unique_ptr<YaoProvider>, 2> yao_providers_;
  std::array<std::shared_ptr<MOTION::Logger>, 2> loggers_;
  std::array<MOTION::Statistics::RunTimeStats, 2> stats_;
};

template <typename T>
class YaoArithmeticGMWTensorTest : public YaoTensorTest {
 public:
  static std::vector<T> generate_inputs(const MOTION::tensor::TensorDimensions dims) {
    return MOTION::Helpers::RandomVector<T>(dims.get_data_size());
  }
  std::pair<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<T>>, MOTION::tensor::TensorCP>
  make_arithmetic_T_tensor_input_my(std::size_t party_id,
                                    const MOTION::tensor::TensorDimensions& dims) {
    auto& gp = *gmw_providers_.at(party_id);
    if constexpr (ENCRYPTO::bit_size_v<T> == 64) {
      return gp.make_arithmetic_64_tensor_input_my(dims);
    } else {
      static_assert(ENCRYPTO::bit_size_v<T> == 32);
      return gp.make_arithmetic_32_tensor_input_my(dims);
    }
  }
  MOTION::tensor::TensorCP make_arithmetic_T_tensor_input_other(
      std::size_t party_id, const MOTION::tensor::TensorDimensions& dims) {
    auto& gp = *gmw_providers_.at(party_id);
    if constexpr (ENCRYPTO::bit_size_v<T> == 64) {
      return gp.make_arithmetic_64_tensor_input_other(dims);
    } else {
      static_assert(ENCRYPTO::bit_size_v<T> == 32);
      return gp.make_arithmetic_32_tensor_input_other(dims);
    }
  }
  ENCRYPTO::ReusableFiberFuture<MOTION::IntegerValues<T>> make_arithmetic_T_tensor_output_my(
      std::size_t party_id, const MOTION::tensor::TensorCP& in) {
    auto& gp = *gmw_providers_.at(party_id);
    if constexpr (ENCRYPTO::bit_size_v<T> == 64) {
      return gp.make_arithmetic_64_tensor_output_my(in);
    } else {
      static_assert(ENCRYPTO::bit_size_v<T> == 32);
      return gp.make_arithmetic_32_tensor_output_my(in);
    }
  }
};

using integer_types = ::testing::Types<std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE(YaoArithmeticGMWTensorTest, integer_types);

TYPED_TEST(YaoArithmeticGMWTensorTest, ConversionToYao) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_0 = this->yao_providers_[0]->make_convert_from_arithmetic_gmw_tensor(tensor_in_0);
  auto tensor_1 = this->yao_providers_[1]->make_convert_from_arithmetic_gmw_tensor(tensor_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto yao_tensor_0 = std::dynamic_pointer_cast<const YaoTensor>(tensor_0);
  const auto yao_tensor_1 = std::dynamic_pointer_cast<const YaoTensor>(tensor_1);
  ASSERT_NE(yao_tensor_0, nullptr);
  ASSERT_NE(yao_tensor_1, nullptr);
  yao_tensor_0->wait_setup();
  yao_tensor_1->wait_online();

  const auto& R = this->yao_providers_[0]->get_global_offset();
  const auto& zero_keys = yao_tensor_0->get_keys();
  const auto& evaluator_keys = yao_tensor_1->get_keys();
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = input.size();
  ASSERT_EQ(zero_keys.size(), data_size * bit_size);
  ASSERT_EQ(evaluator_keys.size(), data_size * bit_size);
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = input.at(int_i);
    for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam>; ++bit_j) {
      auto idx = bit_j * data_size + int_i;
      if (value & (TypeParam(1) << bit_j)) {
        EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx) ^ R);
      } else {
        EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx));
      }
    }
  }
}

TYPED_TEST(YaoArithmeticGMWTensorTest, ConversionBoth) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto yao_tensor_0 = this->yao_providers_[0]->make_convert_from_arithmetic_gmw_tensor(tensor_in_0);
  auto yao_tensor_1 = this->yao_providers_[1]->make_convert_from_arithmetic_gmw_tensor(tensor_in_1);

  auto gmw_tensor_0 = this->yao_providers_[0]->make_convert_to_arithmetic_gmw_tensor(yao_tensor_0);
  auto gmw_tensor_1 = this->yao_providers_[1]->make_convert_to_arithmetic_gmw_tensor(yao_tensor_1);

  this->gmw_providers_[0]->make_arithmetic_tensor_output_other(gmw_tensor_0);
  auto output_future = this->make_arithmetic_T_tensor_output_my(1, gmw_tensor_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  auto output = output_future.get();

  ASSERT_EQ(output.size(), dims.get_data_size());
  ASSERT_EQ(input, output);
}

TYPED_TEST(YaoArithmeticGMWTensorTest, ConversionToBooleanGMW) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_yao_0 = this->yao_providers_[0]->make_convert_from_arithmetic_gmw_tensor(tensor_in_0);
  auto tensor_yao_1 = this->yao_providers_[1]->make_convert_from_arithmetic_gmw_tensor(tensor_in_1);
  auto tensor_0 = this->yao_providers_[0]->make_convert_to_boolean_gmw_tensor(tensor_yao_0);
  auto tensor_1 = this->yao_providers_[1]->make_convert_to_boolean_gmw_tensor(tensor_yao_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto bgmw_tensor_0 = std::dynamic_pointer_cast<const BooleanGMWTensor>(tensor_0);
  const auto bgmw_tensor_1 = std::dynamic_pointer_cast<const BooleanGMWTensor>(tensor_1);
  ASSERT_NE(bgmw_tensor_0, nullptr);
  ASSERT_NE(bgmw_tensor_1, nullptr);
  bgmw_tensor_0->wait_online();
  bgmw_tensor_1->wait_online();

  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = input.size();
  const auto& share_0 = bgmw_tensor_0->get_share();
  const auto& share_1 = bgmw_tensor_1->get_share();
  ASSERT_EQ(share_0.size(), bit_size);
  ASSERT_EQ(share_1.size(), bit_size);
  for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam>; ++bit_j) {
    ASSERT_EQ(share_0.at(bit_j).GetSize(), data_size);
    ASSERT_EQ(share_1.at(bit_j).GetSize(), data_size);
  }
  std::vector<ENCRYPTO::BitVector<>> plain_bits;
  plain_bits.reserve(bit_size);
  std::transform(std::begin(share_0), std::end(share_0), std::begin(share_1),
                 std::back_inserter(plain_bits),
                 [](const auto& bv0, const auto& bv1) { return bv0 ^ bv1; });
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = input.at(int_i);
    for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam>; ++bit_j) {
      const auto bit = plain_bits.at(bit_j).Get(int_i);
      const auto expected_bit = bool(value & (TypeParam(1) << bit_j));
      EXPECT_EQ(bit, expected_bit);
    }
  }
}

TYPED_TEST(YaoArithmeticGMWTensorTest, ConversionToBooleanGMWAndBack) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto yao_tensor_0 = this->yao_providers_[0]->make_convert_from_arithmetic_gmw_tensor(tensor_in_0);
  auto yao_tensor_1 = this->yao_providers_[1]->make_convert_from_arithmetic_gmw_tensor(tensor_in_1);

  auto bgmw_tensor_0 = this->yao_providers_[0]->make_convert_to_boolean_gmw_tensor(yao_tensor_0);
  auto bgmw_tensor_1 = this->yao_providers_[1]->make_convert_to_boolean_gmw_tensor(yao_tensor_1);

  auto agmw_tensor_0 =
      this->gmw_providers_[0]->make_convert_boolean_to_arithmetic_gmw_tensor(bgmw_tensor_0);
  auto agmw_tensor_1 =
      this->gmw_providers_[1]->make_convert_boolean_to_arithmetic_gmw_tensor(bgmw_tensor_1);

  this->gmw_providers_[0]->make_arithmetic_tensor_output_other(agmw_tensor_0);
  auto output_future = this->make_arithmetic_T_tensor_output_my(1, agmw_tensor_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  auto output = output_future.get();

  ASSERT_EQ(output.size(), dims.get_data_size());
  ASSERT_EQ(input, output);
}

TYPED_TEST(YaoArithmeticGMWTensorTest, ReLU) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_0 = this->yao_providers_[0]->make_convert_from_arithmetic_gmw_tensor(tensor_in_0);
  auto tensor_1 = this->yao_providers_[1]->make_convert_from_arithmetic_gmw_tensor(tensor_in_1);
  auto output_tensor_0 = this->yao_providers_[0]->make_tensor_relu_op(tensor_0);
  auto output_tensor_1 = this->yao_providers_[1]->make_tensor_relu_op(tensor_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto yao_tensor_0 = std::dynamic_pointer_cast<const YaoTensor>(output_tensor_0);
  const auto yao_tensor_1 = std::dynamic_pointer_cast<const YaoTensor>(output_tensor_1);
  ASSERT_NE(yao_tensor_0, nullptr);
  ASSERT_NE(yao_tensor_1, nullptr);
  yao_tensor_0->wait_setup();
  yao_tensor_1->wait_online();

  const auto& R = this->yao_providers_[0]->get_global_offset();
  const auto& zero_keys = yao_tensor_0->get_keys();
  const auto& evaluator_keys = yao_tensor_1->get_keys();
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = input.size();
  ASSERT_EQ(zero_keys.size(), data_size * bit_size);
  ASSERT_EQ(evaluator_keys.size(), data_size * bit_size);
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = input.at(int_i);
    bool zero = (value >> (ENCRYPTO::bit_size_v<TypeParam> - 1)) == 0;
    if (zero) {
      for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam>; ++bit_j) {
        auto idx = bit_j * data_size + int_i;
        EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx));
      }
    } else {
      for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam> - 1; ++bit_j) {
        auto idx = bit_j * data_size + int_i;
        if (value & (TypeParam(1) << bit_j)) {
          EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx) ^ R);
        } else {
          EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx));
        }
      }
      auto idx = (ENCRYPTO::bit_size_v<TypeParam> - 1) * data_size + int_i;
      EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx) ^ R);
    }
  }
}

TYPED_TEST(YaoArithmeticGMWTensorTest, ReLUInBooleanGMW) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_yao_0 = this->yao_providers_[0]->make_convert_from_arithmetic_gmw_tensor(tensor_in_0);
  auto tensor_yao_1 = this->yao_providers_[1]->make_convert_from_arithmetic_gmw_tensor(tensor_in_1);
  auto tensor_gmw_0 = this->yao_providers_[0]->make_convert_to_boolean_gmw_tensor(tensor_yao_0);
  auto tensor_gmw_1 = this->yao_providers_[1]->make_convert_to_boolean_gmw_tensor(tensor_yao_1);
  auto output_tensor_0 = this->gmw_providers_[0]->make_tensor_relu_op(tensor_gmw_0);
  auto output_tensor_1 = this->gmw_providers_[1]->make_tensor_relu_op(tensor_gmw_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto gmw_tensor_0 = std::dynamic_pointer_cast<const BooleanGMWTensor>(output_tensor_0);
  const auto gmw_tensor_1 = std::dynamic_pointer_cast<const BooleanGMWTensor>(output_tensor_1);
  ASSERT_NE(gmw_tensor_0, nullptr);
  ASSERT_NE(gmw_tensor_1, nullptr);
  gmw_tensor_0->wait_online();
  gmw_tensor_1->wait_online();

  const auto& share_0 = gmw_tensor_0->get_share();
  const auto& share_1 = gmw_tensor_1->get_share();
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = input.size();
  ASSERT_EQ(share_0.size(), bit_size);
  ASSERT_EQ(share_1.size(), bit_size);
  ASSERT_TRUE(std::all_of(std::begin(share_0), std::end(share_0),
                          [data_size](const auto& bv) { return bv.GetSize() == data_size; }));
  ASSERT_TRUE(std::all_of(std::begin(share_1), std::end(share_1),
                          [data_size](const auto& bv) { return bv.GetSize() == data_size; }));
  std::vector<ENCRYPTO::BitVector<>> plain_bits;
  std::transform(std::begin(share_0), std::end(share_0), std::begin(share_1),
                 std::back_inserter(plain_bits),
                 [](const auto& x, const auto& y) { return x ^ y; });
  const auto plain_ints = ENCRYPTO::ToVectorOutput<TypeParam>(plain_bits);
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = input.at(int_i);
    const auto msb = bool(value >> (ENCRYPTO::bit_size_v<TypeParam> - 1));
    if (msb) {
      EXPECT_EQ(plain_ints.at(int_i), 0);
    } else {
      EXPECT_EQ(plain_ints.at(int_i), value);
    }
  }
}

TYPED_TEST(YaoArithmeticGMWTensorTest, ReLUInBooleanXArithmeticGMW) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_yao_0 = this->yao_providers_[0]->make_convert_from_arithmetic_gmw_tensor(tensor_in_0);
  auto tensor_yao_1 = this->yao_providers_[1]->make_convert_from_arithmetic_gmw_tensor(tensor_in_1);
  auto tensor_gmw_0 = this->yao_providers_[0]->make_convert_to_boolean_gmw_tensor(tensor_yao_0);
  auto tensor_gmw_1 = this->yao_providers_[1]->make_convert_to_boolean_gmw_tensor(tensor_yao_1);
  auto output_tensor_0 = this->gmw_providers_[0]->make_tensor_relu_op(tensor_gmw_0, tensor_in_0);
  auto output_tensor_1 = this->gmw_providers_[1]->make_tensor_relu_op(tensor_gmw_1, tensor_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto gmw_tensor_0 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(output_tensor_0);
  const auto gmw_tensor_1 =
      std::dynamic_pointer_cast<const ArithmeticGMWTensor<TypeParam>>(output_tensor_1);
  ASSERT_NE(gmw_tensor_0, nullptr);
  ASSERT_NE(gmw_tensor_1, nullptr);
  gmw_tensor_0->wait_online();
  gmw_tensor_1->wait_online();

  const auto& share_0 = gmw_tensor_0->get_share();
  const auto& share_1 = gmw_tensor_1->get_share();
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = input.size();
  ASSERT_EQ(share_0.size(), data_size);
  ASSERT_EQ(share_1.size(), data_size);
  const auto plain_ints = MOTION::Helpers::AddVectors(share_0, share_1);
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = input.at(int_i);
    const auto msb = bool(value >> (ENCRYPTO::bit_size_v<TypeParam> - 1));
    if (msb) {
      EXPECT_EQ(plain_ints.at(int_i), 0);
    } else {
      EXPECT_EQ(plain_ints.at(int_i), value);
    }
  }
}

TYPED_TEST(YaoArithmeticGMWTensorTest, MaxPoolSimple) {
  const MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 2, .width_ = 2};
  const MOTION::tensor::TensorDimensions out_dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 1, .width_ = 1};
  const MOTION::tensor::MaxPoolOp maxpool_op = {.input_shape_ = {1, 2, 2},
                                                .output_shape_ = {1, 1, 1},
                                                .kernel_shape_ = {2, 2},
                                                .strides_ = {1, 1}};
  ASSERT_TRUE(maxpool_op.verify());
  const std::vector<TypeParam> input = {13, 42, 47, 37};

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_0 = this->yao_providers_[0]->make_convert_from_arithmetic_gmw_tensor(tensor_in_0);
  auto tensor_1 = this->yao_providers_[1]->make_convert_from_arithmetic_gmw_tensor(tensor_in_1);
  auto output_tensor_0 =
      this->yao_providers_[0]->make_tensor_maxpool_op(maxpool_op, tensor_0);
  auto output_tensor_1 =
      this->yao_providers_[1]->make_tensor_maxpool_op(maxpool_op, tensor_1);

  ASSERT_EQ(output_tensor_0->get_dimensions(), out_dims);
  ASSERT_EQ(output_tensor_1->get_dimensions(), out_dims);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto yao_tensor_0 = std::dynamic_pointer_cast<const YaoTensor>(output_tensor_0);
  const auto yao_tensor_1 = std::dynamic_pointer_cast<const YaoTensor>(output_tensor_1);
  ASSERT_NE(yao_tensor_0, nullptr);
  ASSERT_NE(yao_tensor_1, nullptr);
  yao_tensor_0->wait_setup();
  yao_tensor_1->wait_online();

  const auto& R = this->yao_providers_[0]->get_global_offset();
  const auto& zero_keys = yao_tensor_0->get_keys();
  const auto& evaluator_keys = yao_tensor_1->get_keys();
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = maxpool_op.compute_output_size();
  ASSERT_EQ(zero_keys.size(), data_size * bit_size);
  ASSERT_EQ(evaluator_keys.size(), data_size * bit_size);
  const TypeParam value = 47;
  for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam> - 1; ++bit_j) {
    if (value & (TypeParam(1) << bit_j)) {
      EXPECT_EQ(evaluator_keys.at(bit_j), zero_keys.at(bit_j) ^ R);
    } else {
      EXPECT_EQ(evaluator_keys.at(bit_j), zero_keys.at(bit_j));
    }
  }
}

TYPED_TEST(YaoArithmeticGMWTensorTest, MaxPool) {
  const MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 4, .width_ = 4};
  const MOTION::tensor::TensorDimensions out_dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 3, .width_ = 2};
  const MOTION::tensor::MaxPoolOp maxpool_op = {.input_shape_ = {1, 4, 4},
                                                .output_shape_ = {1, 3, 2},
                                                .kernel_shape_ = {2, 2},
                                                .strides_ = {1, 2}};
  ASSERT_TRUE(maxpool_op.verify());

  // clang-format off
  // const std::vector<TypeParam> input = {
  //   629, 499, 147, 593,
  //   335, 313, 191, 159,
  //   829, 569, 975, 846,
  //   758, 466, 868, 403};
  // const std::vector<TypeParam> expected_output = {
  //   629, 593,
  //   829, 975,
  //   829, 975,
  // };
  std::vector<TypeParam> input;
  std::vector<TypeParam> expected_output;
  if constexpr (std::is_same_v<TypeParam, std::uint64_t>) {
    input = {
       9752516871661491360u,  4662446014583943733u, 11447746383552160793u,  6355249606212339500u,
      18175355885135292618u,  7196435132770689189u,  2104876926157817893u, 14012399909055774597u,
      18122604083447938486u,  2509168573361965776u, 17592664428622712039u,  2487988269217325321u,
       7153114465142520673u, 16795418668493668102u, 11793396511977302258u,  1400963290576875662u};
    expected_output = {
      7196435132770689189u, 6355249606212339500u,
      7196435132770689189u, 2487988269217325321u,
      7153114465142520673u, 2487988269217325321u,
    };
  } else if constexpr (std::is_same_v<TypeParam, std::uint32_t>) {
    input = {
      0xfc9af2d7u, 0x80a495efu, 0x521b859cu, 0xf6c040d6u,
      0x220aee7du, 0x100cb900u, 0x33541084u, 0xa8971444u,
      0xb1f10b19u, 0x7ef65ca4u, 0x121716d5u, 0x997ffefau,
      0x84a54d02u, 0xb9d7c593u, 0x369fedbfu, 0xd79ed530u
    };
    expected_output = {
      0x220aee7du, 0x521b859cu,
      0x7ef65ca4u, 0x33541084u,
      0x7ef65ca4u, 0x369fedbfu,
    };
  }
  // clang-format on

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_0 = this->yao_providers_[0]->make_convert_from_arithmetic_gmw_tensor(tensor_in_0);
  auto tensor_1 = this->yao_providers_[1]->make_convert_from_arithmetic_gmw_tensor(tensor_in_1);
  auto output_tensor_0 =
      this->yao_providers_[0]->make_tensor_maxpool_op(maxpool_op, tensor_0);
  auto output_tensor_1 =
      this->yao_providers_[1]->make_tensor_maxpool_op(maxpool_op, tensor_1);
  auto gmw_output_tensor_0 =
      this->yao_providers_[0]->make_convert_to_arithmetic_gmw_tensor(output_tensor_0);
  auto gmw_output_tensor_1 =
      this->yao_providers_[1]->make_convert_to_arithmetic_gmw_tensor(output_tensor_1);
  this->gmw_providers_[0]->make_arithmetic_tensor_output_other(gmw_output_tensor_0);
  auto output_future = this->make_arithmetic_T_tensor_output_my(1, gmw_output_tensor_1);

  ASSERT_EQ(output_tensor_0->get_dimensions(), out_dims);
  ASSERT_EQ(output_tensor_1->get_dimensions(), out_dims);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto yao_tensor_0 = std::dynamic_pointer_cast<const YaoTensor>(output_tensor_0);
  const auto yao_tensor_1 = std::dynamic_pointer_cast<const YaoTensor>(output_tensor_1);
  ASSERT_NE(yao_tensor_0, nullptr);
  ASSERT_NE(yao_tensor_1, nullptr);
  yao_tensor_0->wait_setup();
  yao_tensor_1->wait_online();

  const auto& R = this->yao_providers_[0]->get_global_offset();
  const auto& zero_keys = yao_tensor_0->get_keys();
  const auto& evaluator_keys = yao_tensor_1->get_keys();
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = maxpool_op.compute_output_size();
  ASSERT_EQ(zero_keys.size(), data_size * bit_size);
  ASSERT_EQ(evaluator_keys.size(), data_size * bit_size);
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = expected_output.at(int_i);
    for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam> - 1; ++bit_j) {
      auto idx = bit_j * data_size + int_i;
      if (value & (TypeParam(1) << bit_j)) {
        EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx) ^ R);
      } else {
        EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx));
      }
    }
  }

  const auto output = output_future.get();
  EXPECT_EQ(output, expected_output);
}

template <typename T>
class YaoArithmeticBEAVYTensorTest : public YaoTensorTest {
 public:
  static std::vector<T> generate_inputs(const MOTION::tensor::TensorDimensions dims) {
    return MOTION::Helpers::RandomVector<T>(dims.get_data_size());
  }
  std::pair<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<T>>, MOTION::tensor::TensorCP>
  make_arithmetic_T_tensor_input_my(std::size_t party_id,
                                    const MOTION::tensor::TensorDimensions& dims) {
    auto& bp = *beavy_providers_.at(party_id);
    if constexpr (ENCRYPTO::bit_size_v<T> == 64) {
      return bp.make_arithmetic_64_tensor_input_my(dims);
    } else {
      static_assert(ENCRYPTO::bit_size_v<T> == 32);
      return bp.make_arithmetic_32_tensor_input_my(dims);
    }
  }
  MOTION::tensor::TensorCP make_arithmetic_T_tensor_input_other(
      std::size_t party_id, const MOTION::tensor::TensorDimensions& dims) {
    auto& bp = *beavy_providers_.at(party_id);
    if constexpr (ENCRYPTO::bit_size_v<T> == 64) {
      return bp.make_arithmetic_64_tensor_input_other(dims);
    } else {
      static_assert(ENCRYPTO::bit_size_v<T> == 32);
      return bp.make_arithmetic_32_tensor_input_other(dims);
    }
  }
  ENCRYPTO::ReusableFiberFuture<MOTION::IntegerValues<T>> make_arithmetic_T_tensor_output_my(
      std::size_t party_id, const MOTION::tensor::TensorCP& in) {
    auto& bp = *beavy_providers_.at(party_id);
    if constexpr (ENCRYPTO::bit_size_v<T> == 64) {
      return bp.make_arithmetic_64_tensor_output_my(in);
    } else {
      static_assert(ENCRYPTO::bit_size_v<T> == 32);
      return bp.make_arithmetic_32_tensor_output_my(in);
    }
  }
};

using integer_types_beavy = ::testing::Types<std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE(YaoArithmeticBEAVYTensorTest, integer_types_beavy);

TYPED_TEST(YaoArithmeticBEAVYTensorTest, ConversionToYao) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_0 = this->yao_providers_[0]->make_convert_from_arithmetic_beavy_tensor(tensor_in_0);
  auto tensor_1 = this->yao_providers_[1]->make_convert_from_arithmetic_beavy_tensor(tensor_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto yao_tensor_0 = std::dynamic_pointer_cast<const YaoTensor>(tensor_0);
  const auto yao_tensor_1 = std::dynamic_pointer_cast<const YaoTensor>(tensor_1);
  ASSERT_NE(yao_tensor_0, nullptr);
  ASSERT_NE(yao_tensor_1, nullptr);
  yao_tensor_0->wait_setup();
  yao_tensor_1->wait_online();

  const auto& R = this->yao_providers_[0]->get_global_offset();
  const auto& zero_keys = yao_tensor_0->get_keys();
  const auto& evaluator_keys = yao_tensor_1->get_keys();
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = input.size();
  ASSERT_EQ(zero_keys.size(), data_size * bit_size);
  ASSERT_EQ(evaluator_keys.size(), data_size * bit_size);
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = input.at(int_i);
    for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam>; ++bit_j) {
      auto idx = bit_j * data_size + int_i;
      if (value & (TypeParam(1) << bit_j)) {
        EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx) ^ R);
      } else {
        EXPECT_EQ(evaluator_keys.at(idx), zero_keys.at(idx));
      }
    }
  }
}

TYPED_TEST(YaoArithmeticBEAVYTensorTest, ConversionBoth) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto yao_tensor_0 =
      this->yao_providers_[0]->make_convert_from_arithmetic_beavy_tensor(tensor_in_0);
  auto yao_tensor_1 =
      this->yao_providers_[1]->make_convert_from_arithmetic_beavy_tensor(tensor_in_1);

  auto beavy_tensor_0 =
      this->yao_providers_[0]->make_convert_to_arithmetic_beavy_tensor(yao_tensor_0);
  auto beavy_tensor_1 =
      this->yao_providers_[1]->make_convert_to_arithmetic_beavy_tensor(yao_tensor_1);

  this->beavy_providers_[0]->make_arithmetic_tensor_output_other(beavy_tensor_0);
  auto output_future = this->make_arithmetic_T_tensor_output_my(1, beavy_tensor_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  auto output = output_future.get();

  ASSERT_EQ(output.size(), dims.get_data_size());
  ASSERT_EQ(input, output);
}

TYPED_TEST(YaoArithmeticBEAVYTensorTest, ConversionToBooleanBEAVY) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_yao_0 =
      this->yao_providers_[0]->make_convert_from_arithmetic_beavy_tensor(tensor_in_0);
  auto tensor_yao_1 =
      this->yao_providers_[1]->make_convert_from_arithmetic_beavy_tensor(tensor_in_1);
  auto tensor_0 = this->yao_providers_[0]->make_convert_to_boolean_beavy_tensor(tensor_yao_0);
  auto tensor_1 = this->yao_providers_[1]->make_convert_to_boolean_beavy_tensor(tensor_yao_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto bbeavy_tensor_0 = std::dynamic_pointer_cast<const BooleanBEAVYTensor>(tensor_0);
  const auto bbeavy_tensor_1 = std::dynamic_pointer_cast<const BooleanBEAVYTensor>(tensor_1);
  ASSERT_NE(bbeavy_tensor_0, nullptr);
  ASSERT_NE(bbeavy_tensor_1, nullptr);
  bbeavy_tensor_0->wait_online();
  bbeavy_tensor_1->wait_online();

  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = input.size();
  const auto& pshare_0 = bbeavy_tensor_0->get_public_share();
  const auto& pshare_1 = bbeavy_tensor_1->get_public_share();
  const auto& sshare_0 = bbeavy_tensor_0->get_secret_share();
  const auto& sshare_1 = bbeavy_tensor_1->get_secret_share();
  ASSERT_EQ(pshare_0.size(), bit_size);
  ASSERT_EQ(pshare_0, pshare_1);
  ASSERT_EQ(sshare_0.size(), bit_size);
  ASSERT_EQ(sshare_1.size(), bit_size);
  for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam>; ++bit_j) {
    ASSERT_EQ(pshare_0.at(bit_j).GetSize(), data_size);
    ASSERT_EQ(sshare_0.at(bit_j).GetSize(), data_size);
    ASSERT_EQ(sshare_1.at(bit_j).GetSize(), data_size);
  }
  std::vector<ENCRYPTO::BitVector<>> plain_bits(bit_size);
  for (std::size_t bit_j = 0; bit_j < bit_size; ++bit_j) {
    plain_bits.at(bit_j) = pshare_0.at(bit_j) ^ sshare_0.at(bit_j) ^ sshare_1.at(bit_j);
  }
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = input.at(int_i);
    for (std::size_t bit_j = 0; bit_j < ENCRYPTO::bit_size_v<TypeParam>; ++bit_j) {
      const auto bit = plain_bits.at(bit_j).Get(int_i);
      const auto expected_bit = bool(value & (TypeParam(1) << bit_j));
      EXPECT_EQ(bit, expected_bit);
    }
  }
}

TYPED_TEST(YaoArithmeticBEAVYTensorTest, ConversionToBooleanBEAVYAndBack) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto yao_tensor_0 =
      this->yao_providers_[0]->make_convert_from_arithmetic_beavy_tensor(tensor_in_0);
  auto yao_tensor_1 =
      this->yao_providers_[1]->make_convert_from_arithmetic_beavy_tensor(tensor_in_1);

  auto bbeavy_tensor_0 =
      this->yao_providers_[0]->make_convert_to_boolean_beavy_tensor(yao_tensor_0);
  auto bbeavy_tensor_1 =
      this->yao_providers_[1]->make_convert_to_boolean_beavy_tensor(yao_tensor_1);

  auto abeavy_tensor_0 =
      this->beavy_providers_[0]->make_convert_boolean_to_arithmetic_beavy_tensor(bbeavy_tensor_0);
  auto abeavy_tensor_1 =
      this->beavy_providers_[1]->make_convert_boolean_to_arithmetic_beavy_tensor(bbeavy_tensor_1);

  this->beavy_providers_[0]->make_arithmetic_tensor_output_other(abeavy_tensor_0);
  auto output_future = this->make_arithmetic_T_tensor_output_my(1, abeavy_tensor_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  auto output = output_future.get();

  ASSERT_EQ(output.size(), dims.get_data_size());
  ASSERT_EQ(input, output);
}

TYPED_TEST(YaoArithmeticBEAVYTensorTest, ReLUInBooleanBEAVY) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_yao_0 =
      this->yao_providers_[0]->make_convert_from_arithmetic_beavy_tensor(tensor_in_0);
  auto tensor_yao_1 =
      this->yao_providers_[1]->make_convert_from_arithmetic_beavy_tensor(tensor_in_1);
  auto tensor_beavy_0 = this->yao_providers_[0]->make_convert_to_boolean_beavy_tensor(tensor_yao_0);
  auto tensor_beavy_1 = this->yao_providers_[1]->make_convert_to_boolean_beavy_tensor(tensor_yao_1);
  auto output_tensor_0 = this->beavy_providers_[0]->make_tensor_relu_op(tensor_beavy_0);
  auto output_tensor_1 = this->beavy_providers_[1]->make_tensor_relu_op(tensor_beavy_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto beavy_tensor_0 = std::dynamic_pointer_cast<const BooleanBEAVYTensor>(output_tensor_0);
  const auto beavy_tensor_1 = std::dynamic_pointer_cast<const BooleanBEAVYTensor>(output_tensor_1);
  ASSERT_NE(beavy_tensor_0, nullptr);
  ASSERT_NE(beavy_tensor_1, nullptr);
  beavy_tensor_0->wait_online();
  beavy_tensor_1->wait_online();

  const auto& pshare_0 = beavy_tensor_0->get_public_share();
  const auto& pshare_1 = beavy_tensor_1->get_public_share();
  const auto& sshare_0 = beavy_tensor_0->get_secret_share();
  const auto& sshare_1 = beavy_tensor_1->get_secret_share();
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = input.size();
  ASSERT_EQ(pshare_0.size(), bit_size);
  ASSERT_EQ(pshare_0, pshare_1);
  ASSERT_EQ(sshare_0.size(), bit_size);
  ASSERT_EQ(sshare_1.size(), bit_size);
  for (const auto& v : {pshare_0, sshare_0, sshare_1}) {
    ASSERT_TRUE(std::all_of(std::begin(v), std::end(v),
                            [data_size](const auto& bv) { return bv.GetSize() == data_size; }));
  }
  std::vector<ENCRYPTO::BitVector<>> plain_bits(bit_size);
  for (std::size_t bit_j = 0; bit_j < bit_size; ++bit_j) {
    plain_bits.at(bit_j) = pshare_0.at(bit_j) ^ sshare_0.at(bit_j) ^ sshare_1.at(bit_j);
  }
  const auto plain_ints = ENCRYPTO::ToVectorOutput<TypeParam>(plain_bits);
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = input.at(int_i);
    const auto msb = bool(value >> (ENCRYPTO::bit_size_v<TypeParam> - 1));
    if (msb) {
      EXPECT_EQ(plain_ints.at(int_i), 0);
    } else {
      EXPECT_EQ(plain_ints.at(int_i), value);
    }
  }
}

TYPED_TEST(YaoArithmeticBEAVYTensorTest, DISABLED_ReLUInBooleanXArithmeticBEAVY) {
  MOTION::tensor::TensorDimensions dims = {
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const auto input = this->generate_inputs(dims);

  auto [input_promise, tensor_in_0] = this->make_arithmetic_T_tensor_input_my(0, dims);
  auto tensor_in_1 = this->make_arithmetic_T_tensor_input_other(1, dims);

  auto tensor_yao_0 =
      this->yao_providers_[0]->make_convert_from_arithmetic_beavy_tensor(tensor_in_0);
  auto tensor_yao_1 =
      this->yao_providers_[1]->make_convert_from_arithmetic_beavy_tensor(tensor_in_1);
  auto tensor_beavy_0 = this->yao_providers_[0]->make_convert_to_boolean_beavy_tensor(tensor_yao_0);
  auto tensor_beavy_1 = this->yao_providers_[1]->make_convert_to_boolean_beavy_tensor(tensor_yao_1);
  auto output_tensor_0 =
      this->beavy_providers_[0]->make_tensor_relu_op(tensor_beavy_0, tensor_in_0);
  auto output_tensor_1 =
      this->beavy_providers_[1]->make_tensor_relu_op(tensor_beavy_1, tensor_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(input);
  this->run_gates_online();

  const auto beavy_tensor_0 =
      std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<TypeParam>>(output_tensor_0);
  const auto beavy_tensor_1 =
      std::dynamic_pointer_cast<const ArithmeticBEAVYTensor<TypeParam>>(output_tensor_1);
  ASSERT_NE(beavy_tensor_0, nullptr);
  ASSERT_NE(beavy_tensor_1, nullptr);
  beavy_tensor_0->wait_online();
  beavy_tensor_1->wait_online();

  const auto& pshare_0 = beavy_tensor_0->get_public_share();
  const auto& pshare_1 = beavy_tensor_1->get_public_share();
  const auto& sshare_0 = beavy_tensor_0->get_secret_share();
  const auto& sshare_1 = beavy_tensor_1->get_secret_share();
  constexpr auto bit_size = ENCRYPTO::bit_size_v<TypeParam>;
  const auto data_size = input.size();
  ASSERT_EQ(pshare_0.size(), data_size);
  ASSERT_EQ(pshare_0, pshare_1);
  ASSERT_EQ(sshare_0.size(), data_size);
  ASSERT_EQ(sshare_1.size(), data_size);
  const auto plain_ints =
      MOTION::Helpers::SubVectors(pshare_0, MOTION::Helpers::AddVectors(sshare_0, sshare_1));
  for (std::size_t int_i = 0; int_i < data_size; ++int_i) {
    const auto value = input.at(int_i);
    const auto msb = bool(value >> (ENCRYPTO::bit_size_v<TypeParam> - 1));
    if (msb) {
      EXPECT_EQ(plain_ints.at(int_i), 0);
    } else {
      EXPECT_EQ(plain_ints.at(int_i), value);
    }
  }
}
