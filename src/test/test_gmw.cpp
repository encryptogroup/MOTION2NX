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
#include "crypto/multiplication_triple/mt_provider.h"
#include "crypto/multiplication_triple/sb_provider.h"
#include "crypto/multiplication_triple/sp_provider.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "gate/new_gate.h"
#include "protocols/gmw/gmw_provider.h"
#include "protocols/gmw/wire.h"
#include "protocols/plain/wire.h"
#include "statistics/run_time_stats.h"
#include "utility/helpers.h"
#include "utility/logger.h"

using namespace MOTION::proto::gmw;
namespace plain = MOTION::proto::plain;

class GMWTest : public ::testing::Test {
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
      mt_providers_[i] = std::make_unique<MOTION::MTProviderFromOTs>(
          i, 2, true, *arithmetic_provider_managers_[i], *ot_provider_managers_[i], stats_[i],
          nullptr);
      sp_providers_[i] = std::make_unique<MOTION::SPProviderFromOTs>(
          ot_provider_managers_[i]->get_providers(), i, stats_[i], nullptr);
      sb_providers_[i] = std::make_unique<MOTION::TwoPartySBProvider>(
          *comm_layers_[i], ot_provider_managers_[i]->get_provider(1 - i), stats_[i], nullptr);
      gate_registers_[i] = std::make_unique<MOTION::GateRegister>();
      gmw_providers_[i] = std::make_unique<GMWProvider>(
          *comm_layers_[i], *gate_registers_[i], circuit_loader_, *motion_base_providers_[i],
          *ot_provider_managers_[i], *mt_providers_[i], *sp_providers_[i], *sb_providers_[i],
          loggers_[i]);
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

  MOTION::CircuitLoader circuit_loader_;
  std::vector<std::unique_ptr<MOTION::Communication::CommunicationLayer>> comm_layers_;
  std::array<std::unique_ptr<MOTION::BaseOTProvider>, 2> base_ot_providers_;
  std::array<std::unique_ptr<MOTION::Crypto::MotionBaseProvider>, 2> motion_base_providers_;
  std::array<std::unique_ptr<ENCRYPTO::ObliviousTransfer::OTProviderManager>, 2>
      ot_provider_managers_;
  std::array<std::unique_ptr<MOTION::ArithmeticProviderManager>, 2> arithmetic_provider_managers_;
  std::array<std::unique_ptr<MOTION::MTProvider>, 2> mt_providers_;
  std::array<std::unique_ptr<MOTION::SPProvider>, 2> sp_providers_;
  std::array<std::unique_ptr<MOTION::SBProvider>, 2> sb_providers_;
  std::array<std::unique_ptr<MOTION::GateRegister>, 2> gate_registers_;
  std::array<std::unique_ptr<GMWProvider>, 2> gmw_providers_;
  std::array<std::shared_ptr<MOTION::Logger>, 2> loggers_;
  std::array<MOTION::Statistics::RunTimeStats, 2> stats_;
};

class BooleanGMWTest : public GMWTest {
 public:
  static std::vector<ENCRYPTO::BitVector<>> generate_inputs(std::size_t num_wires,
                                                            std::size_t num_simd) {
    std::vector<ENCRYPTO::BitVector<>> inputs;
    std::generate_n(std::back_inserter(inputs), num_wires,
                    [num_simd] { return ENCRYPTO::BitVector<>::Random(num_simd); });
    return inputs;
  }
};

TEST_F(BooleanGMWTest, Input) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs_a = generate_inputs(num_wires, num_simd);
  const auto inputs_b = generate_inputs(num_wires, num_simd);

  // input of party 0
  auto [input_a_promise, wires_a_in_0] =
      gmw_providers_[0]->make_boolean_input_gate_my(0, num_wires, num_simd);
  auto wires_a_in_1 = gmw_providers_[1]->make_boolean_input_gate_other(0, num_wires, num_simd);

  // input of party 1
  auto wires_b_in_0 = gmw_providers_[0]->make_boolean_input_gate_other(1, num_wires, num_simd);
  auto [input_b_promise, wires_b_in_1] =
      gmw_providers_[1]->make_boolean_input_gate_my(1, num_wires, num_simd);

  ASSERT_EQ(wires_a_in_0.size(), num_wires);
  ASSERT_EQ(wires_a_in_1.size(), num_wires);
  ASSERT_EQ(wires_b_in_0.size(), num_wires);
  ASSERT_EQ(wires_b_in_1.size(), num_wires);
  auto check_num_simd_f = [num_simd](const auto& w) { return w->get_num_simd() == num_simd; };
  ASSERT_TRUE(std::all_of(std::begin(wires_a_in_0), std::end(wires_a_in_0), check_num_simd_f));
  ASSERT_TRUE(std::all_of(std::begin(wires_a_in_1), std::end(wires_a_in_1), check_num_simd_f));
  ASSERT_TRUE(std::all_of(std::begin(wires_b_in_0), std::end(wires_b_in_0), check_num_simd_f));
  ASSERT_TRUE(std::all_of(std::begin(wires_b_in_1), std::end(wires_b_in_1), check_num_simd_f));

  run_setup();
  run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  run_gates_online();

  // check wire values
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& input_a_bits = inputs_a.at(wire_i);
    const auto& input_b_bits = inputs_b.at(wire_i);
    const auto wire_a0 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_a_in_0.at(wire_i));
    const auto wire_a1 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_a_in_1.at(wire_i));
    const auto wire_b0 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_b_in_0.at(wire_i));
    const auto wire_b1 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_b_in_1.at(wire_i));
    wire_a0->wait_online();
    wire_a1->wait_online();
    wire_b0->wait_online();
    wire_b1->wait_online();
    const auto& share_a0 = wire_a0->get_share();
    const auto& share_a1 = wire_a1->get_share();
    const auto& share_b0 = wire_b0->get_share();
    const auto& share_b1 = wire_b1->get_share();
    ASSERT_EQ(share_a0.GetSize(), num_simd);
    ASSERT_EQ(share_a1.GetSize(), num_simd);
    ASSERT_EQ(share_b0.GetSize(), num_simd);
    ASSERT_EQ(share_b1.GetSize(), num_simd);
    ASSERT_EQ(input_a_bits, share_a0 ^ share_a1);
    ASSERT_EQ(input_b_bits, share_b0 ^ share_b1);
  }
}

TEST_F(BooleanGMWTest, OutputSingle) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs_a = generate_inputs(num_wires, num_simd);
  const auto inputs_b = generate_inputs(num_wires, num_simd);

  // input of party 0
  auto [input_a_promise, wires_a_in_0] =
      gmw_providers_[0]->make_boolean_input_gate_my(0, num_wires, num_simd);
  auto wires_a_in_1 = gmw_providers_[1]->make_boolean_input_gate_other(0, num_wires, num_simd);

  // input of party 1
  auto wires_b_in_0 = gmw_providers_[0]->make_boolean_input_gate_other(1, num_wires, num_simd);
  auto [input_b_promise, wires_b_in_1] =
      gmw_providers_[1]->make_boolean_input_gate_my(1, num_wires, num_simd);

  auto output_future_a_0 = gmw_providers_[0]->make_boolean_output_gate_my(0, wires_a_in_0);
  gmw_providers_[1]->make_boolean_output_gate_other(0, wires_a_in_1);
  gmw_providers_[0]->make_boolean_output_gate_other(1, wires_b_in_0);
  auto output_future_b_1 = gmw_providers_[1]->make_boolean_output_gate_my(1, wires_b_in_1);

  run_setup();
  run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  run_gates_online();

  auto outputs_a = output_future_a_0.get();
  auto outputs_b = output_future_b_1.get();

  ASSERT_EQ(outputs_a.size(), num_wires);
  ASSERT_EQ(outputs_b.size(), num_wires);
  auto check_num_simd_f = [num_simd](const auto& bv) { return bv.GetSize() == num_simd; };
  ASSERT_TRUE(std::all_of(std::begin(outputs_a), std::end(outputs_a), check_num_simd_f));
  ASSERT_TRUE(std::all_of(std::begin(outputs_b), std::end(outputs_b), check_num_simd_f));

  // check outputs values
  ASSERT_EQ(inputs_a, outputs_a);
  ASSERT_EQ(inputs_b, outputs_b);
}

TEST_F(BooleanGMWTest, OutputBoth) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs_a = generate_inputs(num_wires, num_simd);
  const auto inputs_b = generate_inputs(num_wires, num_simd);

  // input of party 0
  auto [input_a_promise, wires_a_in_0] =
      gmw_providers_[0]->make_boolean_input_gate_my(0, num_wires, num_simd);
  auto wires_a_in_1 = gmw_providers_[1]->make_boolean_input_gate_other(0, num_wires, num_simd);

  // input of party 1
  auto wires_b_in_0 = gmw_providers_[0]->make_boolean_input_gate_other(1, num_wires, num_simd);
  auto [input_b_promise, wires_b_in_1] =
      gmw_providers_[1]->make_boolean_input_gate_my(1, num_wires, num_simd);

  auto output_future_a_0 =
      gmw_providers_[0]->make_boolean_output_gate_my(MOTION::ALL_PARTIES, wires_a_in_0);
  auto output_future_a_1 =
      gmw_providers_[1]->make_boolean_output_gate_my(MOTION::ALL_PARTIES, wires_a_in_1);
  auto output_future_b_0 =
      gmw_providers_[0]->make_boolean_output_gate_my(MOTION::ALL_PARTIES, wires_b_in_0);
  auto output_future_b_1 =
      gmw_providers_[1]->make_boolean_output_gate_my(MOTION::ALL_PARTIES, wires_b_in_1);

  run_setup();
  run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  run_gates_online();

  auto outputs_a_0 = output_future_a_0.get();
  auto outputs_a_1 = output_future_a_1.get();
  auto outputs_b_0 = output_future_b_0.get();
  auto outputs_b_1 = output_future_b_1.get();

  ASSERT_EQ(outputs_a_0.size(), num_wires);
  ASSERT_EQ(outputs_a_1.size(), num_wires);
  ASSERT_EQ(outputs_b_0.size(), num_wires);
  ASSERT_EQ(outputs_b_1.size(), num_wires);
  auto check_num_simd_f = [num_simd](const auto& bv) { return bv.GetSize() == num_simd; };
  ASSERT_TRUE(std::all_of(std::begin(outputs_a_0), std::end(outputs_a_0), check_num_simd_f));
  ASSERT_TRUE(std::all_of(std::begin(outputs_a_1), std::end(outputs_a_1), check_num_simd_f));
  ASSERT_TRUE(std::all_of(std::begin(outputs_b_0), std::end(outputs_b_0), check_num_simd_f));
  ASSERT_TRUE(std::all_of(std::begin(outputs_b_1), std::end(outputs_b_1), check_num_simd_f));

  // check outputs values
  ASSERT_EQ(inputs_a, outputs_a_0);
  ASSERT_EQ(inputs_a, outputs_a_1);
  ASSERT_EQ(inputs_b, outputs_b_0);
  ASSERT_EQ(inputs_b, outputs_b_1);
}

TEST_F(BooleanGMWTest, INV) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs = generate_inputs(num_wires, num_simd);
  MOTION::BitValues expected_output;
  std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(expected_output),
                 [](const auto& bv) { return ~bv; });

  auto [input_promise, wires_0_in] =
      gmw_providers_[0]->make_boolean_input_gate_my(0, num_wires, num_simd);
  auto wires_1_in = gmw_providers_[1]->make_boolean_input_gate_other(0, num_wires, num_simd);
  auto wires_0_out =
      gmw_providers_[0]->make_unary_gate(ENCRYPTO::PrimitiveOperationType::INV, wires_0_in);
  auto wires_1_out =
      gmw_providers_[1]->make_unary_gate(ENCRYPTO::PrimitiveOperationType::INV, wires_1_in);

  run_setup();
  run_gates_setup();
  input_promise.set_value(inputs);
  run_gates_online();

  // check wire values
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& expected_output_bits = expected_output.at(wire_i);
    const auto wire_0 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_0_out.at(wire_i));
    const auto wire_1 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_1_out.at(wire_i));
    wire_0->wait_online();
    wire_1->wait_online();
    const auto& share_0 = wire_0->get_share();
    const auto& share_1 = wire_1->get_share();
    ASSERT_EQ(share_0.GetSize(), num_simd);
    ASSERT_EQ(share_1.GetSize(), num_simd);
    ASSERT_EQ(expected_output_bits, share_0 ^ share_1);
  }
}

TEST_F(BooleanGMWTest, XOR) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs_a = generate_inputs(num_wires, num_simd);
  const auto inputs_b = generate_inputs(num_wires, num_simd);
  MOTION::BitValues expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output),
                 [](const auto& bv_a, const auto& bv_b) { return bv_a ^ bv_b; });

  auto [input_a_promise, wires_0_in_a] =
      gmw_providers_[0]->make_boolean_input_gate_my(0, num_wires, num_simd);
  auto wires_1_in_a = gmw_providers_[1]->make_boolean_input_gate_other(0, num_wires, num_simd);
  auto wires_0_in_b = gmw_providers_[0]->make_boolean_input_gate_other(1, num_wires, num_simd);
  auto [input_b_promise, wires_1_in_b] =
      gmw_providers_[1]->make_boolean_input_gate_my(1, num_wires, num_simd);
  auto wires_0_out = gmw_providers_[0]->make_binary_gate(ENCRYPTO::PrimitiveOperationType::XOR,
                                                         wires_0_in_a, wires_0_in_b);
  auto wires_1_out = gmw_providers_[1]->make_binary_gate(ENCRYPTO::PrimitiveOperationType::XOR,
                                                         wires_1_in_a, wires_1_in_b);

  run_setup();
  run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  run_gates_online();

  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& expected_output_bits = expected_output.at(wire_i);
    const auto wire_0 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_0_out.at(wire_i));
    const auto wire_1 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_1_out.at(wire_i));
    wire_0->wait_online();
    wire_1->wait_online();
    const auto& share_0 = wire_0->get_share();
    const auto& share_1 = wire_1->get_share();
    ASSERT_EQ(share_0.GetSize(), num_simd);
    ASSERT_EQ(share_1.GetSize(), num_simd);
    ASSERT_EQ(expected_output_bits, share_0 ^ share_1);
  }
}

TEST_F(BooleanGMWTest, ConstantXOR) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs_a = generate_inputs(num_wires, num_simd);
  const auto inputs_b = generate_inputs(num_wires, num_simd);
  MOTION::BitValues expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output),
                 [](const auto& bv_a, const auto& bv_b) { return bv_a ^ bv_b; });

  MOTION::WireVector plain_wires_0(num_wires);
  MOTION::WireVector plain_wires_1(num_wires);
  std::transform(std::begin(inputs_b), std::end(inputs_b), std::begin(plain_wires_0),
                 [](const auto& bv) { return std::make_shared<plain::BooleanPlainWire>(bv); });
  std::transform(std::begin(inputs_b), std::end(inputs_b), std::begin(plain_wires_1),
                 [](const auto& bv) { return std::make_shared<plain::BooleanPlainWire>(bv); });

  auto [input_a_promise, wires_0_in_a] =
      gmw_providers_[0]->make_boolean_input_gate_my(0, num_wires, num_simd);
  auto wires_1_in_a = gmw_providers_[1]->make_boolean_input_gate_other(0, num_wires, num_simd);
  auto wires_0_out = gmw_providers_[0]->make_binary_gate(ENCRYPTO::PrimitiveOperationType::XOR,
                                                         wires_0_in_a, plain_wires_0);
  auto wires_1_out = gmw_providers_[1]->make_binary_gate(ENCRYPTO::PrimitiveOperationType::XOR,
                                                         wires_1_in_a, plain_wires_1);

  run_setup();
  run_gates_setup();
  input_a_promise.set_value(inputs_a);
  run_gates_online();

  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& expected_output_bits = expected_output.at(wire_i);
    const auto wire_0 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_0_out.at(wire_i));
    const auto wire_1 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_1_out.at(wire_i));
    wire_0->wait_online();
    wire_1->wait_online();
    const auto& share_0 = wire_0->get_share();
    const auto& share_1 = wire_1->get_share();
    ASSERT_EQ(share_0.GetSize(), num_simd);
    ASSERT_EQ(share_1.GetSize(), num_simd);
    ASSERT_EQ(expected_output_bits, share_0 ^ share_1);
  }
}

TEST_F(BooleanGMWTest, AND) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs_a = generate_inputs(num_wires, num_simd);
  const auto inputs_b = generate_inputs(num_wires, num_simd);
  MOTION::BitValues expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output),
                 [](const auto& bv_a, const auto& bv_b) { return bv_a & bv_b; });

  auto [input_a_promise, wires_0_in_a] =
      gmw_providers_[0]->make_boolean_input_gate_my(0, num_wires, num_simd);
  auto wires_1_in_a = gmw_providers_[1]->make_boolean_input_gate_other(0, num_wires, num_simd);
  auto wires_0_in_b = gmw_providers_[0]->make_boolean_input_gate_other(1, num_wires, num_simd);
  auto [input_b_promise, wires_1_in_b] =
      gmw_providers_[1]->make_boolean_input_gate_my(1, num_wires, num_simd);
  auto wires_0_out = gmw_providers_[0]->make_binary_gate(ENCRYPTO::PrimitiveOperationType::AND,
                                                         wires_0_in_a, wires_0_in_b);
  auto wires_1_out = gmw_providers_[1]->make_binary_gate(ENCRYPTO::PrimitiveOperationType::AND,
                                                         wires_1_in_a, wires_1_in_b);

  run_setup();
  run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  run_gates_online();

  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& expected_output_bits = expected_output.at(wire_i);
    const auto wire_0 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_0_out.at(wire_i));
    const auto wire_1 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_1_out.at(wire_i));
    wire_0->wait_online();
    wire_1->wait_online();
    const auto& share_0 = wire_0->get_share();
    const auto& share_1 = wire_1->get_share();
    ASSERT_EQ(share_0.GetSize(), num_simd);
    ASSERT_EQ(share_1.GetSize(), num_simd);
    ASSERT_EQ(expected_output_bits, share_0 ^ share_1);
  }
}

TEST_F(BooleanGMWTest, ConstantAND) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs_a = generate_inputs(num_wires, num_simd);
  const auto inputs_b = generate_inputs(num_wires, num_simd);
  MOTION::BitValues expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output),
                 [](const auto& bv_a, const auto& bv_b) { return bv_a & bv_b; });

  MOTION::WireVector plain_wires_0(num_wires);
  MOTION::WireVector plain_wires_1(num_wires);
  std::transform(std::begin(inputs_b), std::end(inputs_b), std::begin(plain_wires_0),
                 [](const auto& bv) { return std::make_shared<plain::BooleanPlainWire>(bv); });
  std::transform(std::begin(inputs_b), std::end(inputs_b), std::begin(plain_wires_1),
                 [](const auto& bv) { return std::make_shared<plain::BooleanPlainWire>(bv); });

  auto [input_a_promise, wires_0_in_a] =
      gmw_providers_[0]->make_boolean_input_gate_my(0, num_wires, num_simd);
  auto wires_1_in_a = gmw_providers_[1]->make_boolean_input_gate_other(0, num_wires, num_simd);
  auto wires_0_out = gmw_providers_[0]->make_binary_gate(ENCRYPTO::PrimitiveOperationType::AND,
                                                         wires_0_in_a, plain_wires_0);
  auto wires_1_out = gmw_providers_[1]->make_binary_gate(ENCRYPTO::PrimitiveOperationType::AND,
                                                         wires_1_in_a, plain_wires_1);

  run_setup();
  run_gates_setup();
  input_a_promise.set_value(inputs_a);
  run_gates_online();

  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& expected_output_bits = expected_output.at(wire_i);
    const auto wire_0 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_0_out.at(wire_i));
    const auto wire_1 = std::dynamic_pointer_cast<BooleanGMWWire>(wires_1_out.at(wire_i));
    wire_0->wait_online();
    wire_1->wait_online();
    const auto& share_0 = wire_0->get_share();
    const auto& share_1 = wire_1->get_share();
    ASSERT_EQ(share_0.GetSize(), num_simd);
    ASSERT_EQ(share_1.GetSize(), num_simd);
    ASSERT_EQ(expected_output_bits, share_0 ^ share_1);
  }
}

template <typename T>
class ArithmeticGMWTest : public GMWTest {
 public:
  static std::vector<T> generate_inputs(std::size_t num_simd) {
    return MOTION::Helpers::RandomVector<T>(num_simd);
  }
  std::pair<ENCRYPTO::ReusableFiberPromise<MOTION::IntegerValues<T>>, MOTION::WireVector>
  make_arithmetic_T_input_gate_my(std::size_t party_id, std::size_t input_owner,
                                  std::size_t num_simd) {
    auto& gp = *gmw_providers_.at(party_id);
    if constexpr (ENCRYPTO::bit_size_v<T> == 8) {
      return gp.make_arithmetic_8_input_gate_my(input_owner, num_simd);
    } else if constexpr (ENCRYPTO::bit_size_v<T> == 16) {
      return gp.make_arithmetic_16_input_gate_my(input_owner, num_simd);
    } else if constexpr (ENCRYPTO::bit_size_v<T> == 32) {
      return gp.make_arithmetic_32_input_gate_my(input_owner, num_simd);
    } else if constexpr (ENCRYPTO::bit_size_v<T> == 64) {
      return gp.make_arithmetic_64_input_gate_my(input_owner, num_simd);
    }
  }
  MOTION::WireVector make_arithmetic_T_input_gate_other(std::size_t party_id,
                                                        std::size_t input_owner,
                                                        std::size_t num_simd) {
    auto& gp = *gmw_providers_.at(party_id);
    if constexpr (ENCRYPTO::bit_size_v<T> == 8) {
      return gp.make_arithmetic_8_input_gate_other(input_owner, num_simd);
    } else if constexpr (ENCRYPTO::bit_size_v<T> == 16) {
      return gp.make_arithmetic_16_input_gate_other(input_owner, num_simd);
    } else if constexpr (ENCRYPTO::bit_size_v<T> == 32) {
      return gp.make_arithmetic_32_input_gate_other(input_owner, num_simd);
    } else if constexpr (ENCRYPTO::bit_size_v<T> == 64) {
      return gp.make_arithmetic_64_input_gate_other(input_owner, num_simd);
    }
  }
  ENCRYPTO::ReusableFiberFuture<MOTION::IntegerValues<T>> make_arithmetic_T_output_gate_my(
      std::size_t party_id, std::size_t output_owner, const MOTION::WireVector& in) {
    auto& gp = *gmw_providers_.at(party_id);
    if constexpr (ENCRYPTO::bit_size_v<T> == 8) {
      return gp.make_arithmetic_8_output_gate_my(output_owner, in);
    } else if constexpr (ENCRYPTO::bit_size_v<T> == 16) {
      return gp.make_arithmetic_16_output_gate_my(output_owner, in);
    } else if constexpr (ENCRYPTO::bit_size_v<T> == 32) {
      return gp.make_arithmetic_32_output_gate_my(output_owner, in);
    } else if constexpr (ENCRYPTO::bit_size_v<T> == 64) {
      return gp.make_arithmetic_64_output_gate_my(output_owner, in);
    }
  }
};

using integer_types = ::testing::Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE(ArithmeticGMWTest, integer_types);

TYPED_TEST(ArithmeticGMWTest, Input) {
  std::size_t num_simd = 10;
  const auto inputs_a = this->generate_inputs(num_simd);
  const auto inputs_b = this->generate_inputs(num_simd);

  // input of party 0
  auto [input_a_promise, wires_a_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_a_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  // input of party 1
  auto wires_b_in_0 = this->make_arithmetic_T_input_gate_other(0, 1, num_simd);
  auto [input_b_promise, wires_b_in_1] = this->make_arithmetic_T_input_gate_my(1, 1, num_simd);

  ASSERT_EQ(wires_a_in_0.size(), 1);
  ASSERT_EQ(wires_a_in_1.size(), 1);
  ASSERT_EQ(wires_b_in_0.size(), 1);
  ASSERT_EQ(wires_b_in_1.size(), 1);
  ASSERT_EQ(wires_a_in_0.at(0)->get_num_simd(), num_simd);
  ASSERT_EQ(wires_a_in_1.at(0)->get_num_simd(), num_simd);
  ASSERT_EQ(wires_b_in_0.at(0)->get_num_simd(), num_simd);
  ASSERT_EQ(wires_b_in_1.at(0)->get_num_simd(), num_simd);

  this->run_setup();
  this->run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  this->run_gates_online();

  // check wire values
  const auto wire_a0 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_a_in_0.at(0));
  const auto wire_a1 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_a_in_1.at(0));
  const auto wire_b0 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_b_in_0.at(0));
  const auto wire_b1 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_b_in_1.at(0));
  wire_a0->wait_online();
  wire_a1->wait_online();
  wire_b0->wait_online();
  wire_b1->wait_online();
  const auto& share_a0 = wire_a0->get_share();
  const auto& share_a1 = wire_a1->get_share();
  const auto& share_b0 = wire_b0->get_share();
  const auto& share_b1 = wire_b1->get_share();
  ASSERT_EQ(share_a0.size(), num_simd);
  ASSERT_EQ(share_a1.size(), num_simd);
  ASSERT_EQ(share_b0.size(), num_simd);
  ASSERT_EQ(share_b1.size(), num_simd);
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    ASSERT_EQ(inputs_a[simd_j], TypeParam(share_a0[simd_j] + share_a1[simd_j]));
    ASSERT_EQ(inputs_b[simd_j], TypeParam(share_b0[simd_j] + share_b1[simd_j]));
  }
}

TYPED_TEST(ArithmeticGMWTest, OutputSingle) {
  std::size_t num_simd = 10;
  const auto inputs_a = this->generate_inputs(num_simd);
  const auto inputs_b = this->generate_inputs(num_simd);

  // input of party 0
  auto [input_a_promise, wires_a_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_a_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  // input of party 1
  auto wires_b_in_0 = this->make_arithmetic_T_input_gate_other(0, 1, num_simd);
  auto [input_b_promise, wires_b_in_1] = this->make_arithmetic_T_input_gate_my(1, 1, num_simd);

  auto output_future_a_0 = this->make_arithmetic_T_output_gate_my(0, 0, wires_a_in_0);
  this->gmw_providers_[1]->make_arithmetic_output_gate_other(0, wires_a_in_1);
  this->gmw_providers_[0]->make_arithmetic_output_gate_other(1, wires_b_in_0);
  auto output_future_b_1 = this->make_arithmetic_T_output_gate_my(1, 1, wires_b_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  this->run_gates_online();

  auto outputs_a = output_future_a_0.get();
  auto outputs_b = output_future_b_1.get();

  ASSERT_EQ(outputs_a.size(), num_simd);
  ASSERT_EQ(outputs_b.size(), num_simd);

  // check outputs values
  ASSERT_EQ(inputs_a, outputs_a);
  ASSERT_EQ(inputs_b, outputs_b);
}

TYPED_TEST(ArithmeticGMWTest, OutputBoth) {
  std::size_t num_simd = 10;
  const auto inputs_a = this->generate_inputs(num_simd);
  const auto inputs_b = this->generate_inputs(num_simd);

  // input of party 0
  auto [input_a_promise, wires_a_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_a_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  // input of party 1
  auto wires_b_in_0 = this->make_arithmetic_T_input_gate_other(0, 1, num_simd);
  auto [input_b_promise, wires_b_in_1] = this->make_arithmetic_T_input_gate_my(1, 1, num_simd);

  auto output_future_a_0 = this->make_arithmetic_T_output_gate_my(0, MOTION::ALL_PARTIES, wires_a_in_0);
  auto output_future_a_1 = this->make_arithmetic_T_output_gate_my(1, MOTION::ALL_PARTIES, wires_a_in_1);
  auto output_future_b_0 = this->make_arithmetic_T_output_gate_my(0, MOTION::ALL_PARTIES, wires_b_in_0);
  auto output_future_b_1 = this->make_arithmetic_T_output_gate_my(1, MOTION::ALL_PARTIES, wires_b_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  this->run_gates_online();

  auto outputs_a_0 = output_future_a_0.get();
  auto outputs_a_1 = output_future_a_1.get();
  auto outputs_b_0 = output_future_b_0.get();
  auto outputs_b_1 = output_future_b_1.get();

  ASSERT_EQ(outputs_a_0.size(), num_simd);
  ASSERT_EQ(outputs_a_1.size(), num_simd);
  ASSERT_EQ(outputs_b_0.size(), num_simd);
  ASSERT_EQ(outputs_b_1.size(), num_simd);

  // check outputs values
  ASSERT_EQ(inputs_a, outputs_a_0);
  ASSERT_EQ(inputs_a, outputs_a_1);
  ASSERT_EQ(inputs_b, outputs_b_0);
  ASSERT_EQ(inputs_b, outputs_b_1);
}

TYPED_TEST(ArithmeticGMWTest, OutputShare) {
  std::size_t num_simd = 10;
  const auto inputs = this->generate_inputs(num_simd);

  // input of party 0
  auto [input_promise, wires_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  auto output_future_0 =
      this->gmw_providers_[0]->template make_arithmetic_output_share_gate<TypeParam>(wires_in_0);
  auto output_future_1 =
      this->gmw_providers_[1]->template make_arithmetic_output_share_gate<TypeParam>(wires_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(inputs);
  this->run_gates_online();

  auto outputs_0 = output_future_0.get();
  auto outputs_1 = output_future_1.get();

  ASSERT_EQ(outputs_0.size(), num_simd);
  ASSERT_EQ(outputs_1.size(), num_simd);

  // check outputs values
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    ASSERT_EQ(inputs.at(simd_j), TypeParam(outputs_0.at(simd_j) + outputs_1.at(simd_j)));
  }
}

TYPED_TEST(ArithmeticGMWTest, NEG) {
  std::size_t num_simd = 10;
  const auto inputs = this->generate_inputs(num_simd);
  std::vector<TypeParam> expected_output;
  std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(expected_output),
                 std::negate{});

  // input of party 0
  auto [input_promise, wires_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  auto wires_0_out =
      this->gmw_providers_[0]->make_unary_gate(ENCRYPTO::PrimitiveOperationType::NEG, wires_in_0);
  auto wires_1_out =
      this->gmw_providers_[1]->make_unary_gate(ENCRYPTO::PrimitiveOperationType::NEG, wires_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(inputs);
  this->run_gates_online();

  // check wire values
  const auto wire_0 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_0_out.at(0));
  const auto wire_1 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_1_out.at(0));
  wire_0->wait_online();
  wire_1->wait_online();
  const auto& share_0 = wire_0->get_share();
  const auto& share_1 = wire_1->get_share();
  ASSERT_EQ(share_0.size(), num_simd);
  ASSERT_EQ(share_1.size(), num_simd);
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    ASSERT_EQ(expected_output.at(simd_j), TypeParam(share_0.at(simd_j) + share_1.at(simd_j)));
  }
}

TYPED_TEST(ArithmeticGMWTest, ADD) {
  std::size_t num_simd = 10;
  const auto inputs_a = this->generate_inputs(num_simd);
  const auto inputs_b = this->generate_inputs(num_simd);
  std::vector<TypeParam> expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output), std::plus{});

  // input of party 0
  auto [input_a_promise, wires_a_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_a_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  // input of party 1
  auto wires_b_in_0 = this->make_arithmetic_T_input_gate_other(0, 1, num_simd);
  auto [input_b_promise, wires_b_in_1] = this->make_arithmetic_T_input_gate_my(1, 1, num_simd);

  auto wires_out_0 = this->gmw_providers_[0]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::ADD, wires_a_in_0, wires_b_in_0);
  auto wires_out_1 = this->gmw_providers_[1]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::ADD, wires_a_in_1, wires_b_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  this->run_gates_online();

  // check wire values
  const auto wire_0 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_out_0.at(0));
  const auto wire_1 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_out_1.at(0));
  wire_0->wait_online();
  wire_1->wait_online();
  const auto& share_0 = wire_0->get_share();
  const auto& share_1 = wire_1->get_share();
  ASSERT_EQ(share_0.size(), num_simd);
  ASSERT_EQ(share_1.size(), num_simd);
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    ASSERT_EQ(expected_output.at(simd_j), TypeParam(share_0.at(simd_j) + share_1.at(simd_j)));
  }
}

TYPED_TEST(ArithmeticGMWTest, ConstantADD) {
  std::size_t num_simd = 10;
  const auto inputs_a = this->generate_inputs(num_simd);
  const auto inputs_b = this->generate_inputs(num_simd);
  std::vector<TypeParam> expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output), std::plus{});

  MOTION::WireVector plain_wires_0 = {
      std::make_shared<plain::ArithmeticPlainWire<TypeParam>>(inputs_b)};
  MOTION::WireVector plain_wires_1 = {
      std::make_shared<plain::ArithmeticPlainWire<TypeParam>>(inputs_b)};

  // input of party 0
  auto [input_a_promise, wires_a_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_a_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  auto wires_out_0 = this->gmw_providers_[0]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::ADD, wires_a_in_0, plain_wires_0);
  auto wires_out_1 = this->gmw_providers_[1]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::ADD, wires_a_in_1, plain_wires_1);

  this->run_setup();
  this->run_gates_setup();
  input_a_promise.set_value(inputs_a);
  this->run_gates_online();

  // check wire values
  const auto wire_0 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_out_0.at(0));
  const auto wire_1 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_out_1.at(0));
  wire_0->wait_online();
  wire_1->wait_online();
  const auto& share_0 = wire_0->get_share();
  const auto& share_1 = wire_1->get_share();
  ASSERT_EQ(share_0.size(), num_simd);
  ASSERT_EQ(share_1.size(), num_simd);
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    ASSERT_EQ(expected_output.at(simd_j), TypeParam(share_0.at(simd_j) + share_1.at(simd_j)));
  }
}

TYPED_TEST(ArithmeticGMWTest, MUL) {
  std::size_t num_simd = 10;
  const auto inputs_a = this->generate_inputs(num_simd);
  const auto inputs_b = this->generate_inputs(num_simd);
  std::vector<TypeParam> expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output), std::multiplies{});

  // input of party 0
  auto [input_a_promise, wires_a_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_a_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  // input of party 1
  auto wires_b_in_0 = this->make_arithmetic_T_input_gate_other(0, 1, num_simd);
  auto [input_b_promise, wires_b_in_1] = this->make_arithmetic_T_input_gate_my(1, 1, num_simd);

  auto wires_out_0 = this->gmw_providers_[0]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::MUL, wires_a_in_0, wires_b_in_0);
  auto wires_out_1 = this->gmw_providers_[1]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::MUL, wires_a_in_1, wires_b_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  this->run_gates_online();

  // check wire values
  const auto wire_0 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_out_0.at(0));
  const auto wire_1 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_out_1.at(0));
  wire_0->wait_online();
  wire_1->wait_online();
  const auto& share_0 = wire_0->get_share();
  const auto& share_1 = wire_1->get_share();
  ASSERT_EQ(share_0.size(), num_simd);
  ASSERT_EQ(share_1.size(), num_simd);
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    ASSERT_EQ(expected_output.at(simd_j), TypeParam(share_0.at(simd_j) + share_1.at(simd_j)));
  }
}

TYPED_TEST(ArithmeticGMWTest, ConstantMUL) {
  std::size_t num_simd = 10;
  const auto inputs_a = this->generate_inputs(num_simd);
  const auto inputs_b = this->generate_inputs(num_simd);
  std::vector<TypeParam> expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output), std::multiplies{});

  MOTION::WireVector plain_wires_0 = {
      std::make_shared<plain::ArithmeticPlainWire<TypeParam>>(inputs_b)};
  MOTION::WireVector plain_wires_1 = {
      std::make_shared<plain::ArithmeticPlainWire<TypeParam>>(inputs_b)};

  // input of party 0
  auto [input_a_promise, wires_a_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_a_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  auto wires_out_0 = this->gmw_providers_[0]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::MUL, wires_a_in_0, plain_wires_0);
  auto wires_out_1 = this->gmw_providers_[1]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::MUL, wires_a_in_1, plain_wires_1);

  this->run_setup();
  this->run_gates_setup();
  input_a_promise.set_value(inputs_a);
  this->run_gates_online();

  // check wire values
  const auto wire_0 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_out_0.at(0));
  const auto wire_1 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_out_1.at(0));
  wire_0->wait_online();
  wire_1->wait_online();
  const auto& share_0 = wire_0->get_share();
  const auto& share_1 = wire_1->get_share();
  ASSERT_EQ(share_0.size(), num_simd);
  ASSERT_EQ(share_1.size(), num_simd);
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    ASSERT_EQ(expected_output.at(simd_j), TypeParam(share_0.at(simd_j) + share_1.at(simd_j)));
  }
}

TYPED_TEST(ArithmeticGMWTest, SQR) {
  std::size_t num_simd = 10;
  const auto inputs = this->generate_inputs(num_simd);
  std::vector<TypeParam> expected_output;
  std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(expected_output),
                 [](auto x) { return x * x; });

  // input of party 0
  auto [input_promise, wires_in_0] = this->make_arithmetic_T_input_gate_my(0, 0, num_simd);
  auto wires_in_1 = this->make_arithmetic_T_input_gate_other(1, 0, num_simd);

  auto wires_0_out =
      this->gmw_providers_[0]->make_unary_gate(ENCRYPTO::PrimitiveOperationType::SQR, wires_in_0);
  auto wires_1_out =
      this->gmw_providers_[1]->make_unary_gate(ENCRYPTO::PrimitiveOperationType::SQR, wires_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(inputs);
  this->run_gates_online();

  // check wire values
  const auto wire_0 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_0_out.at(0));
  const auto wire_1 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_1_out.at(0));
  wire_0->wait_online();
  wire_1->wait_online();
  const auto& share_0 = wire_0->get_share();
  const auto& share_1 = wire_1->get_share();
  ASSERT_EQ(share_0.size(), num_simd);
  ASSERT_EQ(share_1.size(), num_simd);
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    ASSERT_EQ(expected_output.at(simd_j), TypeParam(share_0.at(simd_j) + share_1.at(simd_j)));
  }
}

// naive transposition of integers into bit vectors
template <typename T>
static std::vector<ENCRYPTO::BitVector<>> int_to_bit_vectors(const std::vector<T>& ints) {
  const auto num_wires = ENCRYPTO::bit_size_v<T>;
  const auto num_simd = ints.size();
  std::vector<ENCRYPTO::BitVector<>> bits;
  std::generate_n(std::back_inserter(bits), num_wires,
                  [num_simd] { return ENCRYPTO::BitVector<>(num_simd); });
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
      auto v = bool(ints[simd_j] & (T(1) << wire_i));
      bits[wire_i].Set(v, simd_j);
    }
  }

  return bits;
}

TYPED_TEST(ArithmeticGMWTest, BooleanToArithmeticGMW) {
  std::size_t num_wires = ENCRYPTO::bit_size_v<TypeParam>;
  std::size_t num_simd = 1;
  const auto expected_output = this->generate_inputs(num_simd);
  std::vector<ENCRYPTO::BitVector<>> inputs = int_to_bit_vectors(expected_output);

  auto [input_promise, wires_in_0] =
      this->gmw_providers_[0]->make_boolean_input_gate_my(0, num_wires, num_simd);
  auto wires_in_1 = this->gmw_providers_[1]->make_boolean_input_gate_other(0, num_wires, num_simd);

  auto wires_0 = this->gmw_providers_[0]->convert(MOTION::MPCProtocol::ArithmeticGMW, wires_in_0);
  auto wires_1 = this->gmw_providers_[1]->convert(MOTION::MPCProtocol::ArithmeticGMW, wires_in_1);

  this->run_setup();
  this->run_gates_setup();
  input_promise.set_value(inputs);
  this->run_gates_online();

  // check wire values
  ASSERT_EQ(wires_0.size(), 1);
  ASSERT_EQ(wires_1.size(), 1);
  const auto wire_0 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_0.at(0));
  const auto wire_1 = std::dynamic_pointer_cast<ArithmeticGMWWire<TypeParam>>(wires_1.at(0));
  wire_0->wait_online();
  wire_1->wait_online();
  const auto& share_0 = wire_0->get_share();
  const auto& share_1 = wire_1->get_share();
  ASSERT_EQ(share_0.size(), num_simd);
  ASSERT_EQ(share_1.size(), num_simd);
  for (std::size_t simd_j = 0; simd_j < num_simd; ++simd_j) {
    ASSERT_EQ(expected_output.at(simd_j), TypeParam(share_0.at(simd_j) + share_1.at(simd_j)));
  }
}
