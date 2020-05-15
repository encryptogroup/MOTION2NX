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
#include <array>
#include <iterator>
#include <memory>

#include "base/gate_register.h"
#include "communication/communication_layer.h"
#include "crypto/base_ots/base_ot_provider.h"
#include "crypto/motion_base_provider.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "gate/new_gate.h"
#include "protocols/gmw/wire.h"
#include "protocols/yao/wire.h"
#include "protocols/yao/yao_provider.h"
#include "utility/logger.h"

using namespace MOTION::proto::yao;

class YaoTest : public ::testing::Test {
 protected:
  void SetUp() override {
    comm_layers_ = MOTION::Communication::make_dummy_communication_layers(2);
    for (std::size_t i = 0; i < 2; ++i) {
      loggers_[i] = std::make_shared<MOTION::Logger>(i, boost::log::trivial::severity_level::trace);
      comm_layers_[i]->set_logger(loggers_[i]);
      base_ot_providers_[i] = std::make_unique<MOTION::BaseOTProvider>(*comm_layers_[i], nullptr);
      motion_base_providers_[i] =
          std::make_unique<MOTION::Crypto::MotionBaseProvider>(*comm_layers_[i], nullptr);
      ot_provider_managers_[i] = std::make_unique<ENCRYPTO::ObliviousTransfer::OTProviderManager>(
          *comm_layers_[i], *base_ot_providers_[i], *motion_base_providers_[i], nullptr);
      gate_registers_[i] = std::make_unique<MOTION::GateRegister>();
      yao_providers_[i] =
          std::make_unique<YaoProvider>(*comm_layers_[i], *gate_registers_[i],
                                        ot_provider_managers_[i]->get_provider(1 - i), loggers_[i]);
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
        ot_provider_managers_[i]->get_provider(1 - i).SendSetup();
        ot_provider_managers_[i]->get_provider(1 - i).ReceiveSetup();
        yao_providers_[i]->setup();
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
  std::array<std::unique_ptr<MOTION::GateRegister>, 2> gate_registers_;
  std::array<std::unique_ptr<YaoProvider>, 2> yao_providers_;
  std::array<std::shared_ptr<MOTION::Logger>, 2> loggers_;
};

static std::vector<ENCRYPTO::BitVector<>> generate_inputs(std::size_t num_wires,
                                                          std::size_t num_simd) {
  std::vector<ENCRYPTO::BitVector<>> inputs;
  std::generate_n(std::back_inserter(inputs), num_wires,
                  [num_simd] { return ENCRYPTO::BitVector<>::Random(num_simd); });
  return inputs;
}

TEST_F(YaoTest, InputGarbler) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  auto inputs = generate_inputs(num_wires, num_simd);

  auto [input_promise, wires_g] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);
  ASSERT_EQ(wires_g.size(), num_wires);
  ASSERT_EQ(wires_e.size(), num_wires);
  auto check_num_simd_f = [num_simd](const auto& w) { return w->get_num_simd(); };
  ASSERT_TRUE(std::all_of(std::begin(wires_g), std::end(wires_g), check_num_simd_f));
  ASSERT_TRUE(std::all_of(std::begin(wires_e), std::end(wires_e), check_num_simd_f));

  run_setup();
  run_gates_setup();
  input_promise.set_value(inputs);
  run_gates_online();

  // check wire values
  const auto global_offset = yao_providers_[garbler_i_]->get_global_offset();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& input_bits = inputs.at(wire_i);
    const auto wire_g = std::dynamic_pointer_cast<YaoWire>(wires_g.at(wire_i));
    const auto wire_e = std::dynamic_pointer_cast<YaoWire>(wires_e.at(wire_i));
    const auto& keys_g = wire_g->get_keys();
    const auto& keys_e = wire_e->get_keys();
    ASSERT_EQ(keys_g.size(), num_simd);
    ASSERT_EQ(keys_e.size(), num_simd);
    for (std::size_t simd_i = 0; simd_i < num_simd; ++simd_i) {
      if (input_bits.Get(simd_i)) {
        ASSERT_EQ(keys_e.at(simd_i), keys_g.at(simd_i) ^ global_offset);
      } else {
        ASSERT_EQ(keys_e.at(simd_i), keys_g.at(simd_i));
      }
    }
  }
}

TEST_F(YaoTest, InputEvaluator) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  auto inputs = generate_inputs(num_wires, num_simd);

  auto wires_g =
      yao_providers_[garbler_i_]->make_boolean_input_gate_other(evaluator_i_, num_wires, num_simd);
  auto [input_promise, wires_e] =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_my(evaluator_i_, num_wires, num_simd);
  ASSERT_EQ(wires_g.size(), num_wires);
  ASSERT_EQ(wires_e.size(), num_wires);
  auto check_num_simd_f = [num_simd](const auto& w) { return w->get_num_simd(); };
  ASSERT_TRUE(std::all_of(std::begin(wires_g), std::end(wires_g), check_num_simd_f));
  ASSERT_TRUE(std::all_of(std::begin(wires_e), std::end(wires_e), check_num_simd_f));

  run_setup();
  run_gates_setup();
  input_promise.set_value(inputs);
  run_gates_online();

  // check wire values
  const auto global_offset = yao_providers_[garbler_i_]->get_global_offset();
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& input_bits = inputs.at(wire_i);
    const auto wire_g = std::dynamic_pointer_cast<YaoWire>(wires_g.at(wire_i));
    const auto wire_e = std::dynamic_pointer_cast<YaoWire>(wires_e.at(wire_i));
    const auto& keys_g = wire_g->get_keys();
    const auto& keys_e = wire_e->get_keys();
    ASSERT_EQ(keys_g.size(), num_simd);
    ASSERT_EQ(keys_e.size(), num_simd);
    for (std::size_t simd_i = 0; simd_i < num_simd; ++simd_i) {
      if (input_bits.Get(simd_i)) {
        ASSERT_EQ(keys_e.at(simd_i), keys_g.at(simd_i) ^ global_offset);
      } else {
        ASSERT_EQ(keys_e.at(simd_i), keys_g.at(simd_i));
      }
    }
  }
}

TEST_F(YaoTest, OutputGarbler) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs = generate_inputs(num_wires, num_simd);

  auto [input_promise, wires_g] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);
  auto output_future = yao_providers_[garbler_i_]->make_boolean_output_gate_my(garbler_i_, wires_g);
  yao_providers_[evaluator_i_]->make_boolean_output_gate_other(garbler_i_, wires_e);
  ASSERT_TRUE(output_future.valid());

  run_setup();
  run_gates_setup();
  input_promise.set_value(inputs);
  run_gates_online();

  auto outputs = output_future.get();

  // check output values
  ASSERT_EQ(inputs, outputs);
}

TEST_F(YaoTest, OutputEvaluator) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs = generate_inputs(num_wires, num_simd);

  auto [input_promise, wires_g] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);
  yao_providers_[garbler_i_]->make_boolean_output_gate_other(evaluator_i_, wires_g);
  auto output_future =
      yao_providers_[evaluator_i_]->make_boolean_output_gate_my(evaluator_i_, wires_e);
  ASSERT_TRUE(output_future.valid());

  run_setup();
  run_gates_setup();
  input_promise.set_value(inputs);
  run_gates_online();

  auto outputs = output_future.get();

  // check output values
  ASSERT_EQ(inputs, outputs);
}

TEST_F(YaoTest, OutputBoth) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs = generate_inputs(num_wires, num_simd);

  auto [input_promise, wires_g] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);
  auto output_future_g =
      yao_providers_[garbler_i_]->make_boolean_output_gate_my(MOTION::ALL_PARTIES, wires_g);
  auto output_future_e =
      yao_providers_[evaluator_i_]->make_boolean_output_gate_my(MOTION::ALL_PARTIES, wires_e);
  ASSERT_TRUE(output_future_g.valid());
  ASSERT_TRUE(output_future_e.valid());

  run_setup();
  run_gates_setup();
  input_promise.set_value(inputs);
  run_gates_online();

  auto outputs_g = output_future_g.get();
  auto outputs_e = output_future_e.get();

  // check output values
  ASSERT_EQ(inputs, outputs_g);
  ASSERT_EQ(inputs, outputs_e);
}

TEST_F(YaoTest, INV) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs = generate_inputs(num_wires, num_simd);
  MOTION::BitValues expected_output;
  std::transform(std::begin(inputs), std::end(inputs), std::back_inserter(expected_output),
                 [](const auto& bv) { return ~bv; });

  auto [input_promise, wires_g_in] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e_in =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);

  auto wires_g_out = yao_providers_[garbler_i_]->make_unary_gate(
      ENCRYPTO::PrimitiveOperationType::INV, wires_g_in);
  auto wires_e_out = yao_providers_[evaluator_i_]->make_unary_gate(
      ENCRYPTO::PrimitiveOperationType::INV, wires_e_in);

  auto output_future_g =
      yao_providers_[garbler_i_]->make_boolean_output_gate_my(garbler_i_, wires_g_out);
  yao_providers_[evaluator_i_]->make_boolean_output_gate_other(garbler_i_, wires_e_out);
  ASSERT_TRUE(output_future_g.valid());

  run_setup();
  run_gates_setup();
  input_promise.set_value(inputs);
  run_gates_online();

  auto outputs = output_future_g.get();

  // check output values
  ASSERT_EQ(expected_output, outputs);
}

TEST_F(YaoTest, XOR) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs_a = generate_inputs(num_wires, num_simd);
  const auto inputs_b = generate_inputs(num_wires, num_simd);
  MOTION::BitValues expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output),
                 [](const auto& bv_a, const auto& bv_b) { return bv_a ^ bv_b; });

  auto [input_a_promise, wires_g_in_a] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e_in_a =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);
  auto [input_b_promise, wires_g_in_b] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e_in_b =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);

  auto wires_g_out = yao_providers_[garbler_i_]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::XOR, wires_g_in_a, wires_g_in_b);
  auto wires_e_out = yao_providers_[evaluator_i_]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::XOR, wires_e_in_a, wires_e_in_b);

  auto output_future_g =
      yao_providers_[garbler_i_]->make_boolean_output_gate_my(garbler_i_, wires_g_out);
  yao_providers_[evaluator_i_]->make_boolean_output_gate_other(garbler_i_, wires_e_out);
  ASSERT_TRUE(output_future_g.valid());

  run_setup();
  run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  run_gates_online();

  auto outputs = output_future_g.get();

  // check output values
  ASSERT_EQ(expected_output, outputs);
}

TEST_F(YaoTest, AND) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs_a = generate_inputs(num_wires, num_simd);
  const auto inputs_b = generate_inputs(num_wires, num_simd);
  MOTION::BitValues expected_output;
  std::transform(std::begin(inputs_a), std::end(inputs_a), std::begin(inputs_b),
                 std::back_inserter(expected_output),
                 [](const auto& bv_a, const auto& bv_b) { return bv_a & bv_b; });

  auto [input_a_promise, wires_g_in_a] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e_in_a =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);
  auto [input_b_promise, wires_g_in_b] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e_in_b =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);

  auto wires_g_out = yao_providers_[garbler_i_]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::AND, wires_g_in_a, wires_g_in_b);
  auto wires_e_out = yao_providers_[evaluator_i_]->make_binary_gate(
      ENCRYPTO::PrimitiveOperationType::AND, wires_e_in_a, wires_e_in_b);

  auto output_future_g =
      yao_providers_[garbler_i_]->make_boolean_output_gate_my(garbler_i_, wires_g_out);
  yao_providers_[evaluator_i_]->make_boolean_output_gate_other(garbler_i_, wires_e_out);
  ASSERT_TRUE(output_future_g.valid());

  run_setup();
  run_gates_setup();
  input_a_promise.set_value(inputs_a);
  input_b_promise.set_value(inputs_b);
  run_gates_online();

  auto outputs = output_future_g.get();

  // check output values
  ASSERT_EQ(expected_output, outputs);
}

TEST_F(YaoTest, YaoToBooleanGMW) {
  std::size_t num_wires = 8;
  std::size_t num_simd = 10;
  const auto inputs = generate_inputs(num_wires, num_simd);

  auto [input_promise, wires_g_in] =
      yao_providers_[garbler_i_]->make_boolean_input_gate_my(garbler_i_, num_wires, num_simd);
  auto wires_e_in =
      yao_providers_[evaluator_i_]->make_boolean_input_gate_other(garbler_i_, num_wires, num_simd);

  auto wires_g = yao_providers_[garbler_i_]->convert(MOTION::MPCProtocol::BooleanGMW, wires_g_in);
  auto wires_e = yao_providers_[evaluator_i_]->convert(MOTION::MPCProtocol::BooleanGMW, wires_e_in);

  run_setup();
  run_gates_setup();
  input_promise.set_value(inputs);
  run_gates_online();

  // check wire values
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& expected_output_bits = inputs.at(wire_i);
    const auto wire_g =
        std::dynamic_pointer_cast<MOTION::proto::gmw::BooleanGMWWire>(wires_g.at(wire_i));
    const auto wire_e =
        std::dynamic_pointer_cast<MOTION::proto::gmw::BooleanGMWWire>(wires_e.at(wire_i));
    wire_g->wait_online();
    wire_e->wait_online();
    const auto& share_g = wire_g->get_share();
    const auto& share_e = wire_e->get_share();
    ASSERT_EQ(share_g.GetSize(), num_simd);
    ASSERT_EQ(share_e.GetSize(), num_simd);
    ASSERT_EQ(expected_output_bits, share_g ^ share_e);
  }
}
