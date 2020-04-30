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

#pragma once

#include <memory>

#include "base/gate_factory.h"
#include "utility/bit_vector.h"

namespace MOTION {

class GateRegister;
class Logger;
class NewWire;

namespace Communication {
class CommunicationLayer;
}

namespace proto::gmw {

enum class OutputRecipient : std::uint8_t { garbler, evaluator, both };

class BooleanGMWWire;
using BooleanGMWWireVector = std::vector<std::shared_ptr<BooleanGMWWire>>;

struct GMWMessageHandler;

class GMWProvider : public GateFactory {
 public:
  enum class Role { garbler, evaluator };
  struct my_input_t {};

  GMWProvider(Communication::CommunicationLayer&, GateRegister&, std::shared_ptr<Logger>);
  ~GMWProvider();

  ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>> make_output_gate(
      OutputRecipient, const std::vector<std::shared_ptr<NewWire>>&);

  // Implementation of GateFactors interface
  std::pair<ENCRYPTO::ReusableFiberPromise<BitValues>, WireVector> make_boolean_input_gate_my(
      std::size_t input_owner, std::size_t num_wires, std::size_t num_simd) override;
  WireVector make_boolean_input_gate_other(std::size_t input_owner, std::size_t num_wires,
                                           std::size_t num_simd) override;
  ENCRYPTO::ReusableFiberFuture<BitValues> make_boolean_output_gate_my(std::size_t output_owner,
                                                                       const WireVector&) override;
  void make_boolean_output_gate_other(std::size_t output_owner, const WireVector&) override;

  std::vector<std::shared_ptr<NewWire>> make_unary_gate(
      ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>&) override;

  std::vector<std::shared_ptr<NewWire>> make_binary_gate(
      ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>&,
      const std::vector<std::shared_ptr<NewWire>>&) override;

  void setup();

  std::shared_ptr<Logger> get_logger() const noexcept { return logger_; }

  bool is_my_job(std::size_t gate_id) const noexcept;

 private:
  BooleanGMWWireVector make_inv_gate(BooleanGMWWireVector&& in_a);
  BooleanGMWWireVector make_xor_gate(BooleanGMWWireVector&& in_a, BooleanGMWWireVector&& in_b);
  BooleanGMWWireVector make_and_gate(BooleanGMWWireVector&& in_a, BooleanGMWWireVector&& in_b);

 private:
  Communication::CommunicationLayer& communication_layer_;
  GateRegister& gate_register_;
  std::shared_ptr<GMWMessageHandler> message_handler_;
  std::size_t my_id_;
  Role role_;
  bool setup_ran_;
  std::shared_ptr<Logger> logger_;
};

}  // namespace proto::yao
}  // namespace MOTION
