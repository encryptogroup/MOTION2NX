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

#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "base/gate_factory.h"
#include "protocols/common/comm_mixin.h"
#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include "utility/bit_vector.h"
#include "utility/block.h"
#include "utility/reusable_future.h"

namespace ENCRYPTO {

struct AlgorithmDescription;

enum class PrimitiveOperationType : std::uint8_t;

namespace ObliviousTransfer {
class OTProvider;
}
}  // namespace ENCRYPTO

namespace MOTION {

class CircuitLoader;
class GateRegister;
class Logger;
class NewWire;

namespace Communication {
class CommunicationLayer;
}

namespace Crypto {
class MotionBaseProvider;
namespace garbling {
class HalfGateGarbler;
class HalfGateEvaluator;
}  // namespace garbling
}  // namespace Crypto

namespace proto::yao {

enum class OutputRecipient : std::uint8_t { garbler, evaluator, both };

class YaoWire;
using YaoWireVector = std::vector<std::shared_ptr<YaoWire>>;

struct YaoMessageHandler;

class YaoProvider : public GateFactory, public ENCRYPTO::enable_wait_setup, public CommMixin {
 public:
  enum class Role { garbler, evaluator };
  struct my_input_t {};

  YaoProvider(Communication::CommunicationLayer&, GateRegister&, CircuitLoader&,
              Crypto::MotionBaseProvider&, ENCRYPTO::ObliviousTransfer::OTProvider&,
              std::shared_ptr<Logger>);
  ~YaoProvider();

  std::string get_provider_name() const noexcept override { return "YaoProvider"; }

  // std::vector<std::shared_ptr<NewWire>> make_input_gate(std::size_t num_wire, std::size_t
  // num_simd); std::pair<ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>>,
  //           std::vector<std::shared_ptr<NewWire>>>
  // make_input_gate(std::size_t num_wire, std::size_t num_simd, my_input_t);
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

  WireVector convert_to(MPCProtocol proto, const WireVector&) override;
  WireVector convert_from(MPCProtocol proto, const WireVector&) override;

  void setup();
  ENCRYPTO::block128_t get_global_offset() const;

  void send_blocks_message(std::size_t gate_id, ENCRYPTO::block128_vector&& message) const;
  void send_bits_message(std::size_t gate_id, ENCRYPTO::BitVector<>&& message) const;
  void send_bits_message(std::size_t gate_id, const ENCRYPTO::BitVector<>& message) const;
  [[nodiscard]] ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector>
  register_for_blocks_message(std::size_t gate_id, std::size_t num_blocks);
  [[nodiscard]] ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> register_for_bits_message(
      std::size_t gate_id, std::size_t num_bits);
  void create_garbled_tables(std::size_t gate_id, const ENCRYPTO::block128_vector& keys_a,
                             const ENCRYPTO::block128_vector& keys_b, ENCRYPTO::block128_t* tables,
                             ENCRYPTO::block128_vector& keys_out) const noexcept;
  void evaluate_garbled_tables(std::size_t gate_id, const ENCRYPTO::block128_vector& keys_a,
                               const ENCRYPTO::block128_vector& keys_b,
                               const ENCRYPTO::block128_t* tables,
                               ENCRYPTO::block128_vector& keys_out) const noexcept;
  void create_garbled_circuit(std::size_t gate_id, std::size_t num_simd,
                              const ENCRYPTO::AlgorithmDescription&,
                              const ENCRYPTO::block128_vector& input_keys_a,
                              const ENCRYPTO::block128_vector& input_keys_b,
                              ENCRYPTO::block128_vector& tables,
                              ENCRYPTO::block128_vector& keys_out) const;
  void evaluate_garbled_circuit(std::size_t gate_id, std::size_t num_simd,
                                const ENCRYPTO::AlgorithmDescription&,
                                const ENCRYPTO::block128_vector& input_keys_a,
                                const ENCRYPTO::block128_vector& input_keys_b,
                                const ENCRYPTO::block128_vector& tables,
                                ENCRYPTO::block128_vector& keys_out) const;
  constexpr static std::size_t garbled_table_size = 2;

  Crypto::MotionBaseProvider& get_motion_base_provider() const noexcept {
    return motion_base_provider_;
  }
  ENCRYPTO::ObliviousTransfer::OTProvider& get_ot_provider() const noexcept { return ot_provider_; }
  CircuitLoader& get_circuit_loader() noexcept { return circuit_loader_; }
  std::shared_ptr<Logger> get_logger() const noexcept { return logger_; }

 private:
  YaoWireVector make_inv_gate(YaoWireVector&& in_a);
  YaoWireVector make_xor_gate(YaoWireVector&& in_a, YaoWireVector&& in_b);
  YaoWireVector make_and_gate(YaoWireVector&& in_a, YaoWireVector&& in_b);
  WireVector make_convert_to_boolean_gmw_gate(YaoWireVector&& in_a);
  YaoWireVector make_convert_from_boolean_gmw_gate(const WireVector& in_a);
  template <typename T>
  WireVector basic_make_convert_to_arithmetic_gmw_gate(YaoWireVector&& in_a);
  WireVector make_convert_to_arithmetic_gmw_gate(YaoWireVector&& in_a);
  template <typename T>
  WireVector basic_make_convert_from_arithmetic_gmw_gate(const WireVector& in_a);
  WireVector make_convert_from_arithmetic_gmw_gate(const WireVector& in_a);
  template <typename T>
  WireVector basic_make_convert_to_arithmetic_beavy_gate(YaoWireVector&& in_a);
  WireVector make_convert_to_arithmetic_beavy_gate(YaoWireVector&& in_a);
  template <typename T>
  WireVector basic_make_convert_from_arithmetic_beavy_gate(const WireVector& in_a);
  WireVector make_convert_from_arithmetic_beavy_gate(const WireVector& in_a);

 public:
  // tensor stuff
  template <typename T>
  tensor::TensorCP basic_make_convert_from_arithmetic_gmw_tensor(const tensor::TensorCP in_a);
  tensor::TensorCP make_convert_from_arithmetic_gmw_tensor(const tensor::TensorCP in_a);
  template <typename T>
  tensor::TensorCP basic_make_convert_from_arithmetic_beavy_tensor(const tensor::TensorCP in_a);
  tensor::TensorCP make_convert_from_arithmetic_beavy_tensor(const tensor::TensorCP in_a);
  template <typename T>
  tensor::TensorCP basic_make_convert_to_arithmetic_gmw_tensor(const tensor::TensorCP in_a);
  tensor::TensorCP make_convert_to_arithmetic_gmw_tensor(const tensor::TensorCP in_a);
  template <typename T>
  tensor::TensorCP basic_make_convert_to_arithmetic_beavy_tensor(const tensor::TensorCP in_a);
  tensor::TensorCP make_convert_to_arithmetic_beavy_tensor(const tensor::TensorCP in_a);
  tensor::TensorCP make_boolean_tensor_relu_op(const tensor::TensorCP in_a);
  tensor::TensorCP make_boolean_tensor_maxpool_op(const tensor::MaxPoolOp&, const tensor::TensorCP);

 private:
  Communication::CommunicationLayer& communication_layer_;
  GateRegister& gate_register_;
  CircuitLoader& circuit_loader_;
  Crypto::MotionBaseProvider& motion_base_provider_;
  ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider_;
  std::unique_ptr<Crypto::garbling::HalfGateGarbler> hg_garbler_;
  std::unique_ptr<Crypto::garbling::HalfGateEvaluator> hg_evaluator_;
  std::shared_ptr<YaoMessageHandler> message_handler_;
  std::size_t my_id_;
  Role role_;
  bool setup_ran_;
  std::shared_ptr<Logger> logger_;
};

}  // namespace proto::yao
}  // namespace MOTION
