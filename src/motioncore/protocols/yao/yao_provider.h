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
#include "utility/bit_vector.h"
#include "utility/block.h"
#include "utility/reusable_future.h"

namespace ENCRYPTO {

enum class PrimitiveOperationType : std::uint8_t;

namespace ObliviousTransfer {
class OTProvider;
}
}  // namespace ENCRYPTO

namespace MOTION {

class GateRegister;
class Logger;
class NewWire;

namespace Communication {
class CommunicationLayer;
}

namespace proto::yao {

class YaoWire;
using YaoWireVector = std::vector<std::shared_ptr<YaoWire>>;

class YaoProvider : public GateFactory {
  enum class Role { garbler, evaluator };

 public:
  YaoProvider(Communication::CommunicationLayer&, GateRegister&,
              ENCRYPTO::ObliviousTransfer::OTProvider&, std::shared_ptr<Logger>);

  std::pair<ENCRYPTO::ReusablePromise<ENCRYPTO::BitVector<>>, std::vector<std::shared_ptr<NewWire>>>
  make_input_gate() const;

  std::vector<std::shared_ptr<NewWire>> make_unary_gate(
      ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>&) const;

  std::vector<std::shared_ptr<NewWire>> make_binary_gate(
      ENCRYPTO::PrimitiveOperationType op, const std::vector<std::shared_ptr<NewWire>>&,
      const std::vector<std::shared_ptr<NewWire>>&) const;

  void setup() noexcept;
  ENCRYPTO::block128_t get_global_offset() const noexcept { return global_offset_; }

  void send_keys_message(std::size_t gate_id, ENCRYPTO::block128_vector&& message) const;
  void send_bits_message(std::size_t gate_id, ENCRYPTO::BitVector<>&& message) const;
  void send_bits_message(std::size_t gate_id, const ENCRYPTO::BitVector<>& message) const;
  [[nodiscard]] ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector> register_for_keys_message(
      std::size_t gate_id, std::size_t num_blocks) const;
  [[nodiscard]] ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> register_for_bits_message(
      std::size_t gate_id, std::size_t num_bits) const;
  void create_garbled_tables(const ENCRYPTO::block128_vector& keys_a,
                             const ENCRYPTO::block128_vector& keys_b,
                             ENCRYPTO::block128_vector& tables,
                             ENCRYPTO::block128_vector& keys_out) const noexcept;
  void evaluate_garbled_tables(const ENCRYPTO::block128_vector& keys_a,
                               const ENCRYPTO::block128_vector& keys_b,
                               const ENCRYPTO::block128_vector& tables,
                               ENCRYPTO::block128_vector& keys_out) const noexcept;
  // void send_garbled_tables(const ENCRYPTO::block128_vector& tables) const;
  // ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector> register_for_garbled_tables(
  //     std::size_t num) const;
  constexpr static std::size_t garbled_table_size = 2;

  ENCRYPTO::ObliviousTransfer::OTProvider& get_ot_provider() const noexcept { return ot_provider_; }
  std::shared_ptr<Logger> get_logger() const noexcept { return logger_; }

 private:
  YaoWireVector make_inv_gate(YaoWireVector&& in_a) const;
  YaoWireVector make_xor_gate(YaoWireVector&& in_a, YaoWireVector&& in_b) const;
  YaoWireVector make_and_gate(YaoWireVector&& in_a, YaoWireVector&& in_b) const;

 private:
  Communication::CommunicationLayer& communication_layer_;
  GateRegister& gate_register_;
  ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider_;
  Role role_;
  ENCRYPTO::block128_t global_offset_;
  std::shared_ptr<Logger> logger_;
};

}  // namespace proto::yao
}  // namespace MOTION
