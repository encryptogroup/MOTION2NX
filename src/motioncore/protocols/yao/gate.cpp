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

#include "gate.h"

#include <stdexcept>

#include "crypto/oblivious_transfer/ot_flavors.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "utility/constants.h"
#include "utility/helpers.h"
#include "utility/logger.h"
#include "yao_provider.h"

namespace MOTION::proto::yao {

namespace detail {

BasicYaoInputGate::BasicYaoInputGate(std::size_t gate_id, YaoProvider& yao_provider,
                                     std::size_t num_wires, std::size_t num_simd)
    : NewGate(gate_id), yao_provider_(yao_provider), num_wires_(num_wires), num_simd_(num_simd) {
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<YaoWire>(num_simd); });
}

BasicYaoInputGate::BasicYaoInputGate(
    std::size_t gate_id, YaoProvider& yao_provider, std::size_t num_wires, std::size_t num_simd,
    ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&& input_future)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      num_wires_(num_wires),
      num_simd_(num_simd),
      input_future_(std::move(input_future)) {
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<YaoWire>(num_simd); });
}

BasicYaoOutputGate::BasicYaoOutputGate(std::size_t gate_id, YaoProvider& yao_provider,
                                       YaoWireVector&& in, OutputRecipient output_recipient)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      num_wires_(in.size()),
      inputs_(std::move(in)),
      output_recipient_(output_recipient) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 1; wire_i < num_wires_; ++wire_i) {
    if (inputs_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
}

BasicYaoUnaryGate::BasicYaoUnaryGate(std::size_t gate_id, YaoProvider& yao_provider,
                                     YaoWireVector&& in, bool forward)
    : NewGate(gate_id), yao_provider_(yao_provider), num_wires_(in.size()), inputs_(std::move(in)) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  auto num_simd = inputs_[0]->get_num_simd();
  for (std::size_t wire_i = 1; wire_i < num_wires_; ++wire_i) {
    if (inputs_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  if (forward) {
    outputs_ = inputs_;
  } else {
    outputs_.reserve(num_wires_);
    std::generate_n(std::back_inserter(outputs_), num_wires_,
                    [num_simd] { return std::make_shared<YaoWire>(num_simd); });
  }
}

BasicYaoBinaryGate::BasicYaoBinaryGate(std::size_t gate_id, YaoProvider& yao_provider,
                                       YaoWireVector&& in_a, YaoWireVector&& in_b)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      num_wires_(in_a.size()),
      inputs_a_(std::move(in_a)),
      inputs_b_(std::move(in_b)) {
  if (num_wires_ == 0) {
    throw std::logic_error("number of wires need to be positive");
  }
  if (num_wires_ != inputs_b_.size()) {
    throw std::logic_error("number of wires need to be the same for both inputs");
  }
  auto num_simd = inputs_a_[0]->get_num_simd();
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    if (inputs_a_[wire_i]->get_num_simd() != num_simd ||
        inputs_b_[wire_i]->get_num_simd() != num_simd) {
      throw std::logic_error("number of SIMD values need to be the same for all wires");
    }
  }
  outputs_.reserve(num_wires_);
  std::generate_n(std::back_inserter(outputs_), num_wires_,
                  [num_simd] { return std::make_shared<YaoWire>(num_simd); });
}

// Determine the total number of bits in a collection of wires.
static std::size_t count_bits(const YaoWireVector& wires) {
  return std::transform_reduce(std::begin(wires), std::end(wires), 0, std::plus<>(),
                               [](const auto& a) { return a->get_num_simd(); });
}

}  // namespace detail

YaoInputGateGarbler::YaoInputGateGarbler(std::size_t gate_id, YaoProvider& yao_provider,
                                         std::size_t num_wires, std::size_t num_simd, bool in_setup)
    : BasicYaoInputGate(gate_id, yao_provider, num_wires, num_simd) {
  auto& ot_provider = yao_provider_.get_ot_provider();
  if (in_setup) {
    ot_sender_ = ot_provider.RegisterSendFixedXCOT128(num_wires * num_simd);
  } else {
    ot_sender_ = ot_provider.RegisterSendGOT128(num_wires * num_simd);
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoInputGateGarbler created (evaluator's input)", gate_id_));
    }
  }
}

YaoInputGateGarbler::YaoInputGateGarbler(std::size_t gate_id, YaoProvider& yao_provider,
                                         std::size_t num_wires, std::size_t num_simd)
    : YaoInputGateGarbler(gate_id, yao_provider, num_wires, num_simd, false) {}

YaoInputGateGarbler::YaoInputGateGarbler(
    std::size_t gate_id, YaoProvider& yao_provider, std::size_t num_wires, std::size_t num_simd,
    bool, ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&& input_future)
    : BasicYaoInputGate(gate_id, yao_provider, num_wires, num_simd, std::move(input_future)),
      ot_sender_(false) {
  assert(ot_sender_.index() == 0);
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoInputGateGarbler created (garbler's input)", gate_id_));
    }
  }
}

YaoInputGateGarbler::YaoInputGateGarbler(
    std::size_t gate_id, YaoProvider& yao_provider, std::size_t num_wires, std::size_t num_simd,
    ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&& input_future)
    : YaoInputGateGarbler(gate_id, yao_provider, num_wires, num_simd, false,
                          std::move(input_future)) {}

void YaoInputGateGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoInputGateGarbler::evaluate_setup start", gate_id_));
    }
  }

  if (ot_sender_.index() == 2) {
    // it's the sender's input and it's given during the setup phase
    // => use XCOT
    auto& xcot_sender = std::get<2>(ot_sender_);
    assert(xcot_sender);
    xcot_sender->SetCorrelation(yao_provider_.get_global_offset());
    xcot_sender->SendMessages();
    xcot_sender->ComputeOutputs();
    const auto keys = xcot_sender->GetOutputs();
    auto keys_ptr = keys.data();
    for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
      auto& w_o = outputs_[wire_i];
      std::copy_n(keys_ptr, num_simd_, std::begin(w_o->get_keys()));
      w_o->set_setup_ready();
      keys_ptr += num_simd_;
    }
  } else {
    // generate random keys for each wire
    for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
      auto& w_o = outputs_[wire_i];
      w_o->get_keys().set_to_random();
      w_o->set_setup_ready();
    }
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoInputGateGarbler::evaluate_setup end", gate_id_));
    }
  }
}

void YaoInputGateGarbler::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoInputGateGarbler::evaluate_online start", gate_id_));
    }
  }

  if (ot_sender_.index() == 2) {
    if constexpr (MOTION_VERBOSE_DEBUG) {
      auto logger = yao_provider_.get_logger();
      if (logger) {
        logger->LogTrace(fmt::format(
            "Gate {}: YaoInputGateGarbler::evaluate_online end (nothing to do)", gate_id_));
      }
    }
    return;
  }

  const auto global_offset = yao_provider_.get_global_offset();
  if (ot_sender_.index() == 1) {
    // evaluator's input
    ENCRYPTO::block128_vector ot_inputs(2 * num_wires_ * num_simd_);
    for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
      auto& w_o = outputs_[wire_i];
      auto& zero_keys = w_o->get_keys();
      auto keys_offset = 2 * wire_i * num_simd_;
      for (std::size_t simd_j = 0; simd_j < num_simd_; ++simd_j) {
        ot_inputs[keys_offset + 2 * simd_j] = zero_keys[simd_j];
        ot_inputs[keys_offset + 2 * simd_j + 1] = zero_keys[simd_j] ^ global_offset;
      }
    }
    auto& got_sender = std::get<1>(ot_sender_);
    got_sender->SetInputs(std::move(ot_inputs));
    got_sender->SendMessages();
  } else {
    const auto inputs = input_future_.get();
    if (inputs.size() != num_wires_) {
      throw std::runtime_error("dimension of input vector != num_wires_");
    }
    ENCRYPTO::block128_vector keys(num_wires_ * num_simd_);
    auto keys_it = std::begin(keys);
    for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
      auto& w_o = outputs_[wire_i];
      auto& zero_keys = w_o->get_keys();
      const auto& input_bits = inputs.at(wire_i);
      if (input_bits.GetSize() != num_simd_) {
        throw std::runtime_error("size of input bit vector != num_simd_");
      }
      auto wire_block_ptr = &*keys_it;
      keys_it = std::copy(std::begin(zero_keys), std::end(zero_keys), keys_it);
      for (std::size_t simd_j = 0; simd_j < num_simd_; ++simd_j) {
        if (input_bits.Get(simd_j)) {
          wire_block_ptr[simd_j] ^= global_offset;
        }
      }
    }
    // send keys to evaluator
    yao_provider_.send_blocks_message(gate_id_, std::move(keys));
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoInputGateGarbler::evaluate_online end", gate_id_));
    }
  }
}

YaoInputGateEvaluator::YaoInputGateEvaluator(std::size_t gate_id, YaoProvider& yao_provider,
                                             std::size_t num_wires, std::size_t num_simd, bool)
    : BasicYaoInputGate(gate_id, yao_provider, num_wires, num_simd), ot_receiver_(false) {
  assert(ot_receiver_.index() == 0);
  // garbler's input => register for keys message
  keys_future_ = yao_provider_.register_for_blocks_message(gate_id, num_wires * num_simd);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoInputGateEvaluator created (garbler's input)", gate_id_));
    }
  }
}

YaoInputGateEvaluator::YaoInputGateEvaluator(std::size_t gate_id, YaoProvider& yao_provider,
                                             std::size_t num_wires, std::size_t num_simd)
    : YaoInputGateEvaluator(gate_id, yao_provider, num_wires, num_simd, false) {}

YaoInputGateEvaluator::YaoInputGateEvaluator(
    std::size_t gate_id, YaoProvider& yao_provider, std::size_t num_wires, std::size_t num_simd,
    bool in_setup, ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&& input_future)
    : BasicYaoInputGate(gate_id, yao_provider, num_wires, num_simd, std::move(input_future)) {
  // my_inputs_ => register for GOTs
  auto& ot_provider = yao_provider_.get_ot_provider();
  if (in_setup) {
    ot_receiver_ = ot_provider.RegisterReceiveFixedXCOT128(num_wires * num_simd);
  } else {
    ot_receiver_ = ot_provider.RegisterReceiveGOT128(num_wires * num_simd);
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoInputGateEvaluator created (evaluator's input)", gate_id_));
    }
  }
}

YaoInputGateEvaluator::YaoInputGateEvaluator(
    std::size_t gate_id, YaoProvider& yao_provider, std::size_t num_wires, std::size_t num_simd,
    ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>&& input_future)
    : YaoInputGateEvaluator(gate_id, yao_provider, num_wires, num_simd, false,
                            std::move(input_future)) {}

void YaoInputGateEvaluator::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoInputGateEvaluator::evaluate_setup start", gate_id_));
    }
  }

  if (ot_receiver_.index() != 2) {
    if constexpr (MOTION_VERBOSE_DEBUG) {
      auto logger = yao_provider_.get_logger();
      if (logger) {
        logger->LogTrace(fmt::format(
            "Gate {}: YaoInputGateEvaluator::evaluate_setup end (nothing to do)", gate_id_));
      }
    }
    return;
  }
  // My input provided in setup phase, run C-OTs to obtain input keys
  auto inputs = input_future_.get();
  ENCRYPTO::BitVector choice_bits;  //(num_wires_ * num_simd_);
  for (auto& wire_bits : inputs) {
    choice_bits.Append(wire_bits);
  }
  auto& xcot_receiver = std::get<2>(ot_receiver_);
  assert(xcot_receiver);
  xcot_receiver->SetChoices(std::move(choice_bits));
  xcot_receiver->SendCorrections();
  xcot_receiver->ComputeOutputs();
  const auto received_keys = std::move(xcot_receiver->GetOutputs());
  auto keys_ptr = received_keys.data();
  for (auto& w : outputs_) {
    std::copy_n(keys_ptr, num_simd_, std::begin(w->get_keys()));
    w->set_online_ready();
    keys_ptr += num_simd_;
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoInputGateEvaluator::evaluate_setup end", gate_id_));
    }
  }
}

void YaoInputGateEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoInputGateEvaluator::evaluate_online start", gate_id_));
    }
  }

  if (ot_receiver_.index() == 2) {
    if constexpr (MOTION_VERBOSE_DEBUG) {
      auto logger = yao_provider_.get_logger();
      if (logger) {
        logger->LogTrace(fmt::format(
            "Gate {}: YaoInputGateEvaluator::evaluate_online end (nothing to do)", gate_id_));
      }
    }
    return;
  }

  ENCRYPTO::block128_vector received_keys;
  if (ot_receiver_.index() == 1) {
    // My input, run OTs to obtain input keys
    auto inputs = input_future_.get();
    ENCRYPTO::BitVector choice_bits;  //(num_wires_ * num_simd_);
    for (auto& wire_bits : inputs) {
      choice_bits.Append(wire_bits);
    }
    auto& got_receiver = std::get<1>(ot_receiver_);
    got_receiver->SetChoices(std::move(choice_bits));
    got_receiver->SendCorrections();
    got_receiver->ComputeOutputs();
    received_keys = std::move(got_receiver->GetOutputs());
  } else {
    // Garbler's input, receive input keys
    received_keys = keys_future_.get();
  }
  auto keys_ptr = received_keys.data();
  for (auto& w : outputs_) {
    std::copy_n(keys_ptr, num_simd_, std::begin(w->get_keys()));
    w->set_online_ready();
    keys_ptr += num_simd_;
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoInputGateEvaluator::evaluate_online end", gate_id_));
    }
  }
}

YaoOutputGateGarbler::YaoOutputGateGarbler(std::size_t gate_id, YaoProvider& yao_provider,
                                           YaoWireVector&& in, OutputRecipient output_recipient)
    : BasicYaoOutputGate(gate_id, yao_provider, std::move(in), output_recipient) {
  if (output_recipient_ != OutputRecipient::evaluator) {
    // We receive output, so the evaluator needs to send us the encoded output.
    auto num_gates = detail::count_bits(inputs_);
    bits_future_ = yao_provider_.register_for_bits_message(gate_id, num_gates);
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoOutputGateGarbler created", gate_id_));
    }
  }
}

ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>
YaoOutputGateGarbler::get_output_future() {
  if (output_recipient_ == OutputRecipient::evaluator) {
    throw std::logic_error("it's not the garbler's output");
  }
  return output_promise_.get_future();
}

void YaoOutputGateGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoOutputGateGarbler::evaluate_setup start", gate_id_));
    }
  }

  auto num_bits = detail::count_bits(inputs_);
  ENCRYPTO::BitVector<> decoding_info(num_bits);

  // Collect the decoding information which consists of the lsb of each zero
  // key.
  std::size_t bit_offset = 0;
  for (const auto& wire : inputs_) {
    wire->wait_setup();
    for (const auto& key : wire->get_keys()) {
      decoding_info.Set(bool(*key.data() & std::byte(0x01)), bit_offset);
      ++bit_offset;
    }
  }

  // If we receive output, we need to store the decoding information.  If the
  // evaluator receives output, we need to send them the decoding information.
  switch (output_recipient_) {
    case OutputRecipient::garbler:
      decoding_info_ = std::move(decoding_info);
      break;
    case OutputRecipient::evaluator:
      yao_provider_.send_bits_message(gate_id_, std::move(decoding_info));
      break;
    case OutputRecipient::both:
      decoding_info_ = std::move(decoding_info);
      yao_provider_.send_bits_message(gate_id_, decoding_info_);
      break;
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoOutputGateGarbler::evaluate_setup end", gate_id_));
    }
  }
}

void YaoOutputGateGarbler::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoOutputGateGarbler::evaluate_online start", gate_id_));
    }
  }

  if (output_recipient_ != OutputRecipient::evaluator) {
    std::vector<ENCRYPTO::BitVector<>> outputs;
    outputs.reserve(num_wires_);
    // Receive encoded output and decode it.
    auto plain_output = bits_future_.get() ^ decoding_info_;
    // Split the bits corresponding to the wires.
    std::size_t bit_offset = 0;
    for (const auto& wire : inputs_) {
      auto num_simd = wire->get_num_simd();
      outputs.push_back(plain_output.Subset(bit_offset, bit_offset + num_simd));
      bit_offset += num_simd;
    }
    output_promise_.set_value(std::move(outputs));
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoOutputGateGarbler::evaluate_online end", gate_id_));
    }
  }
}

YaoOutputGateEvaluator::YaoOutputGateEvaluator(std::size_t gate_id, YaoProvider& yao_provider,
                                               YaoWireVector&& in, OutputRecipient output_recipient)
    : BasicYaoOutputGate(gate_id, yao_provider, std::move(in), output_recipient) {
  if (output_recipient_ != OutputRecipient::garbler) {
    // We receive output, so the garbler needs to send us the decoding information.
    auto num_gates = detail::count_bits(inputs_);
    bits_future_ = yao_provider_.register_for_bits_message(gate_id, num_gates);
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoOutputGateEvaluator created", gate_id_));
    }
  }
}

ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>
YaoOutputGateEvaluator::get_output_future() {
  if (output_recipient_ == OutputRecipient::garbler) {
    throw std::logic_error("it's not the evaluator's output");
  }
  return output_promise_.get_future();
}

void YaoOutputGateEvaluator::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoOutputGateEvaluator::evaluate_setup start", gate_id_));
    }
  }

  if (output_recipient_ != OutputRecipient::garbler) {
    // Wait for the decoding information from the garbler.
    bits_future_.wait();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoOutputGateEvaluator::evaluate_setup end", gate_id_));
    }
  }
}

void YaoOutputGateEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoOutputGateEvaluator::evaluate_online start", gate_id_));
    }
  }

  // Compute the decoded output which consists of the lsb of each active key.
  auto num_bits = detail::count_bits(inputs_);
  ENCRYPTO::BitVector<> encoded_output(num_bits);

  // Collect the decoding information which consists of the lsb of each zero
  // key.
  std::size_t bit_offset = 0;
  for (const auto& wire : inputs_) {
    wire->wait_online();
    for (const auto& key : wire->get_keys()) {
      encoded_output.Set(bool(*key.data() & std::byte(0x01)), bit_offset);
      ++bit_offset;
    }
  }

  // If the garbler receives output, we need to send them the encoded output.
  switch (output_recipient_) {
    case OutputRecipient::garbler:
      yao_provider_.send_bits_message(gate_id_, std::move(encoded_output));
      break;
    case OutputRecipient::both:
      yao_provider_.send_bits_message(gate_id_, encoded_output);
      break;
    default:
      break;
  }

  if (output_recipient_ != OutputRecipient::garbler) {
    std::vector<ENCRYPTO::BitVector<>> outputs;
    outputs.reserve(num_wires_);
    // Receive decoding information and decode the encoded output.
    auto plain_output = std::move(encoded_output);
    plain_output ^= bits_future_.get();
    // Split the bits corresponding to the wires.
    std::size_t bit_offset = 0;
    for (const auto& wire : inputs_) {
      auto num_simd = wire->get_num_simd();
      outputs.push_back(plain_output.Subset(bit_offset, bit_offset + num_simd));
      bit_offset += num_simd;
    }
    output_promise_.set_value(std::move(outputs));
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoOutputGateEvaluator::evaluate_online end", gate_id_));
    }
  }
}

YaoINVGateGarbler::YaoINVGateGarbler(std::size_t gate_id, YaoProvider& yao_provider,
                                     YaoWireVector&& in)
    : BasicYaoUnaryGate(gate_id, yao_provider, std::move(in), false) {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoINVGateGarbler created", gate_id_));
    }
  }
}

void YaoINVGateGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoINVGateGarbler::evaluate_setup start", gate_id_));
    }
  }

  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& w_a = inputs_[wire_i];
    w_a->wait_setup();
    auto& w_o = outputs_[wire_i];
    w_o->get_keys() = w_a->get_keys() ^ yao_provider_.get_global_offset();
    w_o->set_setup_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoINVGateGarbler::evaluate_setup end", gate_id_));
    }
  }
}

YaoINVGateEvaluator::YaoINVGateEvaluator(std::size_t gate_id, YaoProvider& yao_provider,
                                         YaoWireVector&& in)
    : BasicYaoUnaryGate(gate_id, yao_provider, std::move(in), true) {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoINVGateEvaluator created", gate_id_));
    }
  }
}

YaoXORGateGarbler::YaoXORGateGarbler(std::size_t gate_id, YaoProvider& yao_provider,
                                     YaoWireVector&& in_a, YaoWireVector&& in_b)
    : BasicYaoBinaryGate(gate_id, yao_provider, std::move(in_a), std::move(in_b)) {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoXORGateGarbler created", gate_id_));
    }
  }
}

void YaoXORGateGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoXORGateGarbler::evaluate_setup start", gate_id_));
    }
  }

  // use freeXOR garbling
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& w_a = inputs_a_[wire_i];
    const auto& w_b = inputs_b_[wire_i];
    w_a->wait_setup();
    w_b->wait_setup();
    auto& w_o = outputs_[wire_i];
    w_o->get_keys() = w_a->get_keys() ^ w_b->get_keys();
    w_o->set_setup_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoXORGateGarbler::evaluate_setup end", gate_id_));
    }
  }
}

YaoXORGateEvaluator::YaoXORGateEvaluator(std::size_t gate_id, YaoProvider& yao_provider,
                                         YaoWireVector&& in_a, YaoWireVector&& in_b)
    : BasicYaoBinaryGate(gate_id, yao_provider, std::move(in_a), std::move(in_b)) {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoXORGateEvaluator created", gate_id_));
    }
  }
}

void YaoXORGateEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoXORGateEvaluator::evaluate_online start", gate_id_));
    }
  }

  // use freeXOR evaluation
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& w_a = inputs_a_[wire_i];
    const auto& w_b = inputs_b_[wire_i];
    w_a->wait_online();
    w_b->wait_online();
    auto& w_o = outputs_[wire_i];
    w_o->get_keys() = w_a->get_keys() ^ w_b->get_keys();
    w_o->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoXORGateEvaluator::evaluate_online end", gate_id_));
    }
  }
}

YaoANDGateGarbler::YaoANDGateGarbler(std::size_t gate_id, YaoProvider& yao_provider,
                                     YaoWireVector&& in_a, YaoWireVector&& in_b)
    : BasicYaoBinaryGate(gate_id, yao_provider, std::move(in_a), std::move(in_b)) {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoANDGateGarbler created", gate_id_));
    }
  }
}

void YaoANDGateGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoANDGateGarbler::evaluate_setup start", gate_id_));
    }
  }

  auto num_gates = detail::count_bits(inputs_a_);
  ENCRYPTO::block128_vector garbled_tables(YaoProvider::garbled_table_size * num_gates);
  std::size_t table_offset = 0;
  for (std::size_t wire_i = 0; wire_i < num_wires_; ++wire_i) {
    const auto& w_a = inputs_a_[wire_i];
    const auto& w_b = inputs_b_[wire_i];
    w_a->wait_setup();
    w_b->wait_setup();
    auto& w_o = outputs_[wire_i];
    yao_provider_.create_garbled_tables(gate_id_, w_a->get_keys(), w_b->get_keys(),
                                        &garbled_tables[table_offset], w_o->get_keys());
    w_o->set_setup_ready();
    table_offset += 2 * w_o->get_num_simd();
  }

  yao_provider_.send_blocks_message(gate_id_, std::move(garbled_tables));

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoANDGateGarbler::evaluate_setup end", gate_id_));
    }
  }
}

YaoANDGateEvaluator::YaoANDGateEvaluator(std::size_t gate_id, YaoProvider& yao_provider,
                                         YaoWireVector&& in_a, YaoWireVector&& in_b)
    : BasicYaoBinaryGate(gate_id, yao_provider, std::move(in_a), std::move(in_b)) {
  auto num_gates = detail::count_bits(inputs_a_);
  garbled_tables_fut_ = yao_provider_.register_for_blocks_message(
      gate_id_, num_gates * YaoProvider::garbled_table_size);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoANDGateEvaluator created", gate_id_));
    }
  }
}

void YaoANDGateEvaluator::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoANDGateEvaluator::evaluate_setup start", gate_id_));
    }
  }

  // nothing to do
  garbled_tables_fut_.wait();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoANDGateEvaluator::evaluate_setup end", gate_id_));
    }
  }
}

void YaoANDGateEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoANDGateEvaluator::evaluate_online start", gate_id_));
    }
  }

  auto num_wires = outputs_.size();
  auto garbled_tables = garbled_tables_fut_.get();

  std::size_t table_offset = 0;
  for (std::size_t wire_i = 0; wire_i < num_wires; ++wire_i) {
    const auto& w_a = inputs_a_[wire_i];
    const auto& w_b = inputs_b_[wire_i];
    w_a->wait_online();
    w_b->wait_online();
    auto& w_o = outputs_[wire_i];
    yao_provider_.evaluate_garbled_tables(gate_id_, w_a->get_keys(), w_b->get_keys(),
                                          &garbled_tables[table_offset], w_o->get_keys());
    w_o->set_online_ready();
    table_offset += 2 * w_o->get_num_simd();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoANDGateEvaluator::evaluate_online end", gate_id_));
    }
  }
}

}  // namespace MOTION::proto::yao
