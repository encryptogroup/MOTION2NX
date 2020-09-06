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

#include <parallel/algorithm>

#include "algorithm/circuit_loader.h"
#include "crypto/motion_base_provider.h"
#include "crypto/oblivious_transfer/ot_flavors.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "crypto/sharing_randomness_generator.h"
#include "tensor_op.h"
// #include "utility/bit_transpose.h"
#include "tools.h"
#include "utility/logger.h"
#include "yao_provider.h"

namespace MOTION::proto::yao {

namespace {

std::size_t padded_size(std::size_t data_size, std::size_t bit_size) {
  return data_size + (bit_size - data_size % bit_size);
}

}  // namespace

// A -> Y Garbler side

template <typename T>
ArithmeticGMWToYaoTensorConversionGarbler<T>::ArithmeticGMWToYaoTensorConversionGarbler(
    std::size_t gate_id, YaoProvider& yao_provider, const gmw::ArithmeticGMWTensorCP<T> input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<YaoTensor>(input->get_dimensions(), bit_size_)),
      addition_algo_(yao_provider_.get_circuit_loader().load_circuit(
          fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>), CircuitFormat::Bristol)) {
  auto& ot_provider = yao_provider_.get_ot_provider();
  ot_sender_ = ot_provider.RegisterSendGOT128(bit_size_ * data_size_);
  output_->get_keys().resize(bit_size_ * data_size_);
}

template <typename T>
ArithmeticGMWToYaoTensorConversionGarbler<T>::~ArithmeticGMWToYaoTensorConversionGarbler() =
    default;

template <typename T>
void ArithmeticGMWToYaoTensorConversionGarbler<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWToYaoTensorConversionGarbler::evaluate_setup start", gate_id_));
    }
  }

  garbler_input_keys_ = ENCRYPTO::block128_vector::make_random(bit_size_ * data_size_);
  evaluator_input_keys_ = ENCRYPTO::block128_vector::make_random(bit_size_ * data_size_);

  // prepare OTs for resharing evaluator's arithmetic share
  {
    ENCRYPTO::block128_vector ot_inputs(2 * bit_size_ * data_size_);
    const auto R = yao_provider_.get_global_offset();
    const auto total_size = bit_size_ * data_size_;
#pragma omp parallel for
    for (std::size_t i = 0; i < total_size; ++i) {
      ot_inputs[2 * i] = evaluator_input_keys_[i];
      ot_inputs[2 * i + 1] = evaluator_input_keys_[i] ^ R;
    }
    ot_sender_->SetInputs(std::move(ot_inputs));
  }

  // garble addition circuit
  yao_provider_.create_garbled_circuit(gate_id_, data_size_, addition_algo_, garbler_input_keys_,
                                       evaluator_input_keys_, garbled_tables_, output_->get_keys(),
                                       true);
  yao_provider_.CommMixin::send_blocks_message(1, gate_id_, std::move(garbled_tables_), 1);
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWToYaoTensorConversionGarbler::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticGMWToYaoTensorConversionGarbler<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWToYaoTensorConversionGarbler::evaluate_online start", gate_id_));
    }
  }

  // send garbler's keys
  {
    input_->wait_online();
    const auto& share = input_->get_share();
    assert(share.size() == data_size_);
    auto msg = garbler_input_keys_;
    const auto R = yao_provider_.get_global_offset();
    // implicit transpose of input data
    // now we have all the first bist, then all the second bits, and so on
#pragma omp parallel for
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      T value = share[int_i];
      for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
        if (value & (T(1) << bit_j)) {
          msg[bit_j * data_size_ + int_i] ^= R;
        }
      }
    }
    yao_provider_.CommMixin::send_blocks_message(1, gate_id_, std::move(msg), 0);
  }

  // send evaluator's keys via OTs
  { ot_sender_->SendMessages(); }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWToYaoTensorConversionGarbler::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWToYaoTensorConversionGarbler<std::uint32_t>;
template class ArithmeticGMWToYaoTensorConversionGarbler<std::uint64_t>;

// A -> Y Evaluator side

template <typename T>
ArithmeticGMWToYaoTensorConversionEvaluator<T>::ArithmeticGMWToYaoTensorConversionEvaluator(
    std::size_t gate_id, YaoProvider& yao_provider, const gmw::ArithmeticGMWTensorCP<T> input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<YaoTensor>(input->get_dimensions(), bit_size_)),
      addition_algo_(yao_provider_.get_circuit_loader().load_circuit(
          fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>), CircuitFormat::Bristol)) {
  // assert(input->get_share().size() % bit_size_ == 0);
  auto& ot_provider = yao_provider_.get_ot_provider();
  ot_receiver_ = ot_provider.RegisterReceiveGOT128(bit_size_ * data_size_);
  garbler_input_keys_future_ =
      yao_provider_.CommMixin::register_for_blocks_message(0, gate_id, bit_size_ * data_size_, 0);
  garbled_tables_future_ = yao_provider_.CommMixin::register_for_blocks_message(
      0, gate_id, 2 * (bit_size_ - 1) * data_size_, 1);
  output_->get_keys().resize(bit_size_ * data_size_);
}

template <typename T>
ArithmeticGMWToYaoTensorConversionEvaluator<T>::~ArithmeticGMWToYaoTensorConversionEvaluator() =
    default;

template <typename T>
void ArithmeticGMWToYaoTensorConversionEvaluator<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWToYaoTensorConversionEvaluator::evaluate_online start", gate_id_));
    }
  }

  // receive garbler's keys
  { garbler_input_keys_ = garbler_input_keys_future_.get(); }
  // receive evaluator's keys
  {
    input_->wait_online();
    std::vector<T> share(padded_size(data_size_, bit_size_));
    std::copy_n(std::begin(input_->get_share()), data_size_, std::begin(share));
    ENCRYPTO::BitVector<> ot_choices;
    {
      // XXX stupid implementation
      ot_choices.Resize(data_size_ * bit_size_);
      const auto& share = input_->get_share();
      for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
        T value = share[int_i];
        for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
          ot_choices.Set(bool(value & (T(1) << bit_j)), bit_j * data_size_ + int_i);
        }
      }
    }
    // {
    //   ot_choices.Reserve(Helpers::Convert::BitsToBytes(data_size_ * bit_size_));
    //   auto padded_data_size = padded_size(data_size_, bit_size_);
    //   std::array<ENCRYPTO::BitVector<>, bit_size_> rows;
    //   std::array<std::byte*, bit_size_> row_pointers;
    //   for (std::size_t i = 0; i < bit_size_; ++i) {
    //     rows[i].Resize(padded_data_size);
    //     row_pointers[i] = rows[i].GetMutableData().data();
    //   }
    //   // do a bitwise transpose
    //   auto N = data_size_ / bit_size_;
    //   transpose_bit_rectangular(row_pointers.data(), share.data(), N);
    //   // concatenate the rows
    //   for (std::size_t row_i = 0; row_i < bit_size_; ++row_i) {
    //     ot_choices.Append(row_pointers[row_i], data_size_);
    //   }
    // }
    ot_receiver_->SetChoices(ot_choices);
    ot_receiver_->SendCorrections();
    ot_receiver_->ComputeOutputs();
    evaluator_input_keys_ = ot_receiver_->GetOutputs();
  }
  // evaluate garbled circuit
  {
    garbled_tables_ = garbled_tables_future_.get();
    yao_provider_.evaluate_garbled_circuit(gate_id_, data_size_, addition_algo_,
                                           garbler_input_keys_, evaluator_input_keys_,
                                           garbled_tables_, output_->get_keys(), true);
    output_->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticGMWToYaoTensorConversionEvaluator::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticGMWToYaoTensorConversionEvaluator<std::uint32_t>;
template class ArithmeticGMWToYaoTensorConversionEvaluator<std::uint64_t>;

// Y -> A Garbler side

template <typename T>
YaoToArithmeticGMWTensorConversionGarbler<T>::YaoToArithmeticGMWTensorConversionGarbler(
    std::size_t gate_id, YaoProvider& yao_provider, const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<gmw::ArithmeticGMWTensor<T>>(input->get_dimensions())),
      addition_algo_(yao_provider_.get_circuit_loader().load_circuit(
          fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>), CircuitFormat::Bristol)) {
}

template <typename T>
void YaoToArithmeticGMWTensorConversionGarbler<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticGMWTensorConversionGarbler::evaluate_setup start", gate_id_));
    }
  }

  // generate mask
  auto mask = Helpers::RandomVector<T>(data_size_);
  // create GMW share
  {
    auto& mbp = yao_provider_.get_motion_base_provider();
    auto& rng = mbp.get_their_randomness_generator(1);
    auto& share = output_->get_share();
    share = rng.GetUnsigned<T>(gate_id_, data_size_);
    __gnu_parallel::transform(std::begin(share), std::end(share), std::begin(mask),
                              std::begin(share), std::minus{});
    output_->set_online_ready();
  }
  mask_input_keys_ = ENCRYPTO::block128_vector::make_random(bit_size_ * data_size_);
  // share mask in Yao
  {
    ENCRYPTO::block128_vector message = mask_input_keys_;
    const auto& R = yao_provider_.get_global_offset();
    auto bit_vectors = ENCRYPTO::ToInput(mask);
    assert(bit_vectors.size() == bit_size_);
#pragma omp parallel for
    for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
      const auto& bv = bit_vectors[bit_j];
      assert(bv.GetSize() == data_size_);
      for (std::size_t i = 0; i < data_size_; ++i) {
        if (bv.Get(i)) {
          message[bit_j * data_size_ + i] ^= R;
        }
      }
    }
    yao_provider_.CommMixin::send_blocks_message(1, gate_id_, std::move(message), 0);
  }
  {
    // garble addition circuit
    input_->wait_setup();
    yao_provider_.create_garbled_circuit(gate_id_, data_size_, addition_algo_, input_->get_keys(),
                                         mask_input_keys_, garbled_tables_, output_keys_, true);
    yao_provider_.CommMixin::send_blocks_message(1, gate_id_, std::move(garbled_tables_), 1);
    // send output information
    ENCRYPTO::BitVector<> output_info(bit_size_ * data_size_);
    for (std::size_t i = 0; i < output_keys_.size(); ++i) {
      output_info.Set((*output_keys_[i].data() & std::byte{0x01}) != std::byte{0x00}, i);
    }
    yao_provider_.CommMixin::send_bits_message(1, gate_id_, std::move(output_info), 2);
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticGMWTensorConversionGarbler::evaluate_setup end", gate_id_));
    }
  }
}

template class YaoToArithmeticGMWTensorConversionGarbler<std::uint32_t>;
template class YaoToArithmeticGMWTensorConversionGarbler<std::uint64_t>;

// Y -> A Evaluator side

template <typename T>
YaoToArithmeticGMWTensorConversionEvaluator<T>::YaoToArithmeticGMWTensorConversionEvaluator(
    std::size_t gate_id, YaoProvider& yao_provider, const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<gmw::ArithmeticGMWTensor<T>>(input->get_dimensions())),
      addition_algo_(yao_provider_.get_circuit_loader().load_circuit(
          fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>), CircuitFormat::Bristol)) {
  garbler_input_keys_future_ =
      yao_provider_.CommMixin::register_for_blocks_message(0, gate_id, bit_size_ * data_size_, 0);
  garbled_tables_future_ = yao_provider_.CommMixin::register_for_blocks_message(
      0, gate_id, 2 * (bit_size_ - 1) * data_size_, 1);
  output_info_future_ =
      yao_provider_.CommMixin::register_for_bits_message(0, gate_id_, bit_size_ * data_size_, 2);
}

template <typename T>
void YaoToArithmeticGMWTensorConversionEvaluator<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticGMWTensorConversionEvaluator::evaluate_setup start", gate_id_));
    }
  }

  auto& mbp = yao_provider_.get_motion_base_provider();
  auto& rng = mbp.get_my_randomness_generator(0);
  output_->get_share() = rng.GetUnsigned<T>(gate_id_, data_size_);

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticGMWTensorConversionEvaluator::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void YaoToArithmeticGMWTensorConversionEvaluator<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticGMWTensorConversionEvaluator::evaluate_online start", gate_id_));
    }
  }

  auto mask_input_keys = garbler_input_keys_future_.get();
  auto garbled_tables = garbled_tables_future_.get();
  ENCRYPTO::block128_vector output_keys;
  input_->wait_online();
  yao_provider_.evaluate_garbled_circuit(gate_id_, data_size_, addition_algo_, input_->get_keys(),
                                         mask_input_keys, garbled_tables, output_keys, true);
  // decode output
  {
    auto padded_data_size = padded_size(data_size_, bit_size_);
    auto output_info = output_info_future_.get();
    ENCRYPTO::BitVector<> encoded_output(bit_size_ * padded_data_size);
    std::size_t bit_offset = 0;
    for (const auto& key : output_keys) {
      encoded_output.Set(bool(*key.data() & std::byte{0x01}), bit_offset);
      ++bit_offset;
    }
    encoded_output ^= output_info;
    std::vector<T> masked_value_int(padded_data_size);
    {
      // XXX stupid implementation
#pragma omp parallel for
      for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
        T& value = masked_value_int[int_i];
        for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
          if (encoded_output.Get(bit_j * data_size_ + int_i)) {
            value |= (T(1) << bit_j);
          }
        }
      }
    }
    // {
    //   std::array<const std::byte*, bit_size_> row_pointers;
    //   for (std::size_t i = 0; i < bit_size_; ++i) {
    //     row_pointers[i] = encoded_output.GetData().data() + i * (data_size_ / 8);
    //   }
    //   auto N = data_size_ / bit_size_;
    //   transpose_bit_rectangular(masked_value_int.data(), row_pointers.data(), N);
    // }

    auto& share = output_->get_share();
    __gnu_parallel::transform(std::begin(masked_value_int),
                              std::begin(masked_value_int) + data_size_, std::begin(share),
                              std::begin(share), std::minus{});
    output_->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticGMWTensorConversionEvaluator::evaluate_online end", gate_id_));
    }
  }
}

template class YaoToArithmeticGMWTensorConversionEvaluator<std::uint32_t>;
template class YaoToArithmeticGMWTensorConversionEvaluator<std::uint64_t>;

// Y -> B Garbler side
YaoToBooleanGMWTensorConversionGarbler::YaoToBooleanGMWTensorConversionGarbler(
    std::size_t gate_id, YaoProvider& yao_provider, const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(std::move(input)) {
  const auto& tensor_dims = input_->get_dimensions();
  output_ = std::make_shared<gmw::BooleanGMWTensor>(tensor_dims, bit_size_);
  auto& shares = output_->get_share();
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    shares.at(bit_j) = ENCRYPTO::BitVector<>(data_size_);
  }
}

void YaoToBooleanGMWTensorConversionGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanGMWTensorConversionGarbler::evaluate_setup start", gate_id_));
    }
  }

  input_->wait_setup();
  auto& input_keys = input_->get_keys();
  auto& output_share = output_->get_share();
#pragma omp parallel for
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    get_lsbs_from_keys(output_share[bit_j], input_keys.data() + bit_j * data_size_, data_size_);
  }
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanGMWTensorConversionGarbler::evaluate_setup end", gate_id_));
    }
  }
}

// Y -> B Evaluator side
YaoToBooleanGMWTensorConversionEvaluator::YaoToBooleanGMWTensorConversionEvaluator(
    std::size_t gate_id, YaoProvider& yao_provider, const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(std::move(input)) {
  const auto& tensor_dims = input_->get_dimensions();
  output_ = std::make_shared<gmw::BooleanGMWTensor>(tensor_dims, bit_size_);
  auto& shares = output_->get_share();
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    shares.at(bit_j) = ENCRYPTO::BitVector<>(data_size_);
  }
}

void YaoToBooleanGMWTensorConversionEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanGMWTensorConversionEvaluator::evaluate_online start", gate_id_));
    }
  }

  input_->wait_online();
  auto& input_keys = input_->get_keys();
  auto& output_share = output_->get_share();
#pragma omp parallel for
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    get_lsbs_from_keys(output_share[bit_j], input_keys.data() + bit_j * data_size_, data_size_);
  }
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanGMWTensorConversionEvaluator::evaluate_online end", gate_id_));
    }
  }
}

// BEAVY A -> Y Garbler side

template <typename T>
ArithmeticBEAVYToYaoTensorConversionGarbler<T>::ArithmeticBEAVYToYaoTensorConversionGarbler(
    std::size_t gate_id, YaoProvider& yao_provider, const beavy::ArithmeticBEAVYTensorCP<T> input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<YaoTensor>(input->get_dimensions(), bit_size_)),
      addition_algo_(yao_provider_.get_circuit_loader().load_circuit(
          fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>), CircuitFormat::Bristol)) {
  auto& ot_provider = yao_provider_.get_ot_provider();
  ot_sender_ = ot_provider.RegisterSendFixedXCOT128(bit_size_ * data_size_);
  output_->get_keys().resize(bit_size_ * data_size_);
}

template <typename T>
ArithmeticBEAVYToYaoTensorConversionGarbler<T>::~ArithmeticBEAVYToYaoTensorConversionGarbler() =
    default;

template <typename T>
void ArithmeticBEAVYToYaoTensorConversionGarbler<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYToYaoTensorConversionGarbler::evaluate_setup start", gate_id_));
    }
  }

  // sample garbler's input keys
  garbler_input_keys_ = ENCRYPTO::block128_vector::make_random(bit_size_ * data_size_);

  // run OTs for resharing evaluator's arithmetic secret share
  ot_sender_->SetCorrelation(yao_provider_.get_global_offset());
  ot_sender_->SendMessages();
  ot_sender_->ComputeOutputs();
  evaluator_input_keys_ = ot_sender_->GetOutputs();

  // garble addition circuit
  yao_provider_.create_garbled_circuit(gate_id_, data_size_, addition_algo_, garbler_input_keys_,
                                       evaluator_input_keys_, garbled_tables_, output_->get_keys(),
                                       true);
  yao_provider_.CommMixin::send_blocks_message(1, gate_id_, std::move(garbled_tables_), 1);
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYToYaoTensorConversionGarbler::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYToYaoTensorConversionGarbler<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYToYaoTensorConversionGarbler::evaluate_online start", gate_id_));
    }
  }

  // send garbler's keys
  {
    input_->wait_online();
    const auto& public_share = input_->get_public_share();
    const auto& secret_share = input_->get_secret_share();
    assert(public_share.size() == data_size_);
    assert(secret_share.size() == data_size_);
    auto msg = garbler_input_keys_;
    const auto R = yao_provider_.get_global_offset();
    // implicit transpose of input data
    // now we have all the first bist, then all the second bits, and so on
#pragma omp parallel for
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      T value = public_share[int_i] - secret_share[int_i];
      for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
        if (value & (T(1) << bit_j)) {
          msg[bit_j * data_size_ + int_i] ^= R;
        }
      }
    }
    yao_provider_.CommMixin::send_blocks_message(1, gate_id_, std::move(msg), 0);
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYToYaoTensorConversionGarbler::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYToYaoTensorConversionGarbler<std::uint32_t>;
template class ArithmeticBEAVYToYaoTensorConversionGarbler<std::uint64_t>;

// BEAVY A -> Y Evaluator side

template <typename T>
ArithmeticBEAVYToYaoTensorConversionEvaluator<T>::ArithmeticBEAVYToYaoTensorConversionEvaluator(
    std::size_t gate_id, YaoProvider& yao_provider, const beavy::ArithmeticBEAVYTensorCP<T> input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<YaoTensor>(input->get_dimensions(), bit_size_)),
      addition_algo_(yao_provider_.get_circuit_loader().load_circuit(
          fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>), CircuitFormat::Bristol)) {
  // assert(input->get_share().size() % bit_size_ == 0);
  auto& ot_provider = yao_provider_.get_ot_provider();
  ot_receiver_ = ot_provider.RegisterReceiveFixedXCOT128(bit_size_ * data_size_);
  garbler_input_keys_future_ =
      yao_provider_.CommMixin::register_for_blocks_message(0, gate_id, bit_size_ * data_size_, 0);
  garbled_tables_future_ = yao_provider_.CommMixin::register_for_blocks_message(
      0, gate_id, 2 * (bit_size_ - 1) * data_size_, 1);
  output_->get_keys().resize(bit_size_ * data_size_);
}

template <typename T>
ArithmeticBEAVYToYaoTensorConversionEvaluator<T>::~ArithmeticBEAVYToYaoTensorConversionEvaluator() =
    default;

template <typename T>
void ArithmeticBEAVYToYaoTensorConversionEvaluator<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYToYaoTensorConversionEvaluator::evaluate_setup start",
          gate_id_));
    }
  }

  // receive evaluator's keys
  {
    input_->wait_setup();
    std::vector<T> share(padded_size(data_size_, bit_size_));
    const auto& my_secret_share = input_->get_secret_share();
    __gnu_parallel::transform(std::begin(my_secret_share), std::end(my_secret_share),
                              std::begin(share), std::negate{});
    ENCRYPTO::BitVector<> ot_choices;
    {
      // XXX stupid implementation
      ot_choices.Resize(data_size_ * bit_size_);
      for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
        T value = share[int_i];
        for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
          ot_choices.Set(bool(value & (T(1) << bit_j)), bit_j* data_size_ + int_i);
        }
      }
    }
    // {
    //   ot_choices.Reserve(Helpers::Convert::BitsToBytes(data_size_ * bit_size_));
    //   auto padded_data_size = padded_size(data_size_, bit_size_);
    //   std::array<ENCRYPTO::BitVector<>, bit_size_> rows;
    //   std::array<std::byte*, bit_size_> row_pointers;
    //   for (std::size_t i = 0; i < bit_size_; ++i) {
    //     rows[i].Resize(padded_data_size);
    //     row_pointers[i] = rows[i].GetMutableData().data();
    //   }
    //   // do a bitwise transpose
    //   auto N = data_size_ / bit_size_;
    //   transpose_bit_rectangular(row_pointers.data(), share.data(), N);
    //   // concatenate the rows
    //   for (std::size_t row_i = 0; row_i < bit_size_; ++row_i) {
    //     ot_choices.Append(row_pointers[row_i], data_size_);
    //   }
    // }
    ot_receiver_->SetChoices(ot_choices);
    ot_receiver_->SendCorrections();
    ot_receiver_->ComputeOutputs();
    evaluator_input_keys_ = ot_receiver_->GetOutputs();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYToYaoTensorConversionEvaluator::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void ArithmeticBEAVYToYaoTensorConversionEvaluator<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYToYaoTensorConversionEvaluator::evaluate_online start",
          gate_id_));
    }
  }

  // receive garbler's keys
  { garbler_input_keys_ = garbler_input_keys_future_.get(); }
  // evaluate garbled circuit
  {
    garbled_tables_ = garbled_tables_future_.get();
    yao_provider_.evaluate_garbled_circuit(gate_id_, data_size_, addition_algo_,
                                           garbler_input_keys_, evaluator_input_keys_,
                                           garbled_tables_, output_->get_keys(), true);
    output_->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: ArithmeticBEAVYToYaoTensorConversionEvaluator::evaluate_online end", gate_id_));
    }
  }
}

template class ArithmeticBEAVYToYaoTensorConversionEvaluator<std::uint32_t>;
template class ArithmeticBEAVYToYaoTensorConversionEvaluator<std::uint64_t>;

// Y -> BEAVY A Garbler side

template <typename T>
YaoToArithmeticBEAVYTensorConversionGarbler<T>::YaoToArithmeticBEAVYTensorConversionGarbler(
    std::size_t gate_id, YaoProvider& yao_provider, const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<beavy::ArithmeticBEAVYTensor<T>>(input->get_dimensions())),
      masked_value_public_share_future_(
          yao_provider.register_for_ints_message<T>(1, gate_id_, data_size_)),
      addition_algo_(yao_provider_.get_circuit_loader().load_circuit(
          fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>), CircuitFormat::Bristol)) {
}

template <typename T>
void YaoToArithmeticBEAVYTensorConversionGarbler<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticBEAVYTensorConversionGarbler::evaluate_setup start", gate_id_));
    }
  }

  // generate mask
  auto mask = Helpers::RandomVector<T>(data_size_);
  // create BEAVY share
  {
    auto& mbp = yao_provider_.get_motion_base_provider();
    auto& rng = mbp.get_their_randomness_generator(1);
    auto& public_share = output_->get_public_share();
    auto& secret_share = output_->get_secret_share();
    // generate shares:
    //   Delta = public_share of mask ||
    //   delta_1 = secret_share^1 of mask ||
    //   delta'_0 = secret_share^0 of masked value
    auto random_ints = rng.GetUnsigned<T>(gate_id_, 3 * data_size_);
    public_share.resize(data_size_);
    // initialize public_share with Delta (Delta - Delta' is computed in the online phase)
    std::copy_n(std::begin(random_ints), data_size_, std::begin(public_share));
    secret_share.resize(data_size_);
    // the secret_share is delta'_0 - delta_0 = delta'_0 + delta_1 - Delta + r
#pragma omp parallel for
    for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
      secret_share[int_i] = random_ints[2 * data_size_ + int_i] + random_ints[data_size_ + int_i] -
                            random_ints[int_i] + mask[int_i];
    }
    output_->set_setup_ready();
  }
  mask_input_keys_ = ENCRYPTO::block128_vector::make_random(bit_size_ * data_size_);
  // share mask in Yao
  {
    ENCRYPTO::block128_vector message = mask_input_keys_;
    const auto& R = yao_provider_.get_global_offset();
    auto bit_vectors = ENCRYPTO::ToInput(mask);
    assert(bit_vectors.size() == bit_size_);
#pragma omp parallel for
    for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
      const auto& bv = bit_vectors[bit_j];
      assert(bv.GetSize() == data_size_);
      for (std::size_t i = 0; i < data_size_; ++i) {
        if (bv.Get(i)) {
          message[bit_j * data_size_ + i] ^= R;
        }
      }
    }
    yao_provider_.CommMixin::send_blocks_message(1, gate_id_, std::move(message), 0);
  }
  {
    // garble addition circuit
    input_->wait_setup();
    yao_provider_.create_garbled_circuit(gate_id_, data_size_, addition_algo_, input_->get_keys(),
                                         mask_input_keys_, garbled_tables_, output_keys_, true);
    yao_provider_.CommMixin::send_blocks_message(1, gate_id_, std::move(garbled_tables_), 1);
    // send output information
    ENCRYPTO::BitVector<> output_info(bit_size_ * data_size_);
    for (std::size_t i = 0; i < output_keys_.size(); ++i) {
      output_info.Set((*output_keys_[i].data() & std::byte{0x01}) != std::byte{0x00}, i);
    }
    yao_provider_.CommMixin::send_bits_message(1, gate_id_, std::move(output_info), 2);
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticBEAVYTensorConversionGarbler::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void YaoToArithmeticBEAVYTensorConversionGarbler<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticBEAVYTensorConversionGarbler::evaluate_online start", gate_id_));
    }
  }

  // receive public share of shared masked value
  auto& public_share = output_->get_public_share();
  auto masked_value_public_share = masked_value_public_share_future_.get();
  __gnu_parallel::transform(std::begin(masked_value_public_share),
                            std::end(masked_value_public_share), std::begin(public_share),
                            std::begin(public_share), std::minus{});
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticBEAVYTensorConversionGarbler::evaluate_online end", gate_id_));
    }
  }
}

template class YaoToArithmeticBEAVYTensorConversionGarbler<std::uint32_t>;
template class YaoToArithmeticBEAVYTensorConversionGarbler<std::uint64_t>;

// Y -> BEAVY A Evaluator side

template <typename T>
YaoToArithmeticBEAVYTensorConversionEvaluator<T>::YaoToArithmeticBEAVYTensorConversionEvaluator(
    std::size_t gate_id, YaoProvider& yao_provider, const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<beavy::ArithmeticBEAVYTensor<T>>(input->get_dimensions())),
      addition_algo_(yao_provider_.get_circuit_loader().load_circuit(
          fmt::format("int_add{}_size.bristol", ENCRYPTO::bit_size_v<T>), CircuitFormat::Bristol)) {
  garbler_input_keys_future_ =
      yao_provider_.CommMixin::register_for_blocks_message(0, gate_id, bit_size_ * data_size_, 0);
  garbled_tables_future_ = yao_provider_.CommMixin::register_for_blocks_message(
      0, gate_id, 2 * (bit_size_ - 1) * data_size_, 1);
  output_info_future_ =
      yao_provider_.CommMixin::register_for_bits_message(0, gate_id_, bit_size_ * data_size_, 2);
}

template <typename T>
void YaoToArithmeticBEAVYTensorConversionEvaluator<T>::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticBEAVYTensorConversionEvaluator::evaluate_setup start",
          gate_id_));
    }
  }

  auto& mbp = yao_provider_.get_motion_base_provider();
  auto& rng = mbp.get_my_randomness_generator(0);
  // secret_share of masked value
  masked_value_secret_share_ = Helpers::RandomVector<T>(data_size_);
  auto& public_share = output_->get_public_share();
  auto& secret_share = output_->get_secret_share();
  // initialize public_share with -Delta (Delta' is added in the online phase)
  // generate shares:
  //   Delta = public_share of mask ||
  //   delta_1 = secret_share^1 of mask ||
  //   delta'_0 = secret_share^0 of masked value
  auto random_ints = rng.GetUnsigned<T>(gate_id_, 3 * data_size_);
  public_share.resize(data_size_);
  // initialize public_share with Delta (Delta' - Delta is computed in the online phase)
  std::copy_n(std::begin(random_ints), data_size_, std::begin(public_share));
  secret_share.resize(data_size_);
#pragma omp parallel for
  for (std::size_t int_j = 0; int_j < data_size_; ++int_j) {
    secret_share[int_j] = masked_value_secret_share_[int_j] - random_ints[data_size_ + int_j];
  }
  // prepared public_share of masked value (add masked value to this in online phase)
  masked_value_public_share_.resize(data_size_);
  __gnu_parallel::transform(std::begin(masked_value_secret_share_),
                            std::end(masked_value_secret_share_),
                            std::begin(random_ints) + 2 * data_size_,
                            std::begin(masked_value_public_share_), std::plus{});

  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticBEAVYTensorConversionEvaluator::evaluate_setup end", gate_id_));
    }
  }
}

template <typename T>
void YaoToArithmeticBEAVYTensorConversionEvaluator<T>::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticBEAVYTensorConversionEvaluator::evaluate_online start",
          gate_id_));
    }
  }

  auto mask_input_keys = garbler_input_keys_future_.get();
  auto garbled_tables = garbled_tables_future_.get();
  ENCRYPTO::block128_vector output_keys;
  input_->wait_online();
  yao_provider_.evaluate_garbled_circuit(gate_id_, data_size_, addition_algo_, input_->get_keys(),
                                         mask_input_keys, garbled_tables, output_keys, true);
  // decode output
  {
    auto padded_data_size = padded_size(data_size_, bit_size_);
    auto output_info = output_info_future_.get();
    ENCRYPTO::BitVector<> encoded_output(bit_size_ * padded_data_size);
    std::size_t bit_offset = 0;
    for (const auto& key : output_keys) {
      encoded_output.Set(bool(*key.data() & std::byte{0x01}), bit_offset);
      ++bit_offset;
    }
    encoded_output ^= output_info;
    std::vector<T> masked_value_int(padded_data_size);
    {
      // XXX stupid implementation
#pragma omp parallel for
      for (std::size_t int_i = 0; int_i < data_size_; ++int_i) {
        T& value = masked_value_int[int_i];
        for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
          if (encoded_output.Get(bit_j * data_size_ + int_i)) {
            value |= (T(1) << bit_j);
          }
        }
      }
    }
    // {
    //   std::array<const std::byte*, bit_size_> row_pointers;
    //   for (std::size_t i = 0; i < bit_size_; ++i) {
    //     row_pointers[i] = encoded_output.GetData().data() + i * (data_size_ / 8);
    //   }
    //   auto N = data_size_ / bit_size_;
    //   transpose_bit_rectangular(masked_value_int.data(), row_pointers.data(), N);
    // }

    // reshare masked value in Arithemtic BEAVY and subtract shared mask
    __gnu_parallel::transform(std::begin(masked_value_int), std::end(masked_value_int),
                              std::begin(masked_value_public_share_),
                              std::begin(masked_value_public_share_), std::plus{});
    yao_provider_.send_ints_message(0, gate_id_, masked_value_public_share_);
    auto& public_share = output_->get_public_share();
    __gnu_parallel::transform(std::begin(masked_value_public_share_),
                              std::end(masked_value_public_share_), std::begin(public_share),
                              std::begin(public_share), std::minus{});
    output_->set_online_ready();
  }

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToArithmeticBEAVYTensorConversionEvaluator::evaluate_online end", gate_id_));
    }
  }
}

template class YaoToArithmeticBEAVYTensorConversionEvaluator<std::uint32_t>;
template class YaoToArithmeticBEAVYTensorConversionEvaluator<std::uint64_t>;

// Y -> beta Garbler side

YaoToBooleanBEAVYTensorConversionGarbler::YaoToBooleanBEAVYTensorConversionGarbler(
    std::size_t gate_id, YaoProvider& yao_provider, const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(std::move(input)) {
  const auto& tensor_dims = input_->get_dimensions();
  output_ = std::make_shared<beavy::BooleanBEAVYTensor>(tensor_dims, bit_size_);
  public_share_future_ = yao_provider_.register_for_bits_message(gate_id, bit_size_ * data_size_);
}

void YaoToBooleanBEAVYTensorConversionGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanBEAVYTensorConversionGarbler::evaluate_setup start", gate_id_));
    }
  }

  input_->wait_setup();
  const auto& keys = input_->get_keys();
  auto& sshares = output_->get_secret_share();
  assert(sshares.size() == bit_size_);
#pragma omp parallel for
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    ENCRYPTO::BitVector<> lsbs;
    get_lsbs_from_keys(lsbs, keys.data() + bit_j * data_size_, data_size_);
    sshares[bit_j] = std::move(lsbs);
  }
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanBEAVYTensorConversionGarbler::evaluate_setup end", gate_id_));
    }
  }
}

void YaoToBooleanBEAVYTensorConversionGarbler::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanBEAVYTensorConversionGarbler::evaluate_online start", gate_id_));
    }
  }

  const auto public_share = public_share_future_.get();
  auto& pshares = output_->get_public_share();
  assert(pshares.size() == bit_size_);
#pragma omp parallel for
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    pshares[bit_j] = public_share.Subset(bit_j * data_size_, (bit_j + 1) * data_size_);
  }
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanBEAVYTensorConversionGarbler::evaluate_online end", gate_id_));
    }
  }
}

// Y -> beta Evaluator side

YaoToBooleanBEAVYTensorConversionEvaluator::YaoToBooleanBEAVYTensorConversionEvaluator(
    std::size_t gate_id, YaoProvider& yao_provider, const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(std::move(input)) {
  const auto& tensor_dims = input_->get_dimensions();
  output_ = std::make_shared<beavy::BooleanBEAVYTensor>(tensor_dims, bit_size_);
}

void YaoToBooleanBEAVYTensorConversionEvaluator::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanBEAVYTensorConversionEvaluator::evaluate_setup start", gate_id_));
    }
  }

  auto& sshares = output_->get_secret_share();
  assert(sshares.size() == bit_size_);
#pragma omp parallel for
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    sshares[bit_j] = ENCRYPTO::BitVector<>::Random(data_size_);
  }
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanBEAVYTensorConversionEvaluator::evaluate_setup end", gate_id_));
    }
  }
}

void YaoToBooleanBEAVYTensorConversionEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanBEAVYTensorConversionEvaluator::evaluate_online start", gate_id_));
    }
  }

  input_->wait_online();
  auto& keys = input_->get_keys();
  auto& pshares = output_->get_public_share();
  const auto& sshares = output_->get_secret_share();
  ENCRYPTO::BitVector<> public_share;
  public_share.Reserve(Helpers::Convert::BitsToBytes(bit_size_ * data_size_));
  for (std::size_t bit_j = 0; bit_j < bit_size_; ++bit_j) {
    ENCRYPTO::BitVector<> tmp;
    get_lsbs_from_keys(tmp, keys.data() + bit_j * data_size_, data_size_);
    tmp ^= sshares[bit_j];
    public_share.Append(tmp);
    pshares[bit_j] = std::move(tmp);
  }
  output_->set_online_ready();
  yao_provider_.send_bits_message(gate_id_, std::move(public_share));

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format(
          "Gate {}: YaoToBooleanBEAVYTensorConversionEvaluator::evaluate_online end", gate_id_));
    }
  }
}

// Relu

YaoTensorReluGarbler::YaoTensorReluGarbler(std::size_t gate_id, YaoProvider& yao_provider,
                                           const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<YaoTensor>(input->get_dimensions(), bit_size_)),
      relu_algo_(yao_provider_.get_circuit_loader().load_relu_circuit(bit_size_)) {
  garbled_tables_.resize(data_size_ * (bit_size_ - 1));
  output_->get_keys().resize(bit_size_);
}

void YaoTensorReluGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoTensorReluGarbler::evaluate_setup start", gate_id_));
    }
  }

  // garble ReLU circuit
  yao_provider_.create_garbled_circuit(gate_id_, data_size_, relu_algo_, input_->get_keys(), {},
                                       garbled_tables_, output_->get_keys(), true);
  yao_provider_.send_blocks_message(gate_id_, std::move(garbled_tables_));
  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(fmt::format("Gate {}: YaoTensorReluGarbler::evaluate_setup end", gate_id_));
    }
  }
}

YaoTensorReluEvaluator::YaoTensorReluEvaluator(std::size_t gate_id, YaoProvider& yao_provider,
                                               const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<YaoTensor>(input->get_dimensions(), bit_size_)),
      relu_algo_(yao_provider_.get_circuit_loader().load_relu_circuit(bit_size_)) {
  garbled_tables_future_ =
      yao_provider_.register_for_blocks_message(gate_id, 2 * (bit_size_ - 1) * data_size_);
  output_->get_keys().resize(bit_size_ * data_size_);
}

void YaoTensorReluEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoTensorReluEvaluator::evaluate_online start", gate_id_));
    }
  }

  // evaluate ReLU circuit
  const auto garbled_tables = garbled_tables_future_.get();
  yao_provider_.evaluate_garbled_circuit(gate_id_, data_size_, relu_algo_, input_->get_keys(), {},
                                         garbled_tables, output_->get_keys(), true);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoTensorReluEvaluator::evaluate_online end", gate_id_));
    }
  }
}

// MaxPool

YaoTensorMaxPoolGarbler::YaoTensorMaxPoolGarbler(std::size_t gate_id, YaoProvider& yao_provider,
                                                 tensor::MaxPoolOp maxpool_op,
                                                 const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      maxpool_op_(maxpool_op),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<YaoTensor>(maxpool_op_.get_output_tensor_dims(), bit_size_)),
      maxpool_algo_(yao_provider_.get_circuit_loader().load_maxpool_circuit(
          bit_size_, maxpool_op_.compute_kernel_size())) {
  assert(maxpool_op_.verify());
  output_->get_keys().resize(data_size_ * bit_size_);
}

void YaoTensorMaxPoolGarbler::evaluate_setup() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoTensorMaxPoolGarbler::evaluate_setup start", gate_id_));
    }
  }

  // garble MaxPool circuit
  ENCRYPTO::block128_vector in_keys;
  ENCRYPTO::block128_vector out_keys;
  maxpool_rearrange_keys_in(in_keys, input_->get_keys(), bit_size_, maxpool_op_);
  yao_provider_.create_garbled_circuit(gate_id_, maxpool_op_.compute_output_size(), maxpool_algo_,
                                       in_keys, {}, garbled_tables_, out_keys, true);
  yao_provider_.send_blocks_message(gate_id_, std::move(garbled_tables_));
  maxpool_rearrange_keys_out(output_->get_keys(), out_keys, bit_size_, maxpool_op_);

  output_->set_setup_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoTensorMaxPoolGarbler::evaluate_setup end", gate_id_));
    }
  }
}

YaoTensorMaxPoolEvaluator::YaoTensorMaxPoolEvaluator(std::size_t gate_id, YaoProvider& yao_provider,
                                                     tensor::MaxPoolOp maxpool_op,
                                                     const YaoTensorCP input)
    : NewGate(gate_id),
      yao_provider_(yao_provider),
      maxpool_op_(maxpool_op),
      bit_size_(input->get_bit_size()),
      data_size_(input->get_dimensions().get_data_size()),
      input_(input),
      output_(std::make_shared<YaoTensor>(maxpool_op_.get_output_tensor_dims(), bit_size_)),
      maxpool_algo_(yao_provider_.get_circuit_loader().load_maxpool_circuit(
          bit_size_, maxpool_op_.compute_kernel_size())) {
  const std::size_t num_and_gates =
      (2 * bit_size_) * (maxpool_op_.compute_kernel_size() - 1) * maxpool_op_.compute_output_size();
  garbled_tables_future_ = yao_provider_.register_for_blocks_message(gate_id, 2 * num_and_gates);
  output_->get_keys().resize(bit_size_);
}

void YaoTensorMaxPoolEvaluator::evaluate_online() {
  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoTensorMaxPoolEvaluator::evaluate_online start", gate_id_));
    }
  }

  // evaluate MaxPool circuit
  const auto garbled_tables = garbled_tables_future_.get();
  ENCRYPTO::block128_vector in_keys;
  ENCRYPTO::block128_vector out_keys;
  maxpool_rearrange_keys_in(in_keys, input_->get_keys(), bit_size_, maxpool_op_);
  yao_provider_.evaluate_garbled_circuit(gate_id_, maxpool_op_.compute_output_size(), maxpool_algo_,
                                         in_keys, {}, garbled_tables, out_keys, true);
  maxpool_rearrange_keys_out(output_->get_keys(), out_keys, bit_size_, maxpool_op_);
  output_->set_online_ready();

  if constexpr (MOTION_VERBOSE_DEBUG) {
    auto logger = yao_provider_.get_logger();
    if (logger) {
      logger->LogTrace(
          fmt::format("Gate {}: YaoTensorMaxPoolEvaluator::evaluate_online end", gate_id_));
    }
  }
}

}  // namespace MOTION::proto::yao
