// MIT License
//
// Copyright (c) 2019 Oleksandr Tkachenko
// Cryptography and Privacy Engineering Group (ENCRYPTO)
// TU Darmstadt, Germany
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

#include "mt_provider.h"
#include <algorithm>

#include "crypto/arithmetic_provider.h"
#include "crypto/oblivious_transfer/ot_flavors.h"
#include "statistics/run_time_stats.h"
#include "utility/constants.h"
#include "utility/logger.h"

namespace MOTION {

bool MTProvider::NeedMTs() const noexcept {
  return (GetNumMTs<bool>() + GetNumMTs<std::uint8_t>() + GetNumMTs<std::uint16_t>() +
          GetNumMTs<std::uint32_t>() + GetNumMTs<std::uint64_t>()) > 0;
}

std::size_t MTProvider::RequestBinaryMTs(const std::size_t num_mts) noexcept {
  const auto offset = num_bit_mts_;
  num_bit_mts_ += num_mts;
  return offset;
}

// get bits [i, i+n] as vector
BinaryMTVector MTProvider::GetBinary(const std::size_t offset, const std::size_t n) const {
  assert(bit_mts_.a.GetSize() == bit_mts_.b.GetSize());
  assert(bit_mts_.b.GetSize() == bit_mts_.c.GetSize());
  WaitFinished();
  return BinaryMTVector{bit_mts_.a.Subset(offset, offset + n),
                        bit_mts_.b.Subset(offset, offset + n),
                        bit_mts_.c.Subset(offset, offset + n)};
}

const BinaryMTVector& MTProvider::GetBinaryAll() const noexcept {
  WaitFinished();
  return bit_mts_;
}

MTProvider::MTProvider(const std::size_t my_id, const std::size_t num_parties)
    : my_id_(my_id), num_parties_(num_parties) {
  finished_condition_ =
      std::make_shared<ENCRYPTO::FiberCondition>([this]() { return finished_.load(); });
}

// ---------- MTProviderFromOTs ----------

namespace {

void generate_random_triples_bool(BinaryMTVector& bit_mts, std::size_t num_bit_mts) {
  if (num_bit_mts > 0u) {
    bit_mts.a = ENCRYPTO::BitVector<>::Random(num_bit_mts);
    bit_mts.b = ENCRYPTO::BitVector<>::Random(num_bit_mts);
    bit_mts.c = bit_mts.a & bit_mts.b;
  }
}

template <typename T>
void generate_random_triples(IntegerMTVector<T>& mts, std::size_t num_mts) {
  if (num_mts > 0u) {
    mts.a = Helpers::RandomVector<T>(num_mts);
    mts.b = Helpers::RandomVector<T>(num_mts);
    mts.c.resize(num_mts);
    std::transform(mts.a.cbegin(), mts.a.cend(), mts.b.cbegin(), mts.c.begin(),
                   [](const auto& a_i, const auto& b_i) { return a_i * b_i; });
  }
}

void register_helper_bool(ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider,
                          std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitSender>& ots_snd,
                          std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitReceiver>& ots_rcv,
                          const BinaryMTVector& bit_mts, std::size_t num_bit_mts) {
  ots_snd = ot_provider.RegisterSendXCOTBit(num_bit_mts);
  ots_rcv = ot_provider.RegisterReceiveXCOTBit(num_bit_mts);

  ots_snd->SetCorrelations(bit_mts.a);
  ots_rcv->SetChoices(bit_mts.b);
}

void register_helper_bool_2pc(ENCRYPTO::ObliviousTransfer::OTProvider& ot_provider,
                              std::unique_ptr<ENCRYPTO::ObliviousTransfer::ROTSender>& ots_snd,
                              std::unique_ptr<ENCRYPTO::ObliviousTransfer::ROTReceiver>& ots_rcv,
                              std::size_t num_bit_mts) {
  ots_snd = ot_provider.RegisterSendROT(num_bit_mts);
  ots_rcv = ot_provider.RegisterReceiveROT(num_bit_mts);
}

template <typename T>
void register_multiplications_helper(
    ArithmeticProvider& arithmetic_provider,
    std::list<std::unique_ptr<IntegerMultiplicationSender<T>>>& mult_senders,
    std::list<std::unique_ptr<IntegerMultiplicationReceiver<T>>>& mult_receivers,
    std::size_t max_batch_size, std::size_t num_mts) {
  for (std::size_t mt_id = 0; mt_id < num_mts;) {
    auto batch_size = std::min(max_batch_size, num_mts - mt_id);

    auto mult_sender = arithmetic_provider.register_integer_multiplication_send<T>(batch_size);
    auto mult_receiver = arithmetic_provider.register_integer_multiplication_receive<T>(batch_size);
    mult_senders.push_back(std::move(mult_sender));
    mult_receivers.push_back(std::move(mult_receiver));

    mt_id += batch_size;
  }
}

template <typename T>
void start_multiplications_helper(
    std::list<std::unique_ptr<IntegerMultiplicationSender<T>>>& mult_senders,
    std::list<std::unique_ptr<IntegerMultiplicationReceiver<T>>>& mult_receivers,
    std::size_t max_batch_size, std::size_t num_mts, const IntegerMTVector<T>& mts) {
  auto it_sender = std::begin(mult_senders);
  auto it_receiver = std::begin(mult_receivers);
  for (std::size_t mt_id = 0; mt_id < num_mts; mt_id += max_batch_size) {
    (*it_sender)->set_inputs(mts.a.data() + mt_id);
    (*it_receiver)->set_inputs(mts.b.data() + mt_id);

    ++it_sender;
    ++it_receiver;
  }
}

template <typename T>
void compute_multiplications_helper(
    std::list<std::unique_ptr<IntegerMultiplicationSender<T>>>& mult_senders,
    std::list<std::unique_ptr<IntegerMultiplicationReceiver<T>>>& mult_receivers) {
  for (auto& sender : mult_senders) {
    sender->compute_outputs();
  }
  for (auto& receiver : mult_receivers) {
    receiver->compute_outputs();
  }
}

void finish_mts_helper_bool(ENCRYPTO::ObliviousTransfer::XCOTBitSender& ot_sender,
                            ENCRYPTO::ObliviousTransfer::XCOTBitReceiver& ot_receiver,
                            BinaryMTVector& bit_mts) {
  const auto& out_s = ot_sender.GetOutputs();
  const auto& out_r = ot_receiver.GetOutputs();
  bit_mts.c ^= out_s;
  bit_mts.c ^= out_r;
}

template <typename T>
void finish_mts_helper(std::list<std::unique_ptr<IntegerMultiplicationSender<T>>>& mult_senders,
                       std::list<std::unique_ptr<IntegerMultiplicationReceiver<T>>>& mult_receivers,
                       std::size_t max_batch_size, std::size_t num_mts, IntegerMTVector<T>& mts) {
  auto it_sender = std::begin(mult_senders);
  auto it_receiver = std::begin(mult_receivers);
  for (std::size_t mt_id = 0; mt_id < num_mts; mt_id += max_batch_size) {
    const auto share_s = (*it_sender)->get_outputs();
    const auto share_r = (*it_receiver)->get_outputs();
    for (std::size_t i = 0; i < share_s.size(); ++i) {
      mts.c.at(mt_id + i) += share_s[i] + share_r[i];
    }

    ++it_sender;
    ++it_receiver;
  }
}

void compute_mts_helper_bool_2pc(ENCRYPTO::ObliviousTransfer::ROTSender& ot_sender,
                                 ENCRYPTO::ObliviousTransfer::ROTReceiver& ot_receiver,
                                 BinaryMTVector& bit_mts) {
  ot_sender.ComputeOutputs();
  ot_receiver.ComputeOutputs();
  const auto [m_0, m_1] = ot_sender.GetOutputs();
  const auto m_r = ot_receiver.GetOutputs();
  bit_mts.a = ot_receiver.GetChoices();
  bit_mts.b = std::move(m_1);
  bit_mts.b ^= m_0;
  bit_mts.c = bit_mts.a & bit_mts.b;
  bit_mts.c ^= m_r;
  bit_mts.c ^= m_0;
}

}  // namespace

struct MTProviderFromOTs::MTProviderFromOTsImpl {
  MTProviderFromOTsImpl(std::size_t num_parties);

  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ROTSender> rots_sender_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::ROTReceiver> rots_receiver_;
  std::vector<std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitSender>> bit_xcots_senders_;
  std::vector<std::unique_ptr<ENCRYPTO::ObliviousTransfer::XCOTBitReceiver>> bit_xcots_receivers_;
  std::vector<std::list<std::unique_ptr<IntegerMultiplicationSender<std::uint8_t>>>>
      mult_senders_8_;
  std::vector<std::list<std::unique_ptr<IntegerMultiplicationReceiver<std::uint8_t>>>>
      mult_receivers_8_;
  std::vector<std::list<std::unique_ptr<IntegerMultiplicationSender<std::uint16_t>>>>
      mult_senders_16_;
  std::vector<std::list<std::unique_ptr<IntegerMultiplicationReceiver<std::uint16_t>>>>
      mult_receivers_16_;
  std::vector<std::list<std::unique_ptr<IntegerMultiplicationSender<std::uint32_t>>>>
      mult_senders_32_;
  std::vector<std::list<std::unique_ptr<IntegerMultiplicationReceiver<std::uint32_t>>>>
      mult_receivers_32_;
  std::vector<std::list<std::unique_ptr<IntegerMultiplicationSender<std::uint64_t>>>>
      mult_senders_64_;
  std::vector<std::list<std::unique_ptr<IntegerMultiplicationReceiver<std::uint64_t>>>>
      mult_receivers_64_;
};

MTProviderFromOTs::MTProviderFromOTsImpl::MTProviderFromOTsImpl(std::size_t num_parties)
    : bit_xcots_senders_(num_parties),
      bit_xcots_receivers_(num_parties),
      mult_senders_8_(num_parties),
      mult_receivers_8_(num_parties),
      mult_senders_16_(num_parties),
      mult_receivers_16_(num_parties),
      mult_senders_32_(num_parties),
      mult_receivers_32_(num_parties),
      mult_senders_64_(num_parties),
      mult_receivers_64_(num_parties) {}

MTProviderFromOTs::MTProviderFromOTs(std::size_t my_id, std::size_t num_parties,
                                     ArithmeticProviderManager& arithmetic_manager,
                                     ENCRYPTO::ObliviousTransfer::OTProviderManager& ot_manager,
                                     Statistics::RunTimeStats& run_time_stats,
                                     std::shared_ptr<Logger> logger)
    : MTProviderFromOTs(my_id, num_parties, false, arithmetic_manager, ot_manager, run_time_stats,
                        logger) {}

MTProviderFromOTs::MTProviderFromOTs(std::size_t my_id, std::size_t num_parties, bool use_2pc,
                                     ArithmeticProviderManager& arithmetic_manager,
                                     ENCRYPTO::ObliviousTransfer::OTProviderManager& ot_manager,
                                     Statistics::RunTimeStats& run_time_stats,
                                     std::shared_ptr<Logger> logger)
    : MTProvider(my_id, num_parties),
      impl_(std::make_unique<MTProviderFromOTsImpl>(num_parties)),
      use_2pc_(use_2pc && num_parties == 2),
      arithmetic_manager_(arithmetic_manager),
      ot_manager_(ot_manager),
      run_time_stats_(run_time_stats),
      logger_(logger) {}

MTProviderFromOTs::~MTProviderFromOTs() = default;

void MTProviderFromOTs::PreSetup() {
  if constexpr (MOTION_DEBUG) {
    if (logger_) {
      logger_->LogDebug("Start computing presetup for MTs");
    }
  }
  run_time_stats_.record_start<Statistics::RunTimeStats::StatID::mt_presetup>();

  if (!use_2pc_) {
    generate_random_triples_bool(bit_mts_, num_bit_mts_);
  }
  generate_random_triples<std::uint8_t>(mts8_, num_mts_8_);
  generate_random_triples<std::uint16_t>(mts16_, num_mts_16_);
  generate_random_triples<std::uint32_t>(mts32_, num_mts_32_);
  generate_random_triples<std::uint64_t>(mts64_, num_mts_64_);

  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }

    if (num_bit_mts_ > 0) {
      if (use_2pc_) {
        assert(party_id == 1 - my_id_);
        register_helper_bool_2pc(ot_manager_.get_provider(party_id), impl_->rots_sender_,
                                 impl_->rots_receiver_, num_bit_mts_);
      } else {
        register_helper_bool(ot_manager_.get_provider(party_id),
                             impl_->bit_xcots_senders_.at(party_id),
                             impl_->bit_xcots_receivers_.at(party_id), bit_mts_, num_bit_mts_);
      }
    }
    register_multiplications_helper<std::uint8_t>(
        arithmetic_manager_.get_provider(party_id), impl_->mult_senders_8_.at(party_id),
        impl_->mult_receivers_8_.at(party_id), max_batch_size_, num_mts_8_);
    register_multiplications_helper<std::uint16_t>(
        arithmetic_manager_.get_provider(party_id), impl_->mult_senders_16_.at(party_id),
        impl_->mult_receivers_16_.at(party_id), max_batch_size_, num_mts_16_);
    register_multiplications_helper<std::uint32_t>(
        arithmetic_manager_.get_provider(party_id), impl_->mult_senders_32_.at(party_id),
        impl_->mult_receivers_32_.at(party_id), max_batch_size_, num_mts_32_);
    register_multiplications_helper<std::uint64_t>(
        arithmetic_manager_.get_provider(party_id), impl_->mult_senders_64_.at(party_id),
        impl_->mult_receivers_64_.at(party_id), max_batch_size_, num_mts_64_);
  }

  run_time_stats_.record_end<Statistics::RunTimeStats::StatID::mt_presetup>();
  if constexpr (MOTION_DEBUG) {
    if (logger_) {
      logger_->LogDebug("Finished computing presetup for MTs");
    }
  }
}

void MTProviderFromOTs::Setup() {
  if constexpr (MOTION_DEBUG) {
    if (logger_) {
      logger_->LogDebug("Start computing setup for MTs");
    }
  }
  run_time_stats_.record_start<Statistics::RunTimeStats::StatID::mt_setup>();

  std::vector<std::future<void>> futures;
  futures.reserve(num_parties_ - 1);

  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    futures.emplace_back(std::async(std::launch::async, [this, party_id] {
      // prepare and send messages
      if (num_bit_mts_ > 0 && !use_2pc_) {
        assert(impl_->bit_xcots_senders_.at(party_id) != nullptr);
        assert(impl_->bit_xcots_receivers_.at(party_id) != nullptr);
        impl_->bit_xcots_senders_.at(party_id)->SendMessages();
        impl_->bit_xcots_receivers_.at(party_id)->SendCorrections();
      }
      start_multiplications_helper<std::uint8_t>(impl_->mult_senders_8_.at(party_id),
                                                 impl_->mult_receivers_8_.at(party_id),
                                                 max_batch_size_, num_mts_8_, mts8_);
      start_multiplications_helper<std::uint16_t>(impl_->mult_senders_16_.at(party_id),
                                                  impl_->mult_receivers_16_.at(party_id),
                                                  max_batch_size_, num_mts_16_, mts16_);
      start_multiplications_helper<std::uint32_t>(impl_->mult_senders_32_.at(party_id),
                                                  impl_->mult_receivers_32_.at(party_id),
                                                  max_batch_size_, num_mts_32_, mts32_);
      start_multiplications_helper<std::uint64_t>(impl_->mult_senders_64_.at(party_id),
                                                  impl_->mult_receivers_64_.at(party_id),
                                                  max_batch_size_, num_mts_64_, mts64_);

      // finish the OTs and multiplications
      if (num_bit_mts_ > 0 && !use_2pc_) {
        assert(impl_->bit_xcots_senders_.at(party_id) != nullptr);
        assert(impl_->bit_xcots_receivers_.at(party_id) != nullptr);
        impl_->bit_xcots_senders_.at(party_id)->ComputeOutputs();
        impl_->bit_xcots_receivers_.at(party_id)->ComputeOutputs();
      }
      compute_multiplications_helper<std::uint8_t>(impl_->mult_senders_8_.at(party_id),
                                                   impl_->mult_receivers_8_.at(party_id));
      compute_multiplications_helper<std::uint16_t>(impl_->mult_senders_16_.at(party_id),
                                                    impl_->mult_receivers_16_.at(party_id));
      compute_multiplications_helper<std::uint32_t>(impl_->mult_senders_32_.at(party_id),
                                                    impl_->mult_receivers_32_.at(party_id));
      compute_multiplications_helper<std::uint64_t>(impl_->mult_senders_64_.at(party_id),
                                                    impl_->mult_receivers_64_.at(party_id));
    }));
  }

  if (num_bit_mts_ > 0 && use_2pc_) {
    assert(num_parties_ == 2);
    compute_mts_helper_bool_2pc(*impl_->rots_sender_, *impl_->rots_receiver_, bit_mts_);
  }

  std::for_each(std::begin(futures), std::end(futures), [](auto& f) { f.get(); });

  // finish computation of MTs (would need synchronization)
  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    if (num_bit_mts_ > 0 && !use_2pc_) {
      assert(impl_->bit_xcots_senders_.at(party_id) != nullptr);
      assert(impl_->bit_xcots_receivers_.at(party_id) != nullptr);
      finish_mts_helper_bool(*impl_->bit_xcots_senders_.at(party_id),
                             *impl_->bit_xcots_receivers_.at(party_id), bit_mts_);
    }
    finish_mts_helper<std::uint8_t>(impl_->mult_senders_8_.at(party_id),
                                    impl_->mult_receivers_8_.at(party_id), max_batch_size_,
                                    num_mts_8_, mts8_);
    finish_mts_helper<std::uint16_t>(impl_->mult_senders_16_.at(party_id),
                                     impl_->mult_receivers_16_.at(party_id), max_batch_size_,
                                     num_mts_16_, mts16_);
    finish_mts_helper<std::uint32_t>(impl_->mult_senders_32_.at(party_id),
                                     impl_->mult_receivers_32_.at(party_id), max_batch_size_,
                                     num_mts_32_, mts32_);
    finish_mts_helper<std::uint64_t>(impl_->mult_senders_64_.at(party_id),
                                     impl_->mult_receivers_64_.at(party_id), max_batch_size_,
                                     num_mts_64_, mts64_);
  }

  // signal MTs are ready
  {
    std::scoped_lock lock(finished_condition_->GetMutex());
    finished_ = true;
  }
  finished_condition_->NotifyAll();

  run_time_stats_.record_end<Statistics::RunTimeStats::StatID::mt_setup>();
  if constexpr (MOTION_DEBUG) {
    if (logger_) {
      logger_->LogDebug("Finished computing setup for MTs");
    }
  }
}

}  // namespace MOTION
