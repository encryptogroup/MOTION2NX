// MIT License
//
// Copyright (c) 2019 Lennart Braun
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

#include "gtest/gtest.h"

#include "test_constants.h"

#include "base/party.h"
#include "crypto/multiplication_triple/sb_impl.h"
#include "crypto/multiplication_triple/sb_provider.h"
#include "crypto/multiplication_triple/sp_provider.h"
#include "crypto/base_ots/base_ot_provider.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "statistics/run_time_stats.h"
#include "utility/type_traits.hpp"

namespace {

constexpr auto num_parties_list = {2u, 3u};

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
void template_test() {
  for (auto i = 0ull; i < TEST_ITERATIONS; ++i) {
    for (auto num_parties : {2u, 3u}) {
      std::size_t num_sbs = 100;

      try {
        auto motion_parties = MOTION::GetNLocalParties(num_parties, PORT_OFFSET);
        for (auto& p : motion_parties) {
          p->GetLogger()->SetEnabled(DETAILED_LOGGING_ENABLED);
          p->GetBackend()->GetSBProvider()->template RequestSBs<T>(num_sbs);
        }

        std::vector<std::future<void>> futs;
        futs.reserve(num_parties);
        for (auto& p : motion_parties) {
          futs.emplace_back(std::async(std::launch::async, [&p] {
            auto& backend = p->GetBackend();
            auto& sp_provider = backend->GetSPProvider();
            auto& sb_provider = backend->GetSBProvider();
            sb_provider->PreSetup();
            sp_provider->PreSetup();
            backend->OTExtensionSetup();
            sp_provider->Setup();
            sb_provider->Setup();
            p->Finish();
          }));
        }
        std::for_each(futs.begin(), futs.end(), [](auto& f) { f.get(); });

        const auto& sbp_0 = motion_parties.at(0)->GetBackend()->GetSBProvider();
        std::vector<T> a = sbp_0->template GetSBsAll<T>();
        EXPECT_EQ(a.size(), num_sbs);
        for (std::size_t j = 1; j < motion_parties.size(); ++j) {
          const auto& sbp_j = motion_parties.at(j)->GetBackend()->GetSBProvider();
          for (std::size_t k = 0; k < a.size(); ++k) {
            a.at(k) += sbp_j->template GetSBsAll<T>().at(k);
          }
        }
        for (std::size_t k = 0; k < a.size(); ++k) {
          EXPECT_TRUE(a.at(k) == T(0) || a.at(k) == T(1));
        }

        // with num_sbs bits generated there should be at least a 0 and a 1 whp
        EXPECT_TRUE(std::any_of(a.cbegin(), a.cend(), [](auto& b) { return b == 0; }));
        EXPECT_TRUE(std::any_of(a.cbegin(), a.cend(), [](auto& b) { return b == 1; }));

        futs.clear();

        for (auto& p : motion_parties) {
          futs.emplace_back(std::async(std::launch::async, [&p] { p->Finish(); }));
        }
        std::for_each(futs.begin(), futs.end(), [](auto& f) { f.get(); });

      } catch (std::exception& e) {
        std::cerr << e.what() << std::endl;
      }
    }
  }
}

TEST(SharedBits, Integer) {
  template_test<std::uint8_t>();
  template_test<std::uint16_t>();
  template_test<std::uint32_t>();
  template_test<std::uint64_t>();
}

TEST(SharedBitsImpl, Invert) {
  std::size_t k = 6;
  std::uint64_t a = 47;
  std::uint64_t x = MOTION::detail::invert(k, a);
  assert(x == 15);
}

TEST(SharedBitsImpl, Sqrt) {
  std::size_t k = 6;
  std::uint64_t a = 49;
  std::uint64_t x = MOTION::detail::sqrt(k, a);
  EXPECT_EQ(x, 7);
}

template <typename T>
std::pair<MOTION::SPVector<T>, std::vector<MOTION::SPVector<T>>> gen_sp_vectors(
    std::size_t num_parties, std::size_t size) {
  MOTION::SPVector<T> plain;
  plain.a = MOTION::Helpers::RandomVector<T>(size);
  std::transform(plain.a.cbegin(), plain.a.cend(), std::back_inserter(plain.c),
                 [](auto a) { return a * a; });
  std::vector<MOTION::SPVector<T>> shares(num_parties);
  std::copy(plain.a.cbegin(), plain.a.cend(), std::back_inserter(shares.at(0).a));
  std::copy(plain.c.cbegin(), plain.c.cend(), std::back_inserter(shares.at(0).c));
  for (std::size_t i = 1; i < num_parties; ++i) {
    shares.at(i).a = MOTION::Helpers::RandomVector<T>(size);
    shares.at(i).c = MOTION::Helpers::RandomVector<T>(size);
  }
  for (std::size_t j = 0; j < size; ++j) {
    for (std::size_t i = 1; i < num_parties; ++i) {
      shares.at(0).a.at(j) -= shares.at(i).a.at(j);
      shares.at(0).c.at(j) -= shares.at(i).c.at(j);
    }
  }
  return {plain, shares};
}

TEST(SharedBitsImpl, Helper) {
  std::size_t num_parties = 3;
  std::size_t num_sbs = 100;
  auto [plain_sps, shared_sps] = gen_sp_vectors<std::uint16_t>(num_parties, num_sbs);
  std::vector<std::vector<std::uint16_t>> shares_a;
  std::vector<std::vector<std::uint16_t>> shares_c;
  shares_a.reserve(shared_sps.size());
  shares_c.reserve(shared_sps.size());
  for (auto& sp_vector : shared_sps) {
    shares_a.push_back(std::move(sp_vector.a));
    shares_c.push_back(std::move(sp_vector.c));
  }
  auto reconstructed_a = MOTION::Helpers::AddVectors(shares_a);
  auto reconstructed_c = MOTION::Helpers::AddVectors(shares_c);
  EXPECT_EQ(plain_sps.a, reconstructed_a);
  EXPECT_EQ(plain_sps.c, reconstructed_c);
}

TEST(SharedBitsImpl, Phase1) {
  std::size_t num_parties = 3;
  std::size_t num_sbs = 100;
  auto reduce_mod = [](auto& v, auto k) {
    std::uint16_t mod_mask = (std::uint16_t(1) << k) - 1;
    std::transform(v.cbegin(), v.cend(), v.begin(), [mod_mask](auto a) { return a & mod_mask; });
  };

  auto [plain_sps, shared_sps] = gen_sp_vectors<std::uint16_t>(num_parties, num_sbs);
  std::vector<std::vector<std::uint16_t>> wb1s;
  std::vector<std::vector<std::uint16_t>> wb2s;
  for (std::size_t i = 0; i < num_parties; ++i) {
    auto [wb1, wb2] = MOTION::detail::compute_sbs_phase_1<std::uint8_t>(num_sbs, i, shared_sps.at(i));
    wb1s.emplace_back(std::move(wb1));
    wb2s.emplace_back(std::move(wb2));
  }

  // party 1 has the odd share of a
  EXPECT_TRUE(
      std::all_of(wb1s.at(0).cbegin(), wb1s.at(0).cend(), [](auto a) { return (a & 1) == 1; }));
  // all other parties have even shares of a
  for (std::size_t i = 1; i < num_parties; ++i) {
    EXPECT_TRUE(
        std::all_of(wb1s.at(i).cbegin(), wb1s.at(i).cend(), [](auto a) { return (a & 1) == 0; }));
  }

  auto a = MOTION::Helpers::AddVectors(wb1s);
  reduce_mod(a, 10);
  // a is odd
  EXPECT_TRUE(std::all_of(a.cbegin(), a.cend(), [](auto a) { return (a & 1) == 1; }));

  auto masked_a = MOTION::Helpers::AddVectors(wb2s);
  reduce_mod(masked_a, 10);
  auto unmasked_masked_a = MOTION::Helpers::AddVectors(masked_a, plain_sps.a);
  reduce_mod(unmasked_masked_a, 10);
  // check that a was masked correctly
  EXPECT_EQ(unmasked_masked_a, a);
}

TEST(SharedBitsImpl, Phase2) {
  std::size_t num_parties = 3;
  std::size_t num_sbs = 100;
  auto reduce_mod = [](auto& v, auto k) {
    std::uint16_t mod_mask = (std::uint16_t(1) << k) - 1;
    std::transform(v.cbegin(), v.cend(), v.begin(), [mod_mask](auto a) { return a & mod_mask; });
  };

  auto [plain_sps, shared_sps] = gen_sp_vectors<std::uint16_t>(num_parties, num_sbs);
  std::vector<std::vector<std::uint16_t>> wb1s;
  std::vector<std::vector<std::uint16_t>> wb2s;
  for (std::size_t i = 0; i < num_parties; ++i) {
    auto [wb1, wb2] = MOTION::detail::compute_sbs_phase_1<std::uint8_t>(num_sbs, i, shared_sps.at(i));
    wb1s.emplace_back(std::move(wb1));
    wb2s.emplace_back(std::move(wb2));
  }

  auto masked_a = MOTION::Helpers::AddVectors(wb2s);
  std::fill(wb2s.begin(), wb2s.end(), masked_a);

  for (std::size_t i = 0; i < num_parties; ++i) {
    MOTION::detail::compute_sbs_phase_2<std::uint8_t>(wb1s.at(i), wb2s.at(i), i, shared_sps.at(i));
  }

  auto a = MOTION::Helpers::AddVectors(wb1s);
  reduce_mod(a, 10);
  std::vector<std::uint16_t> a_squared_plain;
  std::transform(a.cbegin(), a.cend(), std::back_inserter(a_squared_plain),
                 [](auto a) { return a * a; });
  reduce_mod(a_squared_plain, 10);

  // check that wb2 contains shares of a^2
  auto a_squared = MOTION::Helpers::AddVectors(wb2s);
  reduce_mod(a_squared, 10);
  EXPECT_EQ(a_squared, a_squared_plain);
}

TEST(SharedBitsImpl, Phase3) {
  std::size_t num_parties = 3;
  std::size_t num_sbs = 100;
  auto reduce_mod = [](auto& v, auto k) {
    std::uint16_t mod_mask = (std::uint16_t(1) << k) - 1;
    std::transform(v.cbegin(), v.cend(), v.begin(), [mod_mask](auto a) { return a & mod_mask; });
  };

  auto [plain_sps, shared_sps] = gen_sp_vectors<std::uint16_t>(num_parties, num_sbs);
  std::vector<std::vector<std::uint16_t>> wb1s;
  std::vector<std::vector<std::uint16_t>> wb2s;
  for (std::size_t i = 0; i < num_parties; ++i) {
    auto [wb1, wb2] = MOTION::detail::compute_sbs_phase_1<std::uint8_t>(num_sbs, i, shared_sps.at(i));
    wb1s.emplace_back(std::move(wb1));
    wb2s.emplace_back(std::move(wb2));
  }

  auto masked_a = MOTION::Helpers::AddVectors(wb2s);
  std::fill(wb2s.begin(), wb2s.end(), masked_a);

  for (std::size_t i = 0; i < num_parties; ++i) {
    MOTION::detail::compute_sbs_phase_2<std::uint8_t>(wb1s.at(i), wb2s.at(i), i, shared_sps.at(i));
  }

  auto a_squared = MOTION::Helpers::AddVectors(wb2s);
  reduce_mod(a_squared, 10);
  std::fill(wb2s.begin(), wb2s.end(), a_squared);

  std::vector<std::vector<std::uint8_t>> sbs_8(num_parties);
  for (std::size_t i = 0; i < num_parties; ++i) {
    MOTION::detail::compute_sbs_phase_3<std::uint8_t>(wb1s.at(i), wb2s.at(i), sbs_8.at(i), i);
  }

  auto bits = MOTION::Helpers::AddVectors(sbs_8);
  reduce_mod(bits, 8);
  EXPECT_TRUE(std::all_of(bits.cbegin(), bits.cend(), [] (auto b) { return b == 0 || b == 1; }));
}

template <typename T>
class TwoPartySBProviderTest : public ::testing::Test {
  using is_enabled_t_ = ENCRYPTO::is_unsigned_int_t<T>;

 protected:
  void SetUp() override {
    comm_layers_ = MOTION::Communication::make_dummy_communication_layers(2);
    base_ot_providers_.resize(2);
    motion_base_providers_.resize(2);
    ot_provider_managers_.resize(2);
    sb_providers_.resize(2);
    for (std::size_t i = 0; i < 2; ++i) {
      base_ot_providers_[i] = std::make_unique<MOTION::BaseOTProvider>(*comm_layers_[i], nullptr);
      motion_base_providers_[i] =
          std::make_unique<MOTION::Crypto::MotionBaseProvider>(*comm_layers_[i], nullptr);
      ot_provider_managers_[i] = std::make_unique<ENCRYPTO::ObliviousTransfer::OTProviderManager>(
          *comm_layers_[i], *base_ot_providers_[i], *motion_base_providers_[i], nullptr);
      sb_providers_[i] = std::make_unique<MOTION::TwoPartySBProvider>(
          *comm_layers_[i], ot_provider_managers_[i]->get_provider(1 - i), stats_[i], nullptr);
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
        comm_layers_[i]->start();
        motion_base_providers_[i]->setup();
        base_ot_providers_[i]->ComputeBaseOTs();
        sb_providers_[i]->PreSetup();
        auto f = std::async(std::launch::async, [this, i] {
          ot_provider_managers_[i]->get_provider(1 - i).SendSetup();
        });
        ot_provider_managers_[i]->get_provider(1 - i).ReceiveSetup();
        f.get();
        sb_providers_[i]->Setup();
      }));
    }
    std::for_each(std::begin(futs), std::end(futs), [](auto& f) { f.get(); });
  }

  std::vector<std::unique_ptr<MOTION::Communication::CommunicationLayer>> comm_layers_;
  std::vector<std::unique_ptr<MOTION::BaseOTProvider>> base_ot_providers_;
  std::vector<std::unique_ptr<MOTION::Crypto::MotionBaseProvider>> motion_base_providers_;
  std::vector<std::unique_ptr<ENCRYPTO::ObliviousTransfer::OTProviderManager>>
      ot_provider_managers_;
  std::vector<std::unique_ptr<MOTION::SBProvider>> sb_providers_;
  std::array<MOTION::Statistics::RunTimeStats, 2> stats_;
};

using integer_types = ::testing::Types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;
TYPED_TEST_SUITE(TwoPartySBProviderTest, integer_types);

TYPED_TEST(TwoPartySBProviderTest, SBs) {
  std::size_t num_sbs_a = 47;
  std::size_t num_sbs_b = 42;
  auto& sb_provider_0 = *this->sb_providers_[0];
  auto& sb_provider_1 = *this->sb_providers_[1];
  auto offset_a_0 = sb_provider_0.template RequestSBs<TypeParam>(num_sbs_a);
  auto offset_a_1 = sb_provider_1.template RequestSBs<TypeParam>(num_sbs_a);
  ASSERT_EQ(offset_a_0, offset_a_1);
  auto offset_b_0 = sb_provider_0.template RequestSBs<TypeParam>(num_sbs_b);
  auto offset_b_1 = sb_provider_1.template RequestSBs<TypeParam>(num_sbs_b);
  ASSERT_EQ(offset_b_0, offset_b_1);

  this->run_setup();

  const auto& sbs_all_0 = sb_provider_0.template GetSBsAll<TypeParam>();
  const auto& sbs_all_1 = sb_provider_1.template GetSBsAll<TypeParam>();
  ASSERT_EQ(sbs_all_0.size(), num_sbs_a + num_sbs_b);
  ASSERT_EQ(sbs_all_1.size(), num_sbs_a + num_sbs_b);

  for (std::size_t sb_i = 0; sb_i < num_sbs_a + num_sbs_b; ++sb_i) {
    TypeParam bit = sbs_all_0.at(sb_i) + sbs_all_1.at(sb_i);
    ASSERT_TRUE(bit == TypeParam(0) || bit == TypeParam(1));
  }
}

}  // namespace
