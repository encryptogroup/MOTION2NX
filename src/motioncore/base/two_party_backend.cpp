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

#include "two_party_backend.h"

#include <memory>
#include <stdexcept>

#include <fmt/format.h>

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
#include "executor/new_gate_executor.h"
#include "protocols/beavy/beavy_provider.h"
#include "protocols/gmw/gmw_provider.h"
#include "protocols/yao/yao_provider.h"
#include "statistics/run_time_stats.h"
#include "utility/logger.h"
#include "utility/typedefs.h"

namespace MOTION {

TwoPartyBackend::TwoPartyBackend(Communication::CommunicationLayer& comm_layer,
                                 std::size_t num_threads, std::shared_ptr<Logger> logger)
    : comm_layer_(comm_layer),
      my_id_(comm_layer_.get_my_id()),
      logger_(logger),
      gate_register_(std::make_unique<GateRegister>()),
      gate_executor_(std::make_unique<NewGateExecutor>(
          *gate_register_, [this] { run_preprocessing(); }, num_threads, logger_)),
      circuit_loader_(std::make_unique<CircuitLoader>()),
      run_time_stats_(1),
      motion_base_provider_(std::make_unique<Crypto::MotionBaseProvider>(comm_layer_, logger_)),
      base_ot_provider_(
          std::make_unique<BaseOTProvider>(comm_layer_, &run_time_stats_.back(), logger_)),
      ot_manager_(std::make_unique<ENCRYPTO::ObliviousTransfer::OTProviderManager>(
          comm_layer_, *base_ot_provider_, *motion_base_provider_, &run_time_stats_.back(),
          logger_)),
      arithmetic_manager_(
          std::make_unique<ArithmeticProviderManager>(comm_layer_, *ot_manager_, logger_)),
      mt_provider_(std::make_unique<MTProviderFromOTs>(my_id_, comm_layer_.get_num_parties(), true,
                                                       *arithmetic_manager_, *ot_manager_,
                                                       run_time_stats_.back(), logger_)),
      sp_provider_(std::make_unique<SPProviderFromOTs>(ot_manager_->get_providers(), my_id_,
                                                       run_time_stats_.back(), logger_)),
      sb_provider_(std::make_unique<TwoPartySBProvider>(
          comm_layer_, ot_manager_->get_provider(1 - my_id_), run_time_stats_.back(), logger_)),
      beavy_provider_(std::make_unique<proto::beavy::BEAVYProvider>(
          comm_layer_, *gate_register_, *motion_base_provider_, *ot_manager_, *arithmetic_manager_,
          logger_)),
      gmw_provider_(std::make_unique<proto::gmw::GMWProvider>(
          comm_layer_, *gate_register_, *motion_base_provider_, *ot_manager_, *mt_provider_,
          *sp_provider_, *sb_provider_, logger_)),
      yao_provider_(std::make_unique<proto::yao::YaoProvider>(
          comm_layer_, *gate_register_, *circuit_loader_, *motion_base_provider_,
          ot_manager_->get_provider(1 - my_id_), logger_)) {
  gate_factories_.emplace(MPCProtocol::ArithmeticBEAVY, *beavy_provider_);
  gate_factories_.emplace(MPCProtocol::BooleanBEAVY, *beavy_provider_);
  gate_factories_.emplace(MPCProtocol::ArithmeticGMW, *gmw_provider_);
  gate_factories_.emplace(MPCProtocol::BooleanGMW, *gmw_provider_);
  gate_factories_.emplace(MPCProtocol::Yao, *yao_provider_);
  comm_layer_.start();
}

TwoPartyBackend::~TwoPartyBackend() = default;

void TwoPartyBackend::run_preprocessing() {
  run_time_stats_.back().record_start<Statistics::RunTimeStats::StatID::preprocessing>();

  motion_base_provider_->setup();
  base_ot_provider_->ComputeBaseOTs();
  mt_provider_->PreSetup();
  sp_provider_->PreSetup();
  sb_provider_->PreSetup();
  ot_manager_->run_setup();
  mt_provider_->Setup();
  sp_provider_->Setup();
  sb_provider_->Setup();
  beavy_provider_->setup();
  gmw_provider_->setup();
  yao_provider_->setup();

  run_time_stats_.back().record_end<Statistics::RunTimeStats::StatID::preprocessing>();
}

void TwoPartyBackend::run() {
  gate_executor_->evaluate_setup_online(run_time_stats_.back());
}

std::optional<MPCProtocol> TwoPartyBackend::convert_via(MPCProtocol src_proto,
                                                        MPCProtocol dst_proto) {
  if (src_proto == MPCProtocol::ArithmeticGMW && dst_proto == MPCProtocol::BooleanGMW) {
    return MPCProtocol::Yao;
  } else if (src_proto == MPCProtocol::ArithmeticGMW && dst_proto == MPCProtocol::BooleanBEAVY) {
    return MPCProtocol::Yao;
  } else if (src_proto == MPCProtocol::BooleanGMW && dst_proto == MPCProtocol::ArithmeticBEAVY) {
    return MPCProtocol::BooleanBEAVY;
  } else if (src_proto == MPCProtocol::ArithmeticBEAVY && dst_proto == MPCProtocol::BooleanGMW) {
    return MPCProtocol::Yao;
  } else if (src_proto == MPCProtocol::ArithmeticBEAVY && dst_proto == MPCProtocol::BooleanBEAVY) {
    return MPCProtocol::Yao;
  } else if (src_proto == MPCProtocol::BooleanBEAVY && dst_proto == MPCProtocol::ArithmeticGMW) {
    return MPCProtocol::BooleanGMW;
  }
  return std::nullopt;
}

GateFactory& TwoPartyBackend::get_gate_factory(MPCProtocol proto) {
  try {
    return gate_factories_.at(proto);
  } catch (std::out_of_range& e) {
    throw std::logic_error(
        fmt::format("TwoPartyBackend::get_gate_factory: no GateFactory for protocol {} available",
                    ToString(proto)));
  }
}

const Statistics::RunTimeStats& TwoPartyBackend::get_run_time_stats() const noexcept {
  return run_time_stats_.back();
}

}  // namespace MOTION
