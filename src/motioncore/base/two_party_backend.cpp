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

#include "base/gate_register.h"
#include "communication/communication_layer.h"
#include "crypto/base_ots/base_ot_provider.h"
#include "crypto/motion_base_provider.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "executor/new_gate_executor.h"
#include "protocols/yao/yao_provider.h"
#include "statistics/run_time_stats.h"
#include "utility/typedefs.h"

namespace MOTION {

TwoPartyBackend::TwoPartyBackend(Communication::CommunicationLayer& comm_layer,
                                 std::shared_ptr<Logger> logger)
    : comm_layer_(comm_layer),
      my_id_(comm_layer_.get_my_id()),
      logger_(logger),
      gate_register_(std::make_unique<GateRegister>()),
      gate_executor_(std::make_unique<NewGateExecutor>(*gate_register_, [] {}, logger_)),
      motion_base_provider_(std::make_unique<Crypto::MotionBaseProvider>(comm_layer_, logger_)),
      base_ot_provider_(std::make_unique<BaseOTProvider>(comm_layer_, logger_)),
      ot_manager_(std::make_unique<ENCRYPTO::ObliviousTransfer::OTProviderManager>(
          comm_layer_, *base_ot_provider_, *motion_base_provider_, logger_)),
      yao_provider_(std::make_unique<proto::yao::YaoProvider>(
          comm_layer_, *gate_register_, *motion_base_provider_,
          ot_manager_->get_provider(1 - my_id_), logger_)) {
  gate_factories_.emplace(MPCProtocol::Yao, *yao_provider_);
  comm_layer_.start();
}

TwoPartyBackend::~TwoPartyBackend() = default;

void TwoPartyBackend::run_preprocessing() {
  motion_base_provider_->setup();
  base_ot_provider_->ComputeBaseOTs();
  ot_manager_->run_setup();
  yao_provider_->setup();
}

void TwoPartyBackend::run() {
  Statistics::RunTimeStats stats;
  gate_executor_->evaluate_setup_online(stats);
  std::cout << stats.print_human_readable();
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

}  // namespace MOTION
