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

#include "ot_backend.h"

#include <memory>
#include <stdexcept>

#include <fmt/format.h>

#include "communication/communication_layer.h"
#include "crypto/arithmetic_provider.h"
#include "crypto/base_ots/base_ot_provider.h"
#include "crypto/motion_base_provider.h"
#include "crypto/oblivious_transfer/ot_provider.h"
#include "statistics/run_time_stats.h"
#include "utility/logger.h"
#include "utility/typedefs.h"

namespace MOTION {

OTBackend::OTBackend(Communication::CommunicationLayer& comm_layer, std::shared_ptr<Logger> logger)
    : comm_layer_(comm_layer),
      my_id_(comm_layer_.get_my_id()),
      logger_(logger),
      run_time_stats_(1),
      motion_base_provider_(std::make_unique<Crypto::MotionBaseProvider>(comm_layer_, logger_)),
      base_ot_provider_(
          std::make_unique<BaseOTProvider>(comm_layer_, &run_time_stats_.back(), logger_)),
      ot_manager_(std::make_unique<ENCRYPTO::ObliviousTransfer::OTProviderManager>(
          comm_layer_, *base_ot_provider_, *motion_base_provider_, &run_time_stats_.back(),
          logger_)),
      arithmetic_manager_(
          std::make_unique<ArithmeticProviderManager>(comm_layer_, *ot_manager_, logger_)) {
  comm_layer_.start();
}

OTBackend::~OTBackend() = default;

void OTBackend::run_setup() {
  run_time_stats_.back().record_start<Statistics::RunTimeStats::StatID::preprocessing>();

  if (!ran_base_setup_) {
    motion_base_provider_->setup();
    base_ot_provider_->ComputeBaseOTs();
    ran_base_setup_ = true;
  }
  ot_manager_->run_setup();

  run_time_stats_.back().record_end<Statistics::RunTimeStats::StatID::preprocessing>();
}

void OTBackend::clear() {
  ot_manager_->clear();
  run_time_stats_.push_back({});
}

void OTBackend::sync() {
  comm_layer_.sync();
}

ArithmeticProvider& OTBackend::get_arithmetic_provider() noexcept {
  return arithmetic_manager_->get_provider(1 - my_id_);
}

}  // namespace MOTION
