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
#include <unordered_map>
#include <vector>

#include "tensor/network_builder.h"

namespace ENCRYPTO::ObliviousTransfer {
class OTProviderManager;
}

namespace MOTION {

class ArithmeticProviderManager;
class BaseOTProvider;
class CircuitLoader;
class GateRegister;
class LinAlgTripleProvider;
class Logger;
class MTProvider;
class TensorOpExecutor;
class SBProvider;
class SPProvider;
enum class MPCProtocol : unsigned int;

namespace Communication {
class CommunicationLayer;
}

namespace Crypto {
class MotionBaseProvider;
}

namespace proto {
namespace beavy {
class BEAVYProvider;
}
namespace gmw {
class GMWProvider;
}
namespace yao {
class YaoProvider;
}
}  // namespace proto

namespace Statistics {
struct RunTimeStats;
}

namespace tensor {
class TensorOpFactory;
}

class TwoPartyTensorBackend : public tensor::NetworkBuilder {
 public:
  TwoPartyTensorBackend(Communication::CommunicationLayer&, std::size_t num_threads,
                        std::shared_ptr<Logger>, bool fake_triples = false);
  virtual ~TwoPartyTensorBackend();

  virtual void run_preprocessing();
  void run();

  tensor::TensorOpFactory& get_tensor_op_factory(MPCProtocol) override;
  tensor::TensorCP convert(MPCProtocol, const tensor::TensorCP) override;

  const Statistics::RunTimeStats& get_run_time_stats() const noexcept;

 protected:
  Communication::CommunicationLayer& comm_layer_;
  std::size_t my_id_;
  std::shared_ptr<Logger> logger_;
  std::unique_ptr<GateRegister> gate_register_;
  std::unique_ptr<TensorOpExecutor> gate_executor_;
  std::unique_ptr<CircuitLoader> circuit_loader_;
  std::unordered_map<MPCProtocol, std::reference_wrapper<tensor::TensorOpFactory>>
      tensor_op_factories_;
  std::vector<Statistics::RunTimeStats> run_time_stats_;

  std::unique_ptr<Crypto::MotionBaseProvider> motion_base_provider_;
  std::unique_ptr<BaseOTProvider> base_ot_provider_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::OTProviderManager> ot_manager_;
  std::unique_ptr<ArithmeticProviderManager> arithmetic_manager_;
  std::shared_ptr<LinAlgTripleProvider> linalg_triple_provider_;
  std::unique_ptr<MTProvider> mt_provider_;
  std::unique_ptr<SPProvider> sp_provider_;
  std::unique_ptr<SBProvider> sb_provider_;

  std::unique_ptr<proto::beavy::BEAVYProvider> beavy_provider_;
  std::unique_ptr<proto::gmw::GMWProvider> gmw_provider_;
  std::unique_ptr<proto::yao::YaoProvider> yao_provider_;
};

}  // namespace MOTION
