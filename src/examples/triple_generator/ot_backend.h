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
#include <vector>

namespace ENCRYPTO::ObliviousTransfer {
class OTProviderManager;
}

namespace MOTION {

class ArithmeticProvider;
class ArithmeticProviderManager;
class BaseOTProvider;
class Logger;

namespace Communication {
class CommunicationLayer;
}

namespace Crypto {
class MotionBaseProvider;
}

namespace Statistics {
struct RunTimeStats;
}

class OTBackend {
 public:
  OTBackend(Communication::CommunicationLayer&, std::shared_ptr<Logger>);
  ~OTBackend();

  void run_setup();
  void clear();
  void sync();

  auto get_my_id() const noexcept { return my_id_; }
  const auto& get_run_time_stats() const noexcept { return run_time_stats_; }
  ArithmeticProvider& get_arithmetic_provider() noexcept;

 private:
  Communication::CommunicationLayer& comm_layer_;
  std::size_t my_id_;
  std::shared_ptr<Logger> logger_;
  std::vector<Statistics::RunTimeStats> run_time_stats_;
  bool ran_base_setup_ = false;

  std::unique_ptr<Crypto::MotionBaseProvider> motion_base_provider_;
  std::unique_ptr<BaseOTProvider> base_ot_provider_;
  std::unique_ptr<ENCRYPTO::ObliviousTransfer::OTProviderManager> ot_manager_;
  std::unique_ptr<ArithmeticProviderManager> arithmetic_manager_;
};

}  // namespace MOTION
