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

#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include "base/two_party_tensor_backend.h"
#include "communication/communication_layer.h"
#include "communication/tcp_transport.h"
#include "onnx_adapter.h"
#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include "tensor/tensor_op_factory.h"
#include "utility/helpers.h"
#include "utility/logger.h"
#include "utility/typedefs.h"

namespace po = boost::program_options;

struct Options {
  std::size_t num_repetitions;
  MOTION::MPCProtocol protocol;
  std::size_t my_id;
  MOTION::Communication::tcp_parties_config tcp_config;
  std::string model_path;
  bool no_run = false;
  bool fake_triples = false;
};

std::optional<Options> parse_program_options(int argc, char* argv[]) {
  Options options;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help,h", po::bool_switch()->default_value(false),"produce help message")
    ("config-file", po::value<std::string>(), "config file containing options")
    ("my-id", po::value<std::size_t>()->required(), "my party id")
    ("party", po::value<std::vector<std::string>>()->multitoken(),
     "(party id, IP, port), e.g., --party 1,127.0.0.1,7777")
    ("protocol", po::value<std::string>()->required(), "2PC protocol (GMW or BEAVY)")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("no-run", po::bool_switch()->default_value(false),
     "just build the network, but not execute it")
    ("fake-triples", po::bool_switch()->default_value(false),
     "use random data instead of generating valid Beaver triples")
    ("model", po::value<std::string>()->required(), "path to a model file in ONNX format");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  bool help = vm["help"].as<bool>();
  if (help) {
    std::cerr << desc << "\n";
    return std::nullopt;
  }
  try {
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error:" << e.what() << "\n\n";
    std::cerr << desc << "\n";
    return std::nullopt;
  }

  if (vm.count("config-file")) {
    std::ifstream ifs(vm["config-file"].as<std::string>().c_str());
    po::store(po::parse_config_file(ifs, desc), vm);
    po::notify(vm);
  }

  options.my_id = vm["my-id"].as<std::size_t>();
  options.num_repetitions = vm["repetitions"].as<std::size_t>();
  options.no_run = vm["no-run"].as<bool>();
  options.fake_triples = vm["fake-triples"].as<bool>();
  if (options.my_id > 1) {
    std::cerr << "my-id must be one of 0 and 1\n";
    return std::nullopt;
  }
  auto protocol = vm["protocol"].as<std::string>();
  boost::algorithm::to_lower(protocol);
  if (protocol == "gmw") {
    options.protocol = MOTION::MPCProtocol::ArithmeticGMW;
  } else if (protocol == "beavy") {
    options.protocol = MOTION::MPCProtocol::ArithmeticBEAVY;
  } else {
    std::cerr << "invalid protocol: " << protocol << "\n";
    return std::nullopt;
  }

  options.model_path = vm["model"].as<std::string>();

  const auto parse_party_argument =
      [](const auto& s) -> std::pair<std::size_t, MOTION::Communication::tcp_connection_config> {
    const static std::regex party_argument_re("([01]),([^,]+),(\\d{1,5})");
    std::smatch match;
    if (!std::regex_match(s, match, party_argument_re)) {
      throw std::invalid_argument("invalid party argument");
    }
    auto id = boost::lexical_cast<std::size_t>(match[1]);
    auto host = match[2];
    auto port = boost::lexical_cast<std::uint16_t>(match[3]);
    return {id, {host, port}};
  };

  const std::vector<std::string> party_infos = vm["party"].as<std::vector<std::string>>();
  if (party_infos.size() != 2) {
    std::cerr << "expecting two --party options\n";
    return std::nullopt;
  }

  options.tcp_config.resize(2);
  std::size_t other_id = 2;

  const auto [id0, conn_info0] = parse_party_argument(party_infos[0]);
  const auto [id1, conn_info1] = parse_party_argument(party_infos[1]);
  if (id0 == id1) {
    std::cerr << "need party arguments for party 0 and 1\n";
    return std::nullopt;
  }
  options.tcp_config[id0] = conn_info0;
  options.tcp_config[id1] = conn_info1;

  return options;
}

std::unique_ptr<MOTION::Communication::CommunicationLayer> setup_communication(
    const Options& options) {
  MOTION::Communication::TCPSetupHelper helper(options.my_id, options.tcp_config);
  return std::make_unique<MOTION::Communication::CommunicationLayer>(options.my_id,
                                                                     helper.setup_connections());
}

void run_model(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  MOTION::onnx::OnnxAdapter onnx_adapter(backend, options.protocol, MOTION::MPCProtocol::Yao,
                                         options.my_id == 0);
  onnx_adapter.load_model(options.model_path);

  if (options.no_run) {
    return;
  }

  for (auto& pair : onnx_adapter.get_input_promises()) {
    auto& [tensor_dims, promise] = pair.second;
    promise.set_value(MOTION::Helpers::RandomVector<std::uint64_t>(tensor_dims.get_data_size()));
  }

  backend.run();

  if (options.my_id == 1) {
    for (auto& pair : onnx_adapter.get_output_futures()) {
      auto& [tensor_dims, future] = pair.second;
      future.get();
    }
  }
}

int main(int argc, char* argv[]) {
  auto options = parse_program_options(argc, argv);
  if (!options.has_value()) {
    return EXIT_FAILURE;
  }

  try {
    auto comm_layer = setup_communication(*options);
    auto logger = std::make_shared<MOTION::Logger>(options->my_id,
                                                   boost::log::trivial::severity_level::trace);
    comm_layer->set_logger(logger);
    MOTION::TwoPartyTensorBackend backend(*comm_layer, logger, options->fake_triples);
    run_model(*options, backend);
    comm_layer->shutdown();
  } catch (std::runtime_error& e) {
    std::cerr << "ERROR OCCURRED: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
