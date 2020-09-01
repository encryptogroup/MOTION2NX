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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/json/to_string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>

#include "base/two_party_backend.h"
#include "communication/communication_layer.h"
#include "communication/tcp_transport.h"
#include "hycc_adapter.h"
#include "statistics/analysis.h"
#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include "tensor/tensor_op_factory.h"
#include "utility/helpers.h"
#include "utility/logger.h"
#include "utility/typedefs.h"

namespace po = boost::program_options;

struct Options {
  std::size_t threads;
  bool json;
  std::size_t num_repetitions;
  std::size_t num_simd;
  bool sync_between_setup_and_online;
  MOTION::MPCProtocol arithmetic_protocol;
  MOTION::MPCProtocol boolean_protocol;
  std::size_t my_id;
  MOTION::Communication::tcp_parties_config tcp_config;
  std::string circuit_path;
  bool no_run = false;
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
    ("threads", po::value<std::size_t>()->default_value(0), "number of threads to use for gate evaluation")
    ("json", po::bool_switch()->default_value(false), "output data in JSON format")
    ("arithmetic-protocol", po::value<std::string>()->required(), "2PC protocol (GMW or BEAVY)")
    ("boolean-protocol", po::value<std::string>()->required(), "2PC protocol (GMW or BEAVY)")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("num-simd", po::value<std::size_t>()->default_value(1), "size of SIMD operations")
    ("sync-between-setup-and-online", po::bool_switch()->default_value(false),
     "run a synchronization protocol before the online phase starts")
    ("no-run", po::bool_switch()->default_value(false),
     "just build the network, but not execute it")
    ("circuit", po::value<std::string>()->required(), "path to a HyCC circuit file");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  bool help = vm["help"].as<bool>();
  if (help) {
    std::cerr << desc << "\n";
    return std::nullopt;
  }
  if (vm.count("config-file")) {
    std::ifstream ifs(vm["config-file"].as<std::string>().c_str());
    po::store(po::parse_config_file(ifs, desc), vm);
  }
  try {
    po::notify(vm);
  } catch (std::exception& e) {
    std::cerr << "error:" << e.what() << "\n\n";
    std::cerr << desc << "\n";
    return std::nullopt;
  }

  options.my_id = vm["my-id"].as<std::size_t>();
  options.threads = vm["threads"].as<std::size_t>();
  options.json = vm["json"].as<bool>();
  options.num_repetitions = vm["repetitions"].as<std::size_t>();
  options.num_simd = vm["num-simd"].as<std::size_t>();
  options.sync_between_setup_and_online = vm["sync-between-setup-and-online"].as<bool>();
  options.no_run = vm["no-run"].as<bool>();
  if (options.my_id > 1) {
    std::cerr << "my-id must be one of 0 and 1\n";
    return std::nullopt;
  }

  {
    auto boolean_protocol = vm["boolean-protocol"].as<std::string>();
    boost::algorithm::to_lower(boolean_protocol);
    if (boolean_protocol == "gmw") {
      options.boolean_protocol = MOTION::MPCProtocol::BooleanGMW;
    } else if (boolean_protocol == "beavy") {
      options.boolean_protocol = MOTION::MPCProtocol::BooleanBEAVY;
    } else {
      std::cerr << "invalid Boolean protocol: " << boolean_protocol << "\n";
      return std::nullopt;
    }

    auto arithmetic_protocol = vm["arithmetic-protocol"].as<std::string>();
    boost::algorithm::to_lower(arithmetic_protocol);
    if (arithmetic_protocol == "gmw") {
      options.arithmetic_protocol = MOTION::MPCProtocol::ArithmeticGMW;
    } else if (arithmetic_protocol == "beavy") {
      options.arithmetic_protocol = MOTION::MPCProtocol::ArithmeticBEAVY;
    } else {
      std::cerr << "invalid Arithmetic protocol: " << arithmetic_protocol << "\n";
      return std::nullopt;
    }
  }

  options.circuit_path = vm["circuit"].as<std::string>();

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

void set_inputs(const Options& options, MOTION::hycc::HyCCAdapter& hycc_adapter) {
  for (auto& pair : hycc_adapter.get_input_bit_promises()) {
    auto& [size, promise] = pair.second;
    std::vector<ENCRYPTO::BitVector<>> inputs;
    inputs.reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
      inputs.emplace_back(ENCRYPTO::BitVector<>::Random(options.num_simd));
    }
    promise.set_value(std::move(inputs));
  }
  for (auto& pair : hycc_adapter.get_input_int8_promises()) {
    auto& [size, promise] = pair.second;
    promise.set_value(MOTION::Helpers::RandomVector<std::uint8_t>(options.num_simd * size));
  }
  for (auto& pair : hycc_adapter.get_input_int16_promises()) {
    auto& [size, promise] = pair.second;
    promise.set_value(MOTION::Helpers::RandomVector<std::uint16_t>(options.num_simd * size));
  }
  for (auto& pair : hycc_adapter.get_input_int32_promises()) {
    auto& [size, promise] = pair.second;
    promise.set_value(MOTION::Helpers::RandomVector<std::uint32_t>(options.num_simd * size));
  }
  for (auto& pair : hycc_adapter.get_input_int64_promises()) {
    auto& [size, promise] = pair.second;
    promise.set_value(MOTION::Helpers::RandomVector<std::uint64_t>(options.num_simd * size));
  }
}

void collect_outputs(const Options& options, MOTION::hycc::HyCCAdapter& hycc_adapter) {
  for (auto& pair : hycc_adapter.get_output_bit_futures()) {
    auto& [size, future] = pair.second;
    future.get();
  }
  for (auto& pair : hycc_adapter.get_output_int8_futures()) {
    auto& [size, future] = pair.second;
    future.get();
  }
  for (auto& pair : hycc_adapter.get_output_int16_futures()) {
    auto& [size, future] = pair.second;
    future.get();
  }
  for (auto& pair : hycc_adapter.get_output_int32_futures()) {
    auto& [size, future] = pair.second;
    future.get();
  }
  for (auto& pair : hycc_adapter.get_output_int64_futures()) {
    auto& [size, future] = pair.second;
    future.get();
  }
}

void run_model(const Options& options, MOTION::TwoPartyBackend& backend,
               std::shared_ptr<MOTION::Logger> logger) {
  MOTION::hycc::HyCCAdapter hycc_adapter(options.my_id, backend, options.arithmetic_protocol,
                                         options.boolean_protocol, MOTION::MPCProtocol::Yao,
                                         options.num_simd, logger);
  hycc_adapter.load_circuit(options.circuit_path);
  hycc_adapter.clear_hycc_data();

  if (options.no_run) {
    return;
  }

  set_inputs(options, hycc_adapter);

  backend.run();

  collect_outputs(options, hycc_adapter);
}

void print_stats(const Options& options,
                 const MOTION::Statistics::AccumulatedRunTimeStats& run_time_stats,
                 const MOTION::Statistics::AccumulatedCommunicationStats& comm_stats) {
  const auto filename = std::filesystem::path(options.circuit_path).filename();
  if (options.json) {
    auto obj = MOTION::Statistics::to_json(filename, run_time_stats, comm_stats);
    obj.emplace("party_id", options.my_id);
    obj.emplace("boolean_protocol", MOTION::ToString(options.boolean_protocol));
    obj.emplace("arithmetic_protocol", MOTION::ToString(options.arithmetic_protocol));
    obj.emplace("simd", options.num_simd);
    obj.emplace("threads", options.threads);
    obj.emplace("sync_between_setup_and_online", options.sync_between_setup_and_online);
    obj.emplace("circuit_path", options.circuit_path);
    std::cout << boost::json::to_string(obj) << "\n";
  } else {
    std::cout << MOTION::Statistics::print_stats(filename, run_time_stats, comm_stats);
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
    MOTION::Statistics::AccumulatedRunTimeStats run_time_stats;
    MOTION::Statistics::AccumulatedCommunicationStats comm_stats;
    for (std::size_t i = 0; i < options->num_repetitions; ++i) {
      MOTION::TwoPartyBackend backend(*comm_layer, options->threads,
                                      options->sync_between_setup_and_online, logger);
      run_model(*options, backend, logger);
      comm_layer->sync();
      comm_stats.add(comm_layer->get_transport_statistics());
      comm_layer->reset_transport_statistics();
      run_time_stats.add(backend.get_run_time_stats());
    }
    comm_layer->shutdown();
    print_stats(*options, run_time_stats, comm_stats);
  } catch (std::runtime_error& e) {
    std::cerr << "ERROR OCCURRED: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
