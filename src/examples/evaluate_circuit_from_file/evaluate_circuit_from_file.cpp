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

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <regex>

#include <fmt/format.h>
#include <boost/algorithm/string.hpp>
#include <boost/json/to_string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/program_options.hpp>
#include <stdexcept>

#include "algorithm/algorithm_description.h"
#include "base/two_party_backend.h"
#include "communication/communication_layer.h"
#include "communication/tcp_transport.h"
#include "statistics/analysis.h"
#include "utility/logger.h"

namespace po = boost::program_options;

struct Options {
  std::size_t threads;
  bool json;
  std::size_t num_repetitions;
  std::size_t num_simd;
  bool sync_between_setup_and_online;
  MOTION::MPCProtocol protocol;
  std::size_t my_id;
  MOTION::Communication::tcp_parties_config tcp_config;
  std::string circuit_path;
  bool fashion;
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
    ("protocol", po::value<std::string>()->required(), "2PC protocol (Yao, GMW or BEAVY)")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("num-simd", po::value<std::size_t>()->default_value(1), "number of SIMD values")
    ("sync-between-setup-and-online", po::value<bool>()->default_value(false),
     "run a synchronization protocol before the online phase starts")
    ("no-run", po::bool_switch()->default_value(false), "just build the circuit, but not execute it")
    ("circuit", po::value<std::string>()->required(), "path to a circuit file in the Bristol format")
    ("fashion", po::bool_switch()->default_value(false), "output data in JSON format")
    ;
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
  auto protocol = vm["protocol"].as<std::string>();
  boost::algorithm::to_lower(protocol);
  if (protocol == "yao") {
    options.protocol = MOTION::MPCProtocol::Yao;
  } else if (protocol == "gmw") {
    options.protocol = MOTION::MPCProtocol::BooleanGMW;
  } else if (protocol == "beavy") {
    options.protocol = MOTION::MPCProtocol::BooleanBEAVY;
  } else {
    std::cerr << "invalid protocol: " << protocol << "\n";
    return std::nullopt;
  }

  options.circuit_path = vm["circuit"].as<std::string>();
  options.fashion = vm["fashion"].as<bool>();

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

void run_circuit(const Options& options, MOTION::TwoPartyBackend& backend) {
  ENCRYPTO::AlgorithmDescription algo;
  if (options.fashion) {
    algo = ENCRYPTO::AlgorithmDescription::FromBristolFashion(options.circuit_path);
  } else {
    algo = ENCRYPTO::AlgorithmDescription::FromBristol(options.circuit_path);
  }
  auto& gate_factory = backend.get_gate_factory(options.protocol);
  ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> input_promise;
  MOTION::WireVector w_in_a;
  MOTION::WireVector w_in_b;
  if (options.my_id == 0) {
    auto pair = gate_factory.make_boolean_input_gate_my(options.my_id, algo.n_input_wires_parent_a_,
                                                        options.num_simd);
    input_promise = std::move(pair.first);
    w_in_a = std::move(pair.second);
    w_in_b = gate_factory.make_boolean_input_gate_other(
        1 - options.my_id, *algo.n_input_wires_parent_b_, options.num_simd);
  } else {
    w_in_a = gate_factory.make_boolean_input_gate_other(
        1 - options.my_id, algo.n_input_wires_parent_a_, options.num_simd);
    auto pair = gate_factory.make_boolean_input_gate_my(
        options.my_id, *algo.n_input_wires_parent_b_, options.num_simd);
    input_promise = std::move(pair.first);
    w_in_b = std::move(pair.second);
  }
  auto w_out = backend.make_circuit(algo, w_in_a, w_in_b);
  auto output_future = gate_factory.make_boolean_output_gate_my(MOTION::ALL_PARTIES, w_out);

  if (options.no_run) {
    return;
  }

  std::vector<ENCRYPTO::BitVector<>> inputs;
  if (options.my_id == 0) {
    std::generate_n(std::back_inserter(inputs), algo.n_input_wires_parent_a_,
                    [options] { return ENCRYPTO::BitVector<>(options.num_simd); });
  } else {
    std::generate_n(std::back_inserter(inputs), *algo.n_input_wires_parent_b_,
                    [options] { return ENCRYPTO::BitVector<>(options.num_simd); });
  }
  input_promise.set_value(inputs);

  backend.run();

  output_future.get();
}

void print_stats(const Options& options,
                 const MOTION::Statistics::AccumulatedRunTimeStats& run_time_stats,
                 const MOTION::Statistics::AccumulatedCommunicationStats& comm_stats) {
  const auto filename = std::filesystem::path(options.circuit_path).filename();
  if (options.json) {
    auto obj = MOTION::Statistics::to_json(filename, run_time_stats, comm_stats);
    obj.emplace("party_id", options.my_id);
    obj.emplace("protocol", MOTION::ToString(options.protocol));
    obj.emplace("threads", options.threads);
    obj.emplace("sync_between_setup_and_online", options.sync_between_setup_and_online);
    obj.emplace("circuit_path", options.circuit_path);
    obj.emplace("fashion", options.fashion);
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
      run_circuit(*options, backend);
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
