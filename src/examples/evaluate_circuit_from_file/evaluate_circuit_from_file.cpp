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
#include <fstream>
#include <iostream>
#include <random>
#include <regex>

#include <fmt/format.h>
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

bool CheckPartyArgumentSyntax(const std::string& p);

std::pair<po::variables_map, bool> ParseProgramOptions(int ac, char* av[]);

std::unique_ptr<MOTION::Communication::CommunicationLayer> setup_communication(
    const po::variables_map& vm);

int main(int ac, char* av[]) {
  try {
    auto [vm, help_flag] = ParseProgramOptions(ac, av);
    // if help flag is set - print allowed command line arguments and exit
    if (help_flag) return EXIT_SUCCESS;

    auto comm_layer = setup_communication(vm);
    auto my_id = comm_layer->get_my_id();

    auto logger =
        std::make_shared<MOTION::Logger>(my_id, boost::log::trivial::severity_level::trace);
    comm_layer->set_logger(logger);
    MOTION::TwoPartyBackend backend(*comm_layer, logger);
    ENCRYPTO::AlgorithmDescription algo;
    if (vm.count("fashion")) {
      algo = ENCRYPTO::AlgorithmDescription::FromBristolFashion(vm["circuit"].as<std::string>());
    } else {
      algo = ENCRYPTO::AlgorithmDescription::FromBristol(vm["circuit"].as<std::string>());
    }
    auto proto = [] (auto& vm) {
      auto proto = vm["protocol"].template as<std::string>();
      std::transform(std::begin(proto), std::end(proto), std::begin(proto),
                     [](char x) { return std::tolower(x); });
      if (proto == "yao") {
        return MOTION::MPCProtocol::Yao;
      } else if (proto == "gmw") {
        return MOTION::MPCProtocol::BooleanGMW;
      } else {
        throw std::invalid_argument("unknown protocol");
      }
    }(vm);
    auto& gate_factory = backend.get_gate_factory(proto);
    ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>> input_promise;
    MOTION::WireVector w_in_a;
    MOTION::WireVector w_in_b;
    if (my_id == 0) {
      auto pair = gate_factory.make_boolean_input_gate_my(my_id, algo.n_input_wires_parent_a_, 1);
      input_promise = std::move(pair.first);
      w_in_a = std::move(pair.second);
      w_in_b =
          gate_factory.make_boolean_input_gate_other(1 - my_id, *algo.n_input_wires_parent_b_, 1);
    } else {
      w_in_a =
          gate_factory.make_boolean_input_gate_other(1 - my_id, algo.n_input_wires_parent_a_, 1);
      auto pair = gate_factory.make_boolean_input_gate_my(my_id, *algo.n_input_wires_parent_b_, 1);
      input_promise = std::move(pair.first);
      w_in_b = std::move(pair.second);
    }
    auto w_out = backend.make_circuit(algo, w_in_a, w_in_b);
    auto output_future = gate_factory.make_boolean_output_gate_my(MOTION::ALL_PARTIES, w_out);

    backend.run_preprocessing();

    std::vector<ENCRYPTO::BitVector<>> inputs;
    if (my_id == 0) {
      std::generate_n(std::back_inserter(inputs), algo.n_input_wires_parent_a_,
                      [] { return ENCRYPTO::BitVector<>(1); });
    } else {
      std::generate_n(std::back_inserter(inputs), *algo.n_input_wires_parent_b_,
                      [] { return ENCRYPTO::BitVector<>(1); });
    }
    input_promise.set_value(inputs);

    backend.run();

    output_future.get();

    comm_layer->shutdown();

  } catch (std::runtime_error& e) {
    std::cerr << e.what() << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

const std::regex party_argument_re("(\\d+),(\\S+),(\\d{1,5})");

bool CheckPartyArgumentSyntax(const std::string& p) {
  // other party's id, IP address, and port
  return std::regex_match(p, party_argument_re);
}

std::tuple<std::size_t, std::string, std::uint16_t> ParsePartyArgument(const std::string& p) {
  std::smatch match;
  std::regex_match(p, match, party_argument_re);
  auto id = boost::lexical_cast<std::size_t>(match[1]);
  auto host = match[2];
  auto port = boost::lexical_cast<std::uint16_t>(match[3]);
  return {id, host, port};
}

// <variables map, help flag>
std::pair<po::variables_map, bool> ParseProgramOptions(int ac, char* av[]) {
  using namespace std::string_view_literals;
  constexpr std::string_view config_file_msg =
      "config file, other arguments will overwrite the parameters read from the config file"sv;
  bool print, help;
  boost::program_options::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ("help,h", po::bool_switch(&help)->default_value(false),"produce help message")
      ("config-file,f", po::value<std::string>(), config_file_msg.data())
      ("my-id", po::value<std::size_t>(), "my party id")
      ("other-parties", po::value<std::vector<std::string>>()->multitoken(), "(other party id, IP, port, my role), e.g., --other-parties 1,127.0.0.1,7777")
      ("protocol", po::value<std::string>()->required(), "protocol to use")
      ("circuit", po::value<std::string>()->required(), "path to a circuit file in the Bristol format")
      ("fashion", "use the newer Bristol *Fashion* format");
  // clang-format on

  po::variables_map vm;

  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  // argument help or no arguments (at least a config file is expected)
  if (help) {
    std::cout << desc << "\n";
    return std::make_pair<po::variables_map, bool>({}, true);
  }

  // read config file
  if (vm.count("config-file")) {
    std::ifstream ifs(vm["config-file"].as<std::string>().c_str());
    po::variables_map vm_config_file;
    po::store(po::parse_config_file(ifs, desc), vm);
    po::notify(vm);
  }

  // print parsed parameters
  if (vm.count("my-id")) {
    if (print) std::cout << "My id " << vm["my-id"].as<std::size_t>() << std::endl;
  } else
    throw std::runtime_error("My id is not set but required");

  if (vm.count("other-parties")) {
    const std::vector<std::string> other_parties{
        vm["other-parties"].as<std::vector<std::string>>()};
    std::string parties("Other parties: ");
    for (auto& p : other_parties) {
      if (CheckPartyArgumentSyntax(p)) {
        if (print) parties.append(" " + p);
      } else {
        throw std::runtime_error("Incorrect party argument syntax " + p);
      }
    }
    if (print) std::cout << parties << std::endl;
  } else
    throw std::runtime_error("Other parties' information is not set but required");

  if (print) {
    std::cout << "Number of SIMD AES evaluations: " << vm["num-simd"].as<std::size_t>()
              << std::endl;

    std::cout << "MPC Protocol: " << vm["protocol"].as<std::string>() << std::endl;
  }
  return std::make_pair(vm, help);
}

std::unique_ptr<MOTION::Communication::CommunicationLayer> setup_communication(
    const po::variables_map& vm) {
  const auto parties_str{vm["other-parties"].as<const std::vector<std::string>>()};
  const auto num_parties{parties_str.size()};
  const auto my_id{vm["my-id"].as<std::size_t>()};
  if (my_id >= num_parties) {
    throw std::runtime_error(fmt::format(
        "My id needs to be in the range [0, #parties - 1], current my id is {} and #parties is {}",
        my_id, num_parties));
  }

  MOTION::Communication::tcp_parties_config parties_config(num_parties);

  for (const auto& party_str : parties_str) {
    const auto [party_id, host, port] = ParsePartyArgument(party_str);
    if (party_id >= num_parties) {
      throw std::runtime_error(
          fmt::format("Party's id needs to be in the range [0, #parties - 1], current id "
                      "is {} and #parties is {}",
                      party_id, num_parties));
    }
    parties_config.at(party_id) = std::make_pair(host, port);
  }
  MOTION::Communication::TCPSetupHelper helper(my_id, parties_config);
  return std::make_unique<MOTION::Communication::CommunicationLayer>(my_id,
                                                                     helper.setup_connections());
}
