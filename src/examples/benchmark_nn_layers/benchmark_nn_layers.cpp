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
#include <memory>
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

#include "base/two_party_tensor_backend.h"
#include "communication/communication_layer.h"
#include "communication/tcp_transport.h"
#include "protocols/beavy/tensor.h"
#include "protocols/gmw/tensor.h"
#include "statistics/analysis.h"
#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include "tensor/tensor_op_factory.h"
#include "utility/helpers.h"
#include "utility/logger.h"
#include "utility/typedefs.h"

namespace po = boost::program_options;

struct Options {
  bool json;
  std::size_t num_threads;
  std::size_t num_repetitions;
  std::size_t bit_size;
  std::size_t fractional_bits;
  std::size_t my_id;
  MOTION::Communication::tcp_parties_config tcp_config;
  std::string experiment_name;
  std::string benchmark;
  std::size_t relu_variant;
  std::size_t relu_size;
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
    ("benchmark", po::value<std::string>()->required(), "benchmark name")
    ("relu-variant", po::value<std::size_t>(), "variant of ReLU layer")
    ("relu-size", po::value<std::size_t>(), "size of ReLU layer")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions")
    ("bit-size", po::value<std::size_t>()->default_value(64),
     "number of bits per number (32 or 64)")
    ("fractional-bits", po::value<std::size_t>()->default_value(16),
     "number of fractional bits for fixed-point arithmetic")
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
  options.num_threads = vm["threads"].as<std::size_t>();
  options.json = vm["json"].as<bool>();
  options.num_repetitions = vm["repetitions"].as<std::size_t>();
  options.bit_size = vm["bit-size"].as<std::size_t>();
  options.fractional_bits = vm["fractional-bits"].as<std::size_t>();

  options.benchmark = vm["benchmark"].as<std::string>();
  boost::algorithm::to_lower(options.benchmark);
  if (options.benchmark == "relu") {
    if (vm.count("relu-variant") == 0 || vm.count("relu-size") == 0) {
      std::cerr << "ReLU benchmark needs arguments --relu-variant and --relu-size\n";
      return std::nullopt;
    }
    options.relu_variant = vm["relu-variant"].as<std::size_t>();
    options.relu_size = vm["relu-size"].as<std::size_t>();
    options.experiment_name = fmt::format("relu-{}-{}", options.relu_variant, options.relu_size);
  } else {
    std::cerr << "unknown benchmark: " << options.benchmark << "\n";
    return std::nullopt;
  }

  if (options.my_id > 1) {
    std::cerr << "my-id must be one of 0 and 1\n";
    return std::nullopt;
  }

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

template <typename T>
auto make_input_share(MOTION::MPCProtocol protocol, MOTION::tensor::TensorDimensions dims) {
  if (protocol == MOTION::MPCProtocol::ArithmeticGMW) {
    auto t = std::make_shared<MOTION::proto::gmw::ArithmeticGMWTensor<T>>(dims);
    t->get_share() = MOTION::Helpers::RandomVector<T>(dims.get_data_size());
    t->set_online_ready();
    return std::dynamic_pointer_cast<const MOTION::tensor::Tensor>(t);
  } else if (protocol == MOTION::MPCProtocol::ArithmeticBEAVY) {
    auto t = std::make_shared<MOTION::proto::beavy::ArithmeticBEAVYTensor<T>>(dims);
    t->get_secret_share() = MOTION::Helpers::RandomVector<T>(dims.get_data_size());
    t->get_public_share() = std::vector<T>(dims.get_data_size(), 0x42);
    t->set_setup_ready();
    t->set_online_ready();
    return std::dynamic_pointer_cast<const MOTION::tensor::Tensor>(t);
  } else {
    throw std::invalid_argument("unexpected protocol");
  }
}

void prepare_relu(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  MOTION::tensor::TensorDimensions dims{
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 1, .width_ = options.relu_size};
  auto arithmetic_protocol = (options.relu_variant % 2 == 0) ? MOTION::MPCProtocol::ArithmeticGMW
                                                             : MOTION::MPCProtocol::ArithmeticBEAVY;
  auto boolean_protocol = [&options] {
    if (options.relu_variant < 2) {
      return MOTION::MPCProtocol::Yao;
    } else if (options.relu_variant == 2 || options.relu_variant == 4) {
      return MOTION::MPCProtocol::BooleanGMW;
    } else if (options.relu_variant == 3 || options.relu_variant == 5) {
      return MOTION::MPCProtocol::BooleanBEAVY;
    } else {
      throw std::invalid_argument("unexpected variant");
    }
  }();
  auto input_tensor = [&options, arithmetic_protocol, &dims] {
    switch (options.bit_size) {
      case 64:
        return make_input_share<std::uint64_t>(arithmetic_protocol, dims);
      case 32:
        return make_input_share<std::uint32_t>(arithmetic_protocol, dims);
      default:
        throw std::invalid_argument("unexpected bit size");
    }
  }();

  switch (options.relu_variant) {
    case 0:
    case 1:
    case 2:
    case 3: {
      auto input_boolean_tensor = backend.convert(boolean_protocol, input_tensor);
      auto output_boolean_tensor =
          backend.get_tensor_op_factory(boolean_protocol).make_tensor_relu_op(input_boolean_tensor);
      auto output_tensor = backend.convert(arithmetic_protocol, output_boolean_tensor);
      break;
    }
    case 4:
    case 5: {
      auto input_boolean_tensor = backend.convert(boolean_protocol, input_tensor);
      auto output_tensor = backend.get_tensor_op_factory(boolean_protocol)
                               .make_tensor_relu_op(input_boolean_tensor, input_tensor);
      break;
    }
    default:
      throw std::invalid_argument("unexpected variant");
  }
}

void run_benchmark(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  if (options.benchmark == "relu") {
    prepare_relu(options, backend);
  }

  backend.run();
}

void print_stats(const Options& options,
                 const MOTION::Statistics::AccumulatedRunTimeStats& run_time_stats,
                 const MOTION::Statistics::AccumulatedCommunicationStats& comm_stats) {
  if (options.json) {
    auto obj = MOTION::Statistics::to_json(options.experiment_name, run_time_stats, comm_stats);
    obj.emplace("party_id", options.my_id);
    obj.emplace("threads", options.num_threads);
    obj.emplace("bit-size", options.bit_size);
    obj.emplace("benchmark", options.benchmark);
    if (options.benchmark == "relu") {
      obj.emplace("relu-variant", options.relu_variant);
      obj.emplace("relu-size", options.relu_size);
    }
    std::cout << boost::json::to_string(obj) << "\n";
  } else {
    std::cout << MOTION::Statistics::print_stats(options.experiment_name, run_time_stats,
                                                 comm_stats);
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
      MOTION::TwoPartyTensorBackend backend(*comm_layer, options->num_threads, logger);
      run_benchmark(*options, backend);
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
