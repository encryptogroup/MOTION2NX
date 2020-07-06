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
#include "tensor/tensor.h"
#include "tensor/tensor_op.h"
#include "tensor/tensor_op_factory.h"
#include "utility/logger.h"
#include "utility/typedefs.h"

namespace po = boost::program_options;

struct Options {
  std::size_t num_repetitions;
  MOTION::MPCProtocol protocol;
  std::size_t my_id;
  MOTION::Communication::tcp_parties_config tcp_config;
};

std::optional<Options> parse_program_options(int argc, char* argv[]) {
  boost::program_options::options_description desc("Allowed options");
  bool help;
  // clang-format off
  desc.add_options()
    ("help,h", po::bool_switch(&help)->default_value(false),"produce help message")
    ("config-file", po::value<std::string>(), "config file containing options")
    ("my-id", po::value<std::size_t>(), "my party id")
    ("party", po::value<std::vector<std::string>>()->multitoken(), "(party id, IP, port), e.g., --party 1,127.0.0.1,7777")
    ("protocol", po::value<std::string>(), "2PC protocol (GMW or BEAVY)")
    ("repetitions", po::value<std::size_t>()->default_value(1), "number of repetitions");
  // clang-format on

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (help) {
    std::cerr << desc << "\n";
    return std::nullopt;
  }

  if (vm.count("config-file")) {
    std::ifstream ifs(vm["config-file"].as<std::string>().c_str());
    po::store(po::parse_config_file(ifs, desc), vm);
    po::notify(vm);
  }

  Options options;
  options.my_id = vm["my-id"].as<std::size_t>();
  options.num_repetitions = vm["repetitions"].as<std::size_t>();
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

ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>> build_cryptonets(
    const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  auto& tof = backend.get_tensor_op_factory(options.protocol);

  const MOTION::tensor::TensorDimensions input_dims{
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 28, .width_ = 28};
  const MOTION::tensor::TensorDimensions conv_weights_dims{
      .batch_size_ = 5, .num_channels_ = 1, .height_ = 5, .width_ = 5};
  const MOTION::tensor::TensorDimensions squashed_weights_dims{
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 845, .width_ = 100};
  const MOTION::tensor::TensorDimensions fully_connected_weights_dims{
      .batch_size_ = 1, .num_channels_ = 1, .height_ = 100, .width_ = 10};

  const MOTION::tensor::Conv2DOp conv_op = {.kernel_shape_ = {5, 1, 5, 5},
                                            .input_shape_ = {1, 28, 28},
                                            .output_shape_ = {5, 13, 13},
                                            .dilations_ = {1, 1},
                                            .pads_ = {1, 1, 0, 0},
                                            .strides_ = {2, 2}};
  const MOTION::tensor::GemmOp gemm_op_1 = {
      .input_A_shape_ = {1, 845}, .input_B_shape_ = {845, 100}, .output_shape_ = {1, 100}};
  const MOTION::tensor::GemmOp gemm_op_2 = {
      .input_A_shape_ = {1, 100}, .input_B_shape_ = {100, 10}, .output_shape_ = {1, 10}};

  MOTION::tensor::TensorCP input_tensor;
  MOTION::tensor::TensorCP conv_weights_tensor;
  MOTION::tensor::TensorCP squashed_weights_tensor;
  MOTION::tensor::TensorCP fully_connected_weights_tensor;

  if (options.my_id == 0) {
    input_tensor = tof.make_arithmetic_64_tensor_input_other(input_dims);
    auto ret1 = tof.make_arithmetic_64_tensor_input_my(conv_weights_dims);
    conv_weights_tensor = ret1.second;
    ret1.first.set_value(
        MOTION::Helpers::RandomVector<std::uint64_t>(conv_weights_dims.get_data_size()));
    auto ret2 = tof.make_arithmetic_64_tensor_input_my(squashed_weights_dims);
    squashed_weights_tensor = ret2.second;
    ret2.first.set_value(
        MOTION::Helpers::RandomVector<std::uint64_t>(squashed_weights_dims.get_data_size()));
    auto ret3 = tof.make_arithmetic_64_tensor_input_my(fully_connected_weights_dims);
    fully_connected_weights_tensor = ret3.second;
    ret3.first.set_value(
        MOTION::Helpers::RandomVector<std::uint64_t>(fully_connected_weights_dims.get_data_size()));
  } else {
    auto ret = tof.make_arithmetic_64_tensor_input_my(input_dims);
    input_tensor = ret.second;
    ret.first.set_value(MOTION::Helpers::RandomVector<std::uint64_t>(input_dims.get_data_size()));

    conv_weights_tensor = tof.make_arithmetic_64_tensor_input_other(conv_weights_dims);
    squashed_weights_tensor = tof.make_arithmetic_64_tensor_input_other(squashed_weights_dims);
    fully_connected_weights_tensor =
        tof.make_arithmetic_64_tensor_input_other(fully_connected_weights_dims);
  }

  auto conv_output =
      tof.make_arithmetic_tensor_conv2d_op(conv_op, input_tensor, conv_weights_tensor);
  auto sqr_1_output = tof.make_arithmetic_tensor_sqr_op(conv_output);
  auto flatten_output = tof.make_tensor_flatten_op(sqr_1_output, 0);
  auto squashed_output =
      tof.make_arithmetic_tensor_gemm_op(gemm_op_1, flatten_output, squashed_weights_tensor);
  auto sqr_2_output = tof.make_arithmetic_tensor_sqr_op(squashed_output);
  auto fully_connected_output =
      tof.make_arithmetic_tensor_gemm_op(gemm_op_2, sqr_2_output, fully_connected_weights_tensor);

  ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>> output_future;
  if (options.my_id == 0) {
    tof.make_arithmetic_tensor_output_other(fully_connected_output);
  } else {
    output_future = tof.make_arithmetic_64_tensor_output_my(fully_connected_output);
  }
  return output_future;
}

void run_cryptonets(const Options& options, MOTION::TwoPartyTensorBackend& backend) {
  auto output_future = build_cryptonets(options, backend);
  backend.run();
  if (options.my_id == 1) {
    output_future.get();
  }
}

int main(int argc, char* argv[]) {
  auto options = parse_program_options(argc, argv);
  if (!options.has_value()) {
    return EXIT_FAILURE;
  }

  try {
    auto comm_layer = setup_communication(*options);
    auto logger =
        std::make_shared<MOTION::Logger>(options->my_id, boost::log::trivial::severity_level::trace);
    comm_layer->set_logger(logger);
    MOTION::TwoPartyTensorBackend backend(*comm_layer, logger);
    run_cryptonets(*options, backend);
    comm_layer->shutdown();
  } catch (std::runtime_error& e) {
    std::cerr << "ERROR OCCURRED: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
