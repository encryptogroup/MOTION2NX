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

#include "hycc_adapter.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <libcircuit/simple_circuit.h>

#include "base/circuit_builder.h"
#include "base/gate_factory.h"
#include "protocols/plain/wire.h"
#include "utility/bit_vector.h"
#include "utility/logger.h"
#include "utility/reusable_future.h"
#include "wire/new_wire.h"

namespace MOTION::hycc {

using WireVector = std::vector<std::shared_ptr<NewWire>>;

struct HyCCAdapter::HyCCAdapterImpl {
  HyCCAdapterImpl(CircuitBuilder& circuit_builder) : circuit_builder_(circuit_builder) {}
  CircuitBuilder& circuit_builder_;
  std::size_t my_id_;
  std::size_t num_simd_;
  MPCProtocol arithmetic_protocol_;
  MPCProtocol boolean_protocol_;
  MPCProtocol yao_protocol_;
  MPCProtocol default_boolean_protocol_;
  std::size_t arithmetic_bit_size_;

  std::shared_ptr<Logger> logger_ = nullptr;
  loggert cbmc_logger_;

  std::vector<std::string> circuit_files_;
  std::filesystem::path circuit_directory_;
  std::unordered_map<std::string, simple_circuitt> cbmc_circuits_;

  // input_label -> (size, promise)
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>>>>
      bit_input_promises_;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint8_t>>>>
      uint8_input_promises_;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint16_t>>>>
      uint16_input_promises_;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint32_t>>>>
      uint32_input_promises_;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint64_t>>>>
      uint64_input_promises_;

  // output_label -> (size, future)
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>>>
      bit_output_futures_;
  std::unordered_map<
      std::string, std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint8_t>>>>
      uint8_output_futures_;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint16_t>>>>
      uint16_output_futures_;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint32_t>>>>
      uint32_output_futures_;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>>>>
      uint64_output_futures_;

  std::unordered_map<simple_circuitt::gatet::wire_endpointt, WireVector, wire_endpoint_hasht>
      arithmetic_wires_;
  std::unordered_map<simple_circuitt::gatet::wire_endpointt, WireVector, wire_endpoint_hasht>
      boolean_wires_;
  std::unordered_map<simple_circuitt::gatet::wire_endpointt, WireVector, wire_endpoint_hasht>
      yao_wires_;

  MPCProtocol get_protocol_from_filename(const std::string&);
  simple_circuitt& load_circuit_files();
  void create_input_gate(std::size_t input_owner, const std::string& label, std::size_t index,
                         simple_circuitt::gatet*);
  void process_circuit_inputs(const simple_circuitt&);
  void create_output_gate(std::size_t input_owner, const std::string& label, std::size_t index,
                          const simple_circuitt::gatet*);
  void process_circuit_outputs(const simple_circuitt&);
  // circuit traversal:
  static MPCProtocol get_protocol_from_gate(const simple_circuitt::gatet&);
  const WireVector& create_input_gate(const simple_circuitt::gatet::wire_endpointt&, MPCProtocol);
  const WireVector& get_or_create_wires(const simple_circuitt::gatet::wire_endpointt&, MPCProtocol);
  const WireVector& store_wires(const simple_circuitt::gatet::wire_endpointt&, MPCProtocol,
                                WireVector&&);

  void topological_traversal(simple_circuitt&);
  void visit_output_gate(simple_circuitt::gatet* gate);
  void visit_split_gate(simple_circuitt::gatet* gate);
  void visit_combine_gate(simple_circuitt::gatet* gate);
  void visit_const_gate(simple_circuitt::gatet* gate);
  void visit_unary_gate(simple_circuitt::gatet* gate, simple_circuitt::GATE_OP op);
  void visit_binary_gate(simple_circuitt::gatet* gate, simple_circuitt::GATE_OP op);
};

MPCProtocol HyCCAdapter::HyCCAdapterImpl::get_protocol_from_filename(const std::string& filename) {
  const static std::regex filename_re("\\S+@([^@]+).circ");
  std::smatch match;
  if (!std::regex_match(filename, match, filename_re)) {
    throw std::invalid_argument("invalid filename argument");
  }
  auto protocol_part = match[1];
  if (protocol_part == "bool_size") {
    return yao_protocol_;
  } else if (protocol_part == "bool_depth") {
    return boolean_protocol_;
  } else if (protocol_part == "arith") {
    return arithmetic_protocol_;
  } else {
    throw std::invalid_argument(
        fmt::format("unknown protocol specifier: '{}'", protocol_part.str()));
  }
}

static void set_mpc_protocol(simple_circuitt& circuit, MPCProtocol boolean_protocol,
                             MPCProtocol arithmetic_protocol) {
  std::cerr << "setting protocol of circuit\n";

  const auto set_protocol = [](simple_circuitt::gatet& gate, MPCProtocol protocol) {
    // Use the 64th bit to indicate whether a protocol has been set
    gate.user.uint_val |= 1ull << 63;
    gate.user.uint_val |= static_cast<unsigned int>(protocol);
  };

  set_protocol(circuit.get_one_gate(), boolean_protocol);
  set_protocol(circuit.get_zero_gate(), boolean_protocol);

  for (auto& gate : circuit.gates()) {
    if (gate.get_operation() == simple_circuitt::SPLIT || is_boolean_op(gate.get_operation())) {
      set_protocol(gate, boolean_protocol);
    } else if (gate.get_operation() == simple_circuitt::COMBINE ||
               is_arithmetic_op(gate.get_operation())) {
      set_protocol(gate, arithmetic_protocol);
    }
  }

  for (auto& input : circuit.inputs()) {
    // Assume that single bit inputs are boolean
    if (input.get_width() == 1) {
      set_protocol(input, boolean_protocol);
      // std::cerr << fmt::format("set protocol of input gate {} to {}\n", (void*)&input,
      //                          ToString(boolean_protocol));
    } else {
      set_protocol(input, arithmetic_protocol);
      // std::cerr << fmt::format("set protocol of input gate {} to {}\n", (void*)&input,
      //                          ToString(arithmetic_protocol));
    }
  }

  for (auto& output : circuit.outputs()) {
    // Assume that single bit outputs are boolean
    if (output.get_width() == 1)
      set_protocol(output, boolean_protocol);
    else
      set_protocol(output, arithmetic_protocol);
  }
}

MPCProtocol HyCCAdapter::HyCCAdapterImpl::get_protocol_from_gate(
    const simple_circuitt::gatet& gate) {
  if (gate.user.uint_val & (1ull << 63)) {
    auto protocol = static_cast<MPCProtocol>(gate.user.uint_val & 0xffffffff);
    if (protocol >= MPCProtocol::Invalid) {
      throw hycc_error("invalid protocol detected");
    }
    return protocol;
  }
  throw hycc_error("HyCC gatet has no assigned protocol");
}

static std::size_t get_arithmetic_bit_size(
    const std::unordered_map<std::string, simple_circuitt>& circuit_map) {
  const auto update_bit_size = [](int& cur_width, int new_width) {
    if (new_width > 1) {
      if (cur_width == 0) {
        cur_width = new_width;
      } else if (cur_width != new_width) {
        throw HyCCAdapter::hycc_error(fmt::format(
            "Different bit widths in arithmetic circuit: {} vs {}", cur_width, new_width));
      }
    }
  };

  int arithmetic_bit_size = 0;
  for (const auto& [_, circuit] : circuit_map) {
    if (circuit.get_number_of_gates())
      update_bit_size(arithmetic_bit_size, circuit.gates().b->get_width());
  }

  if (arithmetic_bit_size != 0 && arithmetic_bit_size != 8 && arithmetic_bit_size != 16 &&
      arithmetic_bit_size != 32 && arithmetic_bit_size != 64)
    throw HyCCAdapter::hycc_error(
        fmt::format("Invalid arithmeticmetic bit size: {}", arithmetic_bit_size));

  return arithmetic_bit_size;
}

simple_circuitt& HyCCAdapter::HyCCAdapterImpl::load_circuit_files() {
  std::cerr << "start loading circuit files\n";
  for (const auto& filename : circuit_files_) {
    std::cerr << fmt::format("loading circuit file '{}'\n", filename);
    const auto protocol = get_protocol_from_filename(filename);
    const auto file_path = circuit_directory_ / filename;
    simple_circuitt cbmc_circuit(cbmc_logger_, "");
    std::ifstream file(file_path);
    if (!file.is_open() || !file.good()) {
      throw hycc_error(fmt::format("could not open file: '{}'", file_path.string()));
    }
    cbmc_circuit.read(file);
    if (protocol == arithmetic_protocol_) {
      set_mpc_protocol(cbmc_circuit, default_boolean_protocol_, arithmetic_protocol_);
    } else {
      set_mpc_protocol(cbmc_circuit, protocol, arithmetic_protocol_);
    }
    auto circuit_name = cbmc_circuit.name();
    cbmc_circuits_.emplace(circuit_name, std::move(cbmc_circuit));
  }
  arithmetic_bit_size_ = get_arithmetic_bit_size(cbmc_circuits_);

  const std::string main_name = "mpc_main";
  auto it = cbmc_circuits_.find(main_name);
  if (it == cbmc_circuits_.end()) {
    throw hycc_error(fmt::format("Couldn't find main circuit '{}'", main_name));
  }
  auto& main_circuit = it->second;
  std::cerr << "linking circuits ...";
  main_circuit.link(cbmc_circuits_);
  std::cerr << " done\n";
  return main_circuit;
}

void HyCCAdapter::HyCCAdapterImpl::process_circuit_inputs(const simple_circuitt& circuit) {
  std::cerr << "processing circuit inputs\n";
  for (const auto* input_var : circuit.ordered_inputs()) {
    const auto& label = input_var->name;
    const auto input_owner = static_cast<std::size_t>(input_var->owner);
    auto& gates = input_var->gates;
    // std::cerr << "processing input:\n";
    // std::cerr << fmt::format("- label = {}\n", label);
    // std::cerr << fmt::format("- owner = {}\n", input_owner);
    // std::cerr << fmt::format("- gates: {}\n", gates.size());
    // - assume all gates have the same width: 1 (Boolean) or 8/16/32/64 (arithmetic)
    // - assume all gates have protocol
    // TODO
    // - detect protocol: cannot distinguish Y/B
    // - create single/multiple input gates (Boolean/arithmetic), split wires
    const auto num_gates = gates.size();
    for (std::size_t i = 0; i < num_gates; ++i) {
      // std::cerr << fmt::format("  + gate {} of width {}\n", (void*)gates[i], gates[i]->get_width());
      // std::cerr << fmt::format("    assigned protocol: {}\n",
      //                          ToString(get_protocol_from_gate(*gates[i])));
      create_input_gate(input_owner, label, i, gates[i]);
    }
  }
}

void HyCCAdapter::HyCCAdapterImpl::create_input_gate(std::size_t input_owner,
                                                     const std::string& label, std::size_t index,
                                                     simple_circuitt::gatet* gate) {
  // std::cerr << fmt::format("creating input gate {} ...", (const void*)gate);
  const auto protocol = get_protocol_from_gate(*gate);
  const auto input_label = fmt::format("{}-{}", label, index);
  const auto my_input = input_owner == my_id_;
  auto& gate_factory = circuit_builder_.get_gate_factory(protocol);
  WireVector wires;
  const auto save_promise = [](auto& map, const auto& key, auto size, auto&& promise) {
    auto [_, inserted] = map.emplace(key, std::make_pair(size, std::move(promise)));
    if (!inserted) {
      throw hycc_error(fmt::format("there is already a promise for label '{}'", key));
    }
  };

  if (protocol == boolean_protocol_ || protocol == yao_protocol_) {
    assert(gate->get_width() == 1);
    if (my_input) {
      auto [promise, input_wires] =
          gate_factory.make_boolean_input_gate_my(input_owner, 1, num_simd_);
      wires = std::move(input_wires);
      save_promise(bit_input_promises_, input_label, 1, std::move(promise));
    } else {
      wires = gate_factory.make_boolean_input_gate_other(input_owner, 1, num_simd_);
    }
  } else if (protocol == arithmetic_protocol_) {
    if (my_input) {
      switch (gate->get_width()) {
        case 8: {
          auto [promise, input_wires] =
              gate_factory.make_arithmetic_8_input_gate_my(input_owner, num_simd_);
          wires = std::move(input_wires);
          save_promise(uint8_input_promises_, input_label, 1, std::move(promise));
          break;
        }
        case 16: {
          auto [promise, input_wires] =
              gate_factory.make_arithmetic_16_input_gate_my(input_owner, num_simd_);
          wires = std::move(input_wires);
          save_promise(uint16_input_promises_, input_label, 1, std::move(promise));
        } break;
        case 32: {
          auto [promise, input_wires] =
              gate_factory.make_arithmetic_32_input_gate_my(input_owner, num_simd_);
          wires = std::move(input_wires);
          save_promise(uint32_input_promises_, input_label, 1, std::move(promise));
        } break;
        case 64: {
          auto [promise, input_wires] =
              gate_factory.make_arithmetic_64_input_gate_my(input_owner, num_simd_);
          wires = std::move(input_wires);
          save_promise(uint64_input_promises_, input_label, 1, std::move(promise));
        } break;
        default:
          throw hycc_error(
              fmt::format("unexpected width for arithmetic protocol: {}", gate->get_width()));
      }
    } else {
      switch (gate->get_width()) {
        case 8:
          wires = gate_factory.make_arithmetic_8_input_gate_other(input_owner, num_simd_);
          break;
        case 16:
          wires = gate_factory.make_arithmetic_16_input_gate_other(input_owner, num_simd_);
          break;
        case 32:
          wires = gate_factory.make_arithmetic_32_input_gate_other(input_owner, num_simd_);
          break;
        case 64:
          wires = gate_factory.make_arithmetic_64_input_gate_other(input_owner, num_simd_);
          break;
        default:
          throw hycc_error(
              fmt::format("unexpected width for arithmetic protocol: {}", gate->get_width()));
      }
    }
  } else {
    throw hycc_error(fmt::format("unexpected protocol {}", protocol));
  }
  store_wires(primary_output(const_cast<simple_circuitt::gatet*>(gate)), protocol,
              std::move(wires));
  // std::cerr << " done\n";
}

void HyCCAdapter::HyCCAdapterImpl::process_circuit_outputs(const simple_circuitt& circuit) {
  std::cerr << "processing circuit outputs\n";
  for (const auto* input_var : circuit.ordered_outputs()) {
    const auto& label = input_var->name;
    const std::size_t output_owner = 1;  // assume P_1 gets the output
    auto& gates = input_var->gates;
    // std::cerr << "processing output:\n";
    // std::cerr << fmt::format("- label = {}\n", label);
    // std::cerr << fmt::format("- owner = {}\n", output_owner);
    // std::cerr << fmt::format("- gates: {}\n", gates.size());
    // - assume all gates have the same width: 1 (Boolean) or 8/16/32/64 (arithmetic)
    // - assume all gates have protocol
    // TODO
    // - detect protocol: cannot distinguish Y/B
    // - create single/multiple input gates (Boolean/arithmetic), split wires
    const auto num_gates = gates.size();
    for (std::size_t i = 0; i < num_gates; ++i) {
      // std::cerr << fmt::format("  + gate {} of width {}\n", (void*)gates[i], gates[i]->get_width());
      // std::cerr << fmt::format("    assigned protocol: {}\n",
      //                          ToString(get_protocol_from_gate(*gates[i])));
      create_output_gate(output_owner, label, i, gates[i]);
    }
  }
}

void HyCCAdapter::HyCCAdapterImpl::create_output_gate(std::size_t output_owner,
                                                      const std::string& label, std::size_t index,
                                                      const simple_circuitt::gatet* gate) {
  // std::cerr << fmt::format("creating output gate {} ...", (const void*)gate);
  if (gate->num_fanins() != 1) {
    throw hycc_error(fmt::format("unexpected number of inputs: {}", gate->num_fanins()));
  }
  const auto protocol = get_protocol_from_gate(*gate);
  const auto output_label = fmt::format("{}-{}", label, index);
  const auto my_output = output_owner == my_id_;
  auto& gate_factory = circuit_builder_.get_gate_factory(protocol);
  const auto& key = gate->fanin_range()[0];
  WireVector input = get_or_create_wires(key, protocol);
  const auto save_future = [](auto& map, const auto& key, auto size, auto&& future) {
    auto [_, inserted] = map.emplace(key, std::make_pair(size, std::move(future)));
    if (!inserted) {
      throw hycc_error(fmt::format("there is already a future for label '{}'", key));
    }
  };

  if (protocol == boolean_protocol_ || protocol == yao_protocol_) {
    assert(gate->get_width() == 1);
    if (my_output) {
      auto future = gate_factory.make_boolean_output_gate_my(output_owner, input);
      save_future(bit_output_futures_, output_label, 1, std::move(future));
    } else {
      gate_factory.make_boolean_output_gate_other(output_owner, input);
    }
  } else if (protocol == arithmetic_protocol_) {
    if (my_output) {
      switch (gate->get_width()) {
        case 8: {
          auto future = gate_factory.make_arithmetic_8_output_gate_my(output_owner, input);
          save_future(uint8_output_futures_, output_label, 1, std::move(future));
          break;
        }
        case 16: {
          auto future = gate_factory.make_arithmetic_16_output_gate_my(output_owner, input);
          save_future(uint16_output_futures_, output_label, 1, std::move(future));
        } break;
        case 32: {
          auto future = gate_factory.make_arithmetic_32_output_gate_my(output_owner, input);
          save_future(uint32_output_futures_, output_label, 1, std::move(future));
        } break;
        case 64: {
          auto future = gate_factory.make_arithmetic_64_output_gate_my(output_owner, input);
          save_future(uint64_output_futures_, output_label, 1, std::move(future));
        } break;
        default:
          throw hycc_error(
              fmt::format("unexpected width for arithmetic protocol: {}", gate->get_width()));
      }
    } else {
      switch (gate->get_width()) {
        case 8:
        case 16:
        case 32:
        case 64:
          gate_factory.make_arithmetic_output_gate_other(output_owner, input);
          break;
        default:
          throw hycc_error(
              fmt::format("unexpected width for arithmetic protocol: {}", gate->get_width()));
      }
    }
  } else {
    throw hycc_error(fmt::format("unexpected protocol {}", protocol));
  }
  // std::cerr << " done\n";
}

const WireVector& HyCCAdapter::HyCCAdapterImpl::get_or_create_wires(
    const simple_circuitt::gatet::wire_endpointt& key, MPCProtocol protocol) {
  if (protocol == arithmetic_protocol_) {
    auto it = arithmetic_wires_.find(key);
    if (it != arithmetic_wires_.end()) {
      return it->second;
    }
    throw hycc_error("cannot find arithmetic wires for key");
  }
  if (protocol == boolean_protocol_) {
    auto it = boolean_wires_.find(key);
    if (it != boolean_wires_.end()) {
      return it->second;
    } else {
      // convert Y2B
      auto it = yao_wires_.find(key);
      if (it == yao_wires_.end()) {
        throw hycc_error("cannot find boolean or yao wires for key");
      }
      auto wires = circuit_builder_.convert(boolean_protocol_, it->second);
      return store_wires(key, boolean_protocol_, std::move(wires));
    }
  }
  if (protocol == yao_protocol_) {
    auto it = yao_wires_.find(key);
    if (it != yao_wires_.end()) {
      return it->second;
    } else {
      // convert B2Y
      auto it = boolean_wires_.find(key);
      if (it == boolean_wires_.end()) {
        throw hycc_error("cannot find boolean or yao wires for key");
      }
      auto wires = circuit_builder_.convert(yao_protocol_, it->second);
      return store_wires(key, yao_protocol_, std::move(wires));
    }
  }
  throw hycc_error(fmt::format("unexpected protocol: {}", ToString(protocol)));
}

const WireVector& HyCCAdapter::HyCCAdapterImpl::store_wires(
    const simple_circuitt::gatet::wire_endpointt& key, MPCProtocol protocol, WireVector&& wires) {
  if (protocol == arithmetic_protocol_) {
    auto [it, inserted] = arithmetic_wires_.emplace(key, std::move(wires));
    if (inserted) {
      return it->second;
    }
  } else if (protocol == boolean_protocol_) {
    auto [it, inserted] = boolean_wires_.emplace(key, std::move(wires));
    if (inserted) {
      return it->second;
    }
  } else if (protocol == yao_protocol_) {
    auto [it, inserted] = yao_wires_.emplace(key, std::move(wires));
    if (inserted) {
      return it->second;
    }
  } else {
    throw hycc_error(fmt::format("unexpected protocol: {}", ToString(protocol)));
  }
  throw hycc_error("there are already wires with this key");
}

void HyCCAdapter::HyCCAdapterImpl::visit_output_gate(simple_circuitt::gatet* gate) {
  assert(gate->get_operation() == simple_circuitt::OUTPUT);
}

void HyCCAdapter::HyCCAdapterImpl::visit_const_gate(simple_circuitt::gatet* gate) {
  const auto protocol = get_protocol_from_gate(*gate);
  assert(protocol == arithmetic_protocol_);
  const auto width = gate->get_width();
  WireVector wires;
  const auto make_arithmetic_const_wire = [this, gate](auto dummy) -> WireVector {
    using T = decltype(dummy);
    auto data = std::vector(num_simd_, static_cast<T>(gate->get_value()));
    return {std::make_shared<proto::plain::ArithmeticPlainWire<T>>(std::move(data))};
  };
  switch (width) {
    case 8:
      wires = make_arithmetic_const_wire(std::uint8_t{});
      break;
    case 16:
      wires = make_arithmetic_const_wire(std::uint16_t{});
      break;
    case 32:
      wires = make_arithmetic_const_wire(std::uint32_t{});
      break;
    case 64:
      wires = make_arithmetic_const_wire(std::uint64_t{});
      break;
    default:
      throw hycc_error(fmt::format("unexpected width: {}", width));
  }
  store_wires(primary_output(gate), protocol, std::move(wires));
}

void HyCCAdapter::HyCCAdapterImpl::visit_split_gate(simple_circuitt::gatet* gate) {
  if (gate->num_fanins() != 1) {
    throw hycc_error(fmt::format("unexpected number of inputs: {}", gate->num_fanins()));
  }
  const MPCProtocol src_protocol = arithmetic_protocol_;
  const auto& key = gate->fanin_range()[0];
  WireVector input = get_or_create_wires(key, src_protocol);
  MPCProtocol dst_protocol = get_protocol_from_gate(*gate->get_fanouts()[0]->second.gate);
  assert(src_protocol == arithmetic_protocol_);
  assert(dst_protocol == boolean_protocol_ || dst_protocol == yao_protocol_);
  auto output = circuit_builder_.convert(dst_protocol, input);
  for (std::size_t wire_i = 0; wire_i < output.size(); ++wire_i) {
    // store_wires(gate->get_fanouts().at(wire_i)->second, dst_protocol, {output[wire_i]});
    store_wires({gate, static_cast<unsigned int>(wire_i)}, dst_protocol, {output[wire_i]});
  }
}

void HyCCAdapter::HyCCAdapterImpl::visit_combine_gate(simple_circuitt::gatet* gate) {
  const auto num_fanins = gate->num_fanins();
  if (num_fanins != 8 && num_fanins != 16 && num_fanins != 32 && num_fanins != 64) {
    throw hycc_error(fmt::format("unexpected number of inputs: {}", gate->num_fanins()));
  }
  const auto get_src_protocol = [](const auto* gate) -> MPCProtocol {
    return get_protocol_from_gate(*gate->get_fanin(0));
  };
  const auto src_protocol = get_src_protocol(gate);
  WireVector input;
  for (std::size_t wire_i = 0; wire_i < num_fanins; ++wire_i) {
    input.emplace_back(get_or_create_wires(gate->fanin_range().at(wire_i), src_protocol).at(0));
  }
  const MPCProtocol dst_protocol = arithmetic_protocol_;
  assert(src_protocol == boolean_protocol_ || src_protocol == yao_protocol_);
  auto output = circuit_builder_.convert(dst_protocol, input);
  store_wires(primary_output(gate), dst_protocol, std::move(output));
}

void HyCCAdapter::HyCCAdapterImpl::visit_unary_gate(simple_circuitt::gatet* gate,
                                                    simple_circuitt::GATE_OP op) {
  if (gate->num_fanins() != 1) {
    throw hycc_error(fmt::format("unexpected number of inputs: {}", gate->num_fanins()));
  }
  const auto protocol = get_protocol_from_gate(*gate);
  const auto& key = gate->fanin_range()[0];
  WireVector input = get_or_create_wires(key, protocol);
  WireVector output;
  switch (op) {
    case simple_circuitt::NOT:
      output = circuit_builder_.make_unary_gate(ENCRYPTO::PrimitiveOperationType::INV, input);
      break;
    case simple_circuitt::NEG:
      output = circuit_builder_.make_unary_gate(ENCRYPTO::PrimitiveOperationType::NEG, input);
      break;
    default:
      throw std::logic_error(fmt::format("unexpected unary operation {}", op));
  }
  store_wires(primary_output(gate), protocol, std::move(output));
}

void HyCCAdapter::HyCCAdapterImpl::visit_binary_gate(simple_circuitt::gatet* gate,
                                                     simple_circuitt::GATE_OP op) {
  if (gate->num_fanins() != 2) {
    throw hycc_error(fmt::format("unexpected number of inputs: {}", gate->num_fanins()));
  }
  const auto protocol = get_protocol_from_gate(*gate);
  auto& key_a = gate->fanin_range()[0];
  auto& key_b = gate->fanin_range()[1];
  WireVector input_a = get_or_create_wires(key_a, protocol);
  WireVector input_b = get_or_create_wires(key_b, protocol);
  WireVector output;
  switch (op) {
    case simple_circuitt::AND:
      output = circuit_builder_.make_binary_gate(ENCRYPTO::PrimitiveOperationType::AND, input_a,
                                                 input_b);
      break;
    case simple_circuitt::XOR:
      output = circuit_builder_.make_binary_gate(ENCRYPTO::PrimitiveOperationType::XOR, input_a,
                                                 input_b);
      break;
    case simple_circuitt::OR:
      output =
          circuit_builder_.make_binary_gate(ENCRYPTO::PrimitiveOperationType::OR, input_a, input_b);
      break;
    case simple_circuitt::ADD:
      output = circuit_builder_.make_binary_gate(ENCRYPTO::PrimitiveOperationType::ADD, input_a,
                                                 input_b);
      break;
    case simple_circuitt::SUB: {
      auto tmp = circuit_builder_.make_unary_gate(ENCRYPTO::PrimitiveOperationType::NEG, input_b);
      output =
          circuit_builder_.make_binary_gate(ENCRYPTO::PrimitiveOperationType::ADD, input_a, tmp);
      break;
    }
    case simple_circuitt::MUL:
      output = circuit_builder_.make_binary_gate(ENCRYPTO::PrimitiveOperationType::MUL, input_a,
                                                 input_b);
      break;
    default:
      throw std::logic_error(fmt::format("unexpected unary operation {}", op));
  }
  store_wires(primary_output(gate), protocol, std::move(output));
}

void HyCCAdapter::HyCCAdapterImpl::topological_traversal(simple_circuitt& main_circuit) {
  const auto visitor = [this](simple_circuitt::gatet* gate) {
    auto gate_op = gate->get_operation();
    switch (gate_op) {
      case simple_circuitt::NOT:
      case simple_circuitt::NEG:
        visit_unary_gate(gate, gate_op);
        break;
      case simple_circuitt::AND:
      case simple_circuitt::OR:
      case simple_circuitt::XOR:
      case simple_circuitt::ADD:
      case simple_circuitt::SUB:
      case simple_circuitt::MUL:
        visit_binary_gate(gate, gate_op);
        break;
      case simple_circuitt::COMBINE:
        visit_combine_gate(gate);
        break;
      case simple_circuitt::SPLIT:
        visit_split_gate(gate);
        break;
      case simple_circuitt::ONE:
        throw hycc_error("HyCC operation ONE is not supported");
        break;
      case simple_circuitt::CONST:
        visit_const_gate(gate);
        break;
      case simple_circuitt::INPUT:
      case simple_circuitt::OUTPUT:
        // inputs and outputs are handled separately
        break;
      case simple_circuitt::LUT:
        throw hycc_error("HyCC operation LUT is not supported");
    }
  };
  std::cerr << "run topological_traversal ...";
  main_circuit.topological_traversal(visitor);
  std::cerr << " done\n";
}

HyCCAdapter::HyCCAdapter(std::size_t my_id, CircuitBuilder& circuit_builder,
                         MPCProtocol arithmetic_protocol, MPCProtocol boolean_protocol,
                         MPCProtocol yao_protocol, std::size_t num_simd,
                         std::shared_ptr<Logger> logger)
    : impl_(std::make_unique<HyCCAdapterImpl>(circuit_builder)) {
  impl_->arithmetic_protocol_ = arithmetic_protocol;
  impl_->boolean_protocol_ = boolean_protocol;
  impl_->yao_protocol_ = yao_protocol;
  impl_->default_boolean_protocol_ = yao_protocol;
  impl_->num_simd_ = num_simd;
  impl_->my_id_ = my_id;
  impl_->logger_ = logger;
}

HyCCAdapter::~HyCCAdapter() = default;

void HyCCAdapter::load_circuit(const std::string& file_path) {
  std::cerr << fmt::format("loading HyCC circuit '{}'\n", file_path);
  std::ifstream file(file_path);
  if (!file.is_open() || !file.good()) {
    throw hycc_error(fmt::format("could not open file: '{}'", file_path));
  }
  std::string line;
  while (std::getline(file, line)) {
    impl_->circuit_files_.push_back(line);
  }
  std::filesystem::path path(file_path);
  impl_->circuit_directory_ = path.parent_path();

  auto& main_circuit = impl_->load_circuit_files();
  impl_->process_circuit_inputs(main_circuit);
  impl_->topological_traversal(main_circuit);
  impl_->process_circuit_outputs(main_circuit);
}

std::size_t HyCCAdapter::get_num_simd() const noexcept { return impl_->num_simd_; }

std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>>>&
get_input_bit_promises() noexcept;
std::unordered_map<
    std::string,
    std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>>>>&
HyCCAdapter::get_input_bit_promises() noexcept {
  return impl_->bit_input_promises_;
}

std::unordered_map<
    std::string, std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint8_t>>>>&
HyCCAdapter::get_input_int8_promises() noexcept {
  return impl_->uint8_input_promises_;
}

std::unordered_map<
    std::string,
    std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint16_t>>>>&
HyCCAdapter::get_input_int16_promises() noexcept {
  return impl_->uint16_input_promises_;
}

std::unordered_map<
    std::string,
    std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint32_t>>>>&
HyCCAdapter::get_input_int32_promises() noexcept {
  return impl_->uint32_input_promises_;
}

std::unordered_map<
    std::string,
    std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint64_t>>>>&
HyCCAdapter::get_input_int64_promises() noexcept {
  return impl_->uint64_input_promises_;
}

std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>>&
get_output_bit_futures() noexcept;
std::unordered_map<
    std::string,
    std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>>>&
HyCCAdapter::get_output_bit_futures() noexcept {
  return impl_->bit_output_futures_;
}

std::unordered_map<
    std::string, std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint8_t>>>>&
HyCCAdapter::get_output_int8_futures() noexcept {
  return impl_->uint8_output_futures_;
}

std::unordered_map<
    std::string, std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint16_t>>>>&
HyCCAdapter::get_output_int16_futures() noexcept {
  return impl_->uint16_output_futures_;
}

std::unordered_map<
    std::string, std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint32_t>>>>&
HyCCAdapter::get_output_int32_futures() noexcept {
  return impl_->uint32_output_futures_;
}

std::unordered_map<
    std::string, std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>>>>&
HyCCAdapter::get_output_int64_futures() noexcept {
  return impl_->uint64_output_futures_;
}

}  // namespace MOTION::hycc
