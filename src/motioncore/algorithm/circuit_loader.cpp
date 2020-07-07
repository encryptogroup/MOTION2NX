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

#include "circuit_loader.h"

#include <algorithm>
#include <exception>
#include <filesystem>
#include <optional>
#include <queue>
#include <stdexcept>

#include <fmt/format.h>

#include "algorithm_description.h"
#include "utility/config.h"

namespace fs = std::filesystem;

namespace MOTION {

CircuitLoader::CircuitLoader() {
  fs::path circuit_dir = MOTION::MOTION_ROOT_DIR;
  circuit_dir /= "circuits";
  for (const auto& entry : fs::directory_iterator(circuit_dir)) {
    if (entry.is_directory()) {
      circuit_search_path_.emplace_back(entry);
    }
  }
  circuit_search_path_.emplace_back(".");
}

CircuitLoader::~CircuitLoader() = default;

static std::string format_search_path(std::vector<fs::path> paths) {
  if (paths.empty()) {
    return "";
  }
  std::string s = paths[0].string();
  for (std::size_t i = 1; i < paths.size(); ++i) {
    s += ":" + paths[i].string();
  }
  return s;
}

const ENCRYPTO::AlgorithmDescription& CircuitLoader::load_circuit(std::string name,
                                                                  CircuitFormat format) {
  auto it = algo_cache_.find(name);
  if (it != std::end(algo_cache_)) {
    return it->second;
  }

  for (const auto& dir : circuit_search_path_) {
    auto path = dir / name;
    fs::directory_entry dir_entry(path);
    if (dir_entry.exists() && dir_entry.is_regular_file()) {
      try {
        switch (format) {
          case CircuitFormat::ABY:
            algo_cache_[name] = ENCRYPTO::AlgorithmDescription::FromABY(dir_entry.path());
            break;
          case CircuitFormat::Bristol:
            algo_cache_[name] = ENCRYPTO::AlgorithmDescription::FromBristol(dir_entry.path());
            break;
          case CircuitFormat::BristolFashion:
            algo_cache_[name] =
                ENCRYPTO::AlgorithmDescription::FromBristolFashion(dir_entry.path());
            break;
        }
        return algo_cache_[name];
      } catch (std::runtime_error& e) {
        throw std::runtime_error(
            fmt::format("Could not load circuit description '{}' from file {}: '{}'", name,
                        dir_entry.path().string()));
      }
    }
  }
  throw std::runtime_error(fmt::format("Could not find circuit description '{}' in search path: {}",
                                       name, format_search_path(circuit_search_path_)));
}

const ENCRYPTO::AlgorithmDescription& CircuitLoader::load_relu_circuit(std::size_t bit_size) {
  const auto name = fmt::format("__circuit_loader_builtin__relu_{}_bit", bit_size);
  auto it = algo_cache_.find(name);
  if (it != std::end(algo_cache_)) {
    return it->second;
  }

  std::vector<ENCRYPTO::PrimitiveOperation> gates(bit_size + 1);

  // hack s.t. the last input becomes the last output
  // neccessary since this format does not support simple passthrough
  gates[0] = ENCRYPTO::PrimitiveOperation{.type_ = ENCRYPTO::PrimitiveOperationType::INV,
                                          .parent_a_ = bit_size - 1,
                                          .output_wire_ = bit_size};
  gates[bit_size] = ENCRYPTO::PrimitiveOperation{.type_ = ENCRYPTO::PrimitiveOperationType::INV,
                                                 .parent_a_ = bit_size,
                                                 .output_wire_ = 2 * bit_size};

  for (std::size_t i = 0; i < bit_size - 1; ++i) {
    gates[i + 1] = ENCRYPTO::PrimitiveOperation{.type_ = ENCRYPTO::PrimitiveOperationType::AND,
                                                .parent_a_ = i,
                                                .parent_b_ = bit_size - 1,
                                                .output_wire_ = bit_size + i + 1};
  }

  ENCRYPTO::AlgorithmDescription algo{.n_output_wires_ = bit_size,
                                      .n_input_wires_parent_a_ = bit_size,
                                      .n_wires_ = 2 * bit_size + 1,
                                      .n_gates_ = bit_size + 1,
                                      .gates_ = std::move(gates)};
  algo_cache_[name] = std::move(algo);
  return algo_cache_[name];
}

const ENCRYPTO::AlgorithmDescription& CircuitLoader::load_gt_circuit(std::size_t bit_size) {
  if (bit_size != 8 && bit_size != 16 && bit_size != 32 && bit_size != 64) {
    throw std::logic_error(fmt::format("unsupported bit size: {}", bit_size));
  }
  const auto name = fmt::format("__circuit_loader_builtin__gt_{}_bit", bit_size);
  auto it = algo_cache_.find(name);
  if (it != std::end(algo_cache_)) {
    return it->second;
  }
  auto algo = load_circuit(fmt::format("int_gt{}_size.bristol", bit_size), CircuitFormat::Bristol);

  // remove unnecessary outputs ...
  algo.n_gates_ -= bit_size + 1;
  algo.n_wires_ -= bit_size + 1;
  algo.n_output_wires_ = 1;
  algo.gates_.resize(algo.n_gates_);
  algo.gates_.at(algo.n_gates_ - 1).output_wire_ -= 2;

  return algo_cache_[name] = std::move(algo);
}

const ENCRYPTO::AlgorithmDescription& CircuitLoader::load_gtmux_circuit(std::size_t bit_size) {
  if (bit_size != 8 && bit_size != 16 && bit_size != 32 && bit_size != 64) {
    throw std::logic_error(fmt::format("unsupported bit size: {}", bit_size));
  }
  const auto name = fmt::format("__circuit_loader_builtin__gtmux_{}_bit", bit_size);
  auto it = algo_cache_.find(name);
  if (it != std::end(algo_cache_)) {
    return it->second;
  }
  const auto& gt_algo = load_gt_circuit(bit_size);
  auto algo = gt_algo;

  // X >= Y <-> X - Y >= 0 <-> msb(X - Y) = 0  -> choose X
  // X < Y <-> X - Y < 0 <-> msb(X - Y) = 1  -> choose Y
  auto choice_wire = algo.n_wires_ - 1;

  auto wire_offset = algo.n_wires_;
  auto gate_offset = algo.n_gates_;
  auto& gates = algo.gates_;
  gates.resize(gate_offset + 3 * bit_size);
  algo.n_gates_ += 3 * bit_size;
  algo.n_wires_ += 3 * bit_size;
  algo.n_output_wires_ = bit_size;
  assert(choice_wire == wire_offset - 1);
  for (std::size_t bit_j = 0; bit_j < bit_size; ++bit_j) {
    // X ^ Y
    gates.at(gate_offset + bit_j) =
        ENCRYPTO::PrimitiveOperation{.type_ = ENCRYPTO::PrimitiveOperationType::XOR,
                                     .parent_a_ = bit_j,
                                     .parent_b_ = bit_size + bit_j,
                                     .output_wire_ = wire_offset + bit_j};
    // (X ^ Y) * b
    gates.at(gate_offset + bit_size + bit_j) =
        ENCRYPTO::PrimitiveOperation{.type_ = ENCRYPTO::PrimitiveOperationType::AND,
                                     .parent_a_ = wire_offset + bit_j,
                                     .parent_b_ = choice_wire,
                                     .output_wire_ = wire_offset + bit_size + bit_j};
    // Y ^ (X ^ Y) * b
    gates.at(gate_offset + 2 * bit_size + bit_j) =
        ENCRYPTO::PrimitiveOperation{.type_ = ENCRYPTO::PrimitiveOperationType::XOR,
                                     .parent_a_ = bit_size + bit_j,
                                     .parent_b_ = wire_offset + bit_size + bit_j,
                                     .output_wire_ = wire_offset + 2 * bit_size + bit_j};
  }
  assert(algo.gates_.size() == algo.n_gates_);

  for (std::size_t i = 0; i < algo.n_gates_; ++i) {
    [[maybe_unused]] const auto& op = algo.gates_.at(i);
    assert(op.parent_a_ < algo.n_wires_ - bit_size);
    assert(!op.parent_b_.has_value() || *op.parent_b_ < algo.n_wires_ - bit_size);
    assert(op.output_wire_ < algo.n_wires_);
  }

  algo_cache_[name] = std::move(algo);
  return algo_cache_[name];
}

}  // namespace MOTION
