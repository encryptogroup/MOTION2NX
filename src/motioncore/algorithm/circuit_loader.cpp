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

#include <exception>
#include <filesystem>
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

}  // namespace MOTION
