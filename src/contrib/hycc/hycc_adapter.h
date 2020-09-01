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

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "utility/bit_vector.h"
#include "utility/reusable_future.h"
#include "utility/typedefs.h"

namespace MOTION {

class CircuitBuilder;
class Logger;

namespace hycc {

class HyCCAdapter {
 public:
  class hycc_error : std::runtime_error {
    using std::runtime_error::runtime_error;
  };

  HyCCAdapter(std::size_t my_id, CircuitBuilder& circuit_builder, MPCProtocol arithmetic_protocol,
              MPCProtocol boolean_protocol, MPCProtocol yao_protocol, std::size_t num_simd,
              std::shared_ptr<Logger> logger);
  ~HyCCAdapter();

  void load_circuit(const std::string& path);
  void clear_hycc_data();

  std::size_t get_num_simd() const noexcept;

  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<ENCRYPTO::BitVector<>>>>>&
  get_input_bit_promises() noexcept;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint8_t>>>>&
  get_input_int8_promises() noexcept;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint16_t>>>>&
  get_input_int16_promises() noexcept;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint32_t>>>>&
  get_input_int32_promises() noexcept;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint64_t>>>>&
  get_input_int64_promises() noexcept;

  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<ENCRYPTO::BitVector<>>>>>&
  get_output_bit_futures() noexcept;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint8_t>>>>&
  get_output_int8_futures() noexcept;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint16_t>>>>&
  get_output_int16_futures() noexcept;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint32_t>>>>&
  get_output_int32_futures() noexcept;
  std::unordered_map<
      std::string,
      std::pair<std::size_t, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>>>>&
  get_output_int64_futures() noexcept;

 private:
  struct HyCCAdapterImpl;
  std::unique_ptr<HyCCAdapterImpl> impl_;
};

}  // namespace hycc
}  // namespace MOTION
