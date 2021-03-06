// MIT License
//
// Copyright (c) 2019 Oleksandr Tkachenko, Lennart Braun
// Cryptography and Privacy Engineering Group (ENCRYPTO)
// TU Darmstadt, Germany
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

#include <boost/fiber/future.hpp>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <utility>
#include "utility/bit_vector.h"
#include "utility/block.h"
#include "utility/reusable_future.h"

namespace ENCRYPTO {
class Condition;
}

namespace MOTION {

enum BMRDataType : uint { input_step_0 = 0, input_step_1 = 1, and_gate = 2 };

struct BMRData {
  void MessageReceived(const std::uint8_t* message, const BMRDataType type, const std::size_t i);
  void Reset();

  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> RegisterForInputPublicValues(
      std::size_t gate_id, std::size_t bitlen);
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector> RegisterForInputPublicKeys(
      std::size_t gate_id, std::size_t num_blocks);
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector> RegisterForGarbledRows(
      std::size_t gate_id, std::size_t num_blocks);

  // gate_id -> bit size X promise with public values
  using in_pub_val_t =
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>>;
  std::unordered_map<std::size_t, in_pub_val_t> input_public_value_promises_;

  // gate_id -> block size X promise with keys
  using keys_t = std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::block128_vector>>;
  std::unordered_map<std::size_t, keys_t> input_public_key_promises_;

  // gate_id -> block size X promise with partial garbled rows
  using g_rows_t =
      std::pair<std::size_t, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::block128_vector>>;
  std::unordered_map<std::size_t, g_rows_t> garbled_rows_promises_;
};

}  // namespace MOTION
