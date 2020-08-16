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

#include <cstddef>

#include "utility/bit_vector.h"
#include "utility/block.h"

namespace MOTION::tensor {
struct MaxPoolOp;
}

namespace MOTION::proto::yao {

void get_lsbs_from_keys(ENCRYPTO::BitVector<>& dst, const ENCRYPTO::block128_t* src,
                        std::size_t num_bits);

void transpose_keys(ENCRYPTO::block128_vector& dst, const ENCRYPTO::block128_vector& src,
                    std::size_t bit_size, std::size_t num);

void maxpool_rearrange_keys_in(ENCRYPTO::block128_vector& dst,
                               const ENCRYPTO::block128_vector& keys, std::size_t bit_size,
                               const tensor::MaxPoolOp&);

void maxpool_rearrange_keys_out(ENCRYPTO::block128_vector& dst,
                                const ENCRYPTO::block128_vector& keys, std::size_t bit_size,
                                const tensor::MaxPoolOp&);

}  // namespace MOTION::proto::yao
