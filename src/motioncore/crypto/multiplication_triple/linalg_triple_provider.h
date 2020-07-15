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

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensor/tensor_op.h"
#include "utility/enable_wait.h"
#include "utility/type_traits.hpp"

namespace MOTION {

class ArithmeticProvider;
template <typename T>
class ConvolutionInputSide;
template <typename T>
class ConvolutionKernelSide;
template <typename T>
class MatrixMultiplicationLHS;
template <typename T>
class MatrixMultiplicationRHS;
class Logger;

class LinAlgTripleProvider : public ENCRYPTO::enable_wait_setup {
 public:
  virtual ~LinAlgTripleProvider() = default;
  template <typename T>
  struct LinAlgTriple {
    std::vector<T> a_;
    std::vector<T> b_;
    std::vector<T> c_;
    using is_enabled_ = ENCRYPTO::is_unsigned_int_t<T>;
  };

  template <typename T>
  std::size_t register_for_gemm_triple(const tensor::GemmOp&);

  template <typename T>
  [[nodiscard]] LinAlgTriple<T> get_gemm_triple(const tensor::GemmOp&, std::size_t);

  template <typename T>
  std::size_t register_for_conv2d_triple(const tensor::Conv2DOp&);

  template <typename T>
  [[nodiscard]] LinAlgTriple<T> get_conv2d_triple(const tensor::Conv2DOp&, std::size_t);

  virtual void setup() = 0;

 protected:
  virtual void registration_hook(const tensor::GemmOp&, std::size_t bit_size) = 0;
  virtual void registration_hook(const tensor::Conv2DOp&, std::size_t bit_size) = 0;

  std::unordered_map<tensor::GemmOp, std::size_t> gemm_counts_8_;
  std::unordered_map<tensor::GemmOp, std::size_t> gemm_counts_16_;
  std::unordered_map<tensor::GemmOp, std::size_t> gemm_counts_32_;
  std::unordered_map<tensor::GemmOp, std::size_t> gemm_counts_64_;
  std::unordered_map<tensor::GemmOp, std::size_t> gemm_counts_128_;

  std::unordered_map<tensor::GemmOp, std::vector<LinAlgTriple<std::uint8_t>>> gemm_triples_8_;
  std::unordered_map<tensor::GemmOp, std::vector<LinAlgTriple<std::uint16_t>>> gemm_triples_16_;
  std::unordered_map<tensor::GemmOp, std::vector<LinAlgTriple<std::uint32_t>>> gemm_triples_32_;
  std::unordered_map<tensor::GemmOp, std::vector<LinAlgTriple<std::uint64_t>>> gemm_triples_64_;
  std::unordered_map<tensor::GemmOp, std::vector<LinAlgTriple<__uint128_t>>> gemm_triples_128_;

  std::unordered_map<tensor::Conv2DOp, std::size_t> conv2d_counts_8_;
  std::unordered_map<tensor::Conv2DOp, std::size_t> conv2d_counts_16_;
  std::unordered_map<tensor::Conv2DOp, std::size_t> conv2d_counts_32_;
  std::unordered_map<tensor::Conv2DOp, std::size_t> conv2d_counts_64_;
  std::unordered_map<tensor::Conv2DOp, std::size_t> conv2d_counts_128_;

  std::unordered_map<tensor::Conv2DOp, std::vector<LinAlgTriple<std::uint8_t>>> conv2d_triples_8_;
  std::unordered_map<tensor::Conv2DOp, std::vector<LinAlgTriple<std::uint16_t>>> conv2d_triples_16_;
  std::unordered_map<tensor::Conv2DOp, std::vector<LinAlgTriple<std::uint32_t>>> conv2d_triples_32_;
  std::unordered_map<tensor::Conv2DOp, std::vector<LinAlgTriple<std::uint64_t>>> conv2d_triples_64_;
  std::unordered_map<tensor::Conv2DOp, std::vector<LinAlgTriple<__uint128_t>>> conv2d_triples_128_;
};

class LinAlgTriplesFromAP : public LinAlgTripleProvider {
 public:
  LinAlgTriplesFromAP(ArithmeticProvider&, std::shared_ptr<Logger>);
  ~LinAlgTriplesFromAP();

  void setup() override;

 protected:
  void registration_hook(const tensor::GemmOp&, std::size_t bit_size) override;
  void registration_hook(const tensor::Conv2DOp&, std::size_t bit_size) override;

 private:
  ArithmeticProvider& arith_provider_;
  std::shared_ptr<Logger> logger_;

  template <typename T>
  using gemm_value_type = std::vector<std::pair<std::unique_ptr<MatrixMultiplicationLHS<T>>,
                                                std::unique_ptr<MatrixMultiplicationRHS<T>>>>;
  std::unordered_map<tensor::GemmOp, gemm_value_type<std::uint8_t>> gemm_handles_8_;
  std::unordered_map<tensor::GemmOp, gemm_value_type<std::uint16_t>> gemm_handles_16_;
  std::unordered_map<tensor::GemmOp, gemm_value_type<std::uint32_t>> gemm_handles_32_;
  std::unordered_map<tensor::GemmOp, gemm_value_type<std::uint64_t>> gemm_handles_64_;
  std::unordered_map<tensor::GemmOp, gemm_value_type<__uint128_t>> gemm_handles_128_;

  template <typename T>
  using conv_value_type = std::vector<std::pair<std::unique_ptr<ConvolutionInputSide<T>>,
                                                std::unique_ptr<ConvolutionKernelSide<T>>>>;
  std::unordered_map<tensor::Conv2DOp, conv_value_type<std::uint8_t>> conv2d_handles_8_;
  std::unordered_map<tensor::Conv2DOp, conv_value_type<std::uint16_t>> conv2d_handles_16_;
  std::unordered_map<tensor::Conv2DOp, conv_value_type<std::uint32_t>> conv2d_handles_32_;
  std::unordered_map<tensor::Conv2DOp, conv_value_type<std::uint64_t>> conv2d_handles_64_;
  std::unordered_map<tensor::Conv2DOp, conv_value_type<__uint128_t>> conv2d_handles_128_;
};

// Generator of fake triples which just consists of random data.
class FakeLinAlgTripleProvider : public LinAlgTripleProvider {
 public:
  void setup() override;

 protected:
  void registration_hook(const tensor::GemmOp&, std::size_t bit_size) override;
  void registration_hook(const tensor::Conv2DOp&, std::size_t bit_size) override;
};

}  // namespace MOTION
