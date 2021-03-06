// MIT License
//
// Copyright (c) 2019 Oleksandr Tkachenko
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

#include <flatbuffers/flatbuffers.h>
#include <fmt/format.h>
#include <random>

#include "condition.h"
#include "crypto/random/aes128_ctr_rng.h"
#include "typedefs.h"

namespace MOTION::Helpers {

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
std::vector<T> RandomVector(std::size_t length) {
  auto& rng = AES128_CTR_RNG::get_thread_instance();
  const auto byte_size = sizeof(T) * length;
  std::vector<T> vec(length);
  rng.random_bytes(reinterpret_cast<std::byte*>(vec.data()), byte_size);
  return vec;
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
inline std::vector<std::uint8_t> ToByteVector(const std::vector<T> &values) {
  std::vector<std::uint8_t> result(
      reinterpret_cast<const std::uint8_t *>(values.data()),
      reinterpret_cast<const std::uint8_t *>(values.data()) + sizeof(T) * values.size());
  return result;
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
inline std::vector<T> FromByteVector(const std::vector<std::uint8_t> &buffer) {
  assert(buffer.size() % sizeof(T) == 0);  // buffer length is multiple of the element size
  std::vector<T> result(sizeof(T) * buffer.size());
  std::copy(buffer.data(), buffer.data() + buffer.size(),
            reinterpret_cast<std::uint8_t *>(result.data()));
  return result;
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
inline std::vector<T> FromByteVector(const flatbuffers::Vector<std::uint8_t> &buffer) {
  assert(buffer.size() % sizeof(T) == 0);  // buffer length is multiple of the element size
  std::vector<T> result(buffer.size() / sizeof(T));
  std::copy(buffer.data(), buffer.data() + buffer.size(),
            reinterpret_cast<std::uint8_t *>(result.data()));
  return result;
}

template <typename T>
inline std::vector<T> AddVectors(std::vector<std::vector<T>> &vectors) {
  if (vectors.size() == 0) {
    return {};
  }  // if empty input vector

  std::vector<T> result = vectors.at(0);

  for (auto i = 1ull; i < vectors.size(); ++i) {
    auto &v = vectors.at(i);
    assert(v.size() == result.size());  // expect the vectors to be of the same size
    for (auto j = 0ull; j < result.size(); ++j) {
      result.at(j) += v.at(j);  // TODO: implement using AVX2 and AVX512
    }
  }
  return result;
}

template <typename T>
inline std::vector<T> AddVectors(std::vector<std::vector<T>> &&vectors) {
  return AddVectors(vectors);
}

template <typename T>
inline std::vector<T> AddVectors(const std::vector<T> &a, const std::vector<T> &b) {
  assert(a.size() == b.size());
  if (a.size() == 0) {
    return {};
  }  // if empty input vector
  std::vector<T> result = a;
#pragma omp simd
  for (auto j = 0ull; j < result.size(); ++j) {
    result.at(j) += b.at(j);  // TODO: implement using AVX2 and AVX512
  }
  return result;
}

template <typename T>
inline std::vector<T> RestrictAddVectors(const std::vector<T> &a, const std::vector<T> &b) {
  assert(a.size() == b.size());
  if (a.size() == 0) {
    return {};
  }  // if empty input vector
  std::vector<T> result(a.size());
  const T *__restrict__ a_ptr{a.data()};
  const T *__restrict__ b_ptr{b.data()};
  T *__restrict__ r_ptr{result.data()};
  std::transform(a_ptr, a_ptr + a.size(), b_ptr, r_ptr,
                 [](const T &a_var, const T &b_var) { return a_var + b_var; });
  return result;
}

template <typename T>
inline std::vector<T> RestrictMulVectors(const std::vector<T> &a, const std::vector<T> &b) {
  assert(a.size() == b.size());
  if (a.size() == 0) {
    return {};
  }  // if empty input vector
  std::vector<T> result(a.size());
  const T *__restrict__ a_ptr{a.data()};
  const T *__restrict__ b_ptr{b.data()};
  T *__restrict__ r_ptr{result.data()};
  std::transform(a_ptr, a_ptr + a.size(), b_ptr, r_ptr,
                 [](const T &a_var, const T &b_var) { return a_var * b_var; });
  return result;
}

template <typename T>
inline std::vector<T> RestrictSubVectors(const std::vector<T> &a, const std::vector<T> &b) {
  assert(a.size() == b.size());
  if (a.size() == 0) {
    return {};
  }  // if empty input vector
  std::vector<T> result(a.size());
  const T *__restrict__ a_ptr{a.data()};
  const T *__restrict__ b_ptr{b.data()};
  T *__restrict__ r_ptr{result.data()};
  std::transform(a_ptr, a_ptr + a.size(), b_ptr, r_ptr,
                 [](const T &a_var, const T &b_var) { return a_var - b_var; });
  return result;
}

template <typename T>
inline std::vector<T> SubVectors(const std::vector<T> &a, const std::vector<T> &b) {
  assert(a.size() == b.size());
  if (a.size() == 0) {
    return {};
  }  // if empty input vector
  std::vector<T> result = a;
  for (auto j = 0ull; j < result.size(); ++j) {
    result.at(j) -= b.at(j);  // TODO: implement using AVX2 and AVX512
  }
  return result;
}

template <typename T>
inline T SumReduction(const std::vector<T> &v) {
  if (v.size() == 0) {
    return 0;
  } else if (v.size() == 1) {
    return v.at(0);
  } else {
    T sum = 0;
#pragma omp parallel for reduction(+ : sum) default(none) shared(v)
    for (auto i = 0ull; i < v.size(); ++i) {
      sum += v.at(i);
    }
    return sum;
  }
}

template <typename T>
inline T SubReduction(const std::vector<T> &v) {
  if (v.size() == 0) {
    return 0;
  } else {
    T result = v.at(0);
    for (auto i = 1ull; i < v.size(); ++i) {
      result -= v.at(i);
    }
    return result;
  }
}

// +---------+--------------------------+
// | sum_0 = | v_00 + v_01 + ... + v_0m |
// |  ...    | ........................ |
// | sum_n = | v_n0 + v_n1 + ... + v_nm |
// +---------+--------------------------+

template <typename T>
inline std::vector<T> RowSumReduction(const std::vector<std::vector<T>> &v) {
  if (v.size() == 0) {
    return {};
  } else {
    std::vector<T> sum(v.at(0).size());
    for (auto i = 1ull; i < v.size(); ++i) {
      assert(v.at(0).size() == v.at(i).size());
    }

    for (auto i = 0ull; i < sum.size(); ++i) {
      for (auto j = 0ull; j < v.size(); ++j) {
        sum.at(i) += v.at(j).at(i);
      }
    }
    return std::move(sum);
  }
}

template <typename T>
inline std::vector<T> RowSubReduction(const std::vector<std::vector<T>> &v) {
  if (v.size() == 0) {
    return {};
  } else {
    std::vector<T> result = v.at(0);
    for (auto i = 1ull; i < v.size(); ++i) {
      assert(v.at(0).size() == v.at(i).size());
    }

    for (auto i = 0ull; i < result.size(); ++i) {
      for (auto j = 1ull; j < v.size(); ++j) {
        result.at(i) -= v.at(j).at(i);
      }
    }
    return std::move(result);
  }
}

template <typename T>
inline std::vector<T> MultiplyVectors(std::vector<T> a, std::vector<T> b) {
  assert(a.size() == b.size());
  if (a.size() == 0) {
    return {};
  }  // if empty input vector
  std::vector<T> result = a;

  for (auto j = 0ull; j < result.size(); ++j) {
    result.at(j) *= b.at(j);  // TODO: implement using AVX2 and AVX512
  }
  return result;
}

template <typename T>
inline std::vector<T> RowMulReduction(const std::vector<std::vector<T>> &v) {
  if (v.size() == 0) {
    return {};
  } else {
    std::vector<T> product(v.at(0).size(), 1);
    for (auto i = 1ull; i < v.size(); ++i) {
      assert(v.at(0).size() == v.at(i).size());
    }

    for (auto i = 0ull; i < product.size(); ++i) {
      for (auto j = 0ull; j < v.size(); ++j) {
        product.at(i) *= v.at(j).at(i);
      }
    }
    return std::move(product);
  }
}

template <typename T>
inline T RowMulReduction(const std::vector<T> &v) {
  if (v.size() == 0) {
    return 0;
  } else {
    T product = v.at(0);
    for (auto i = 1ull; i < v.size(); ++i) {
      product *= v.at(i);
    }
    return product;
  }
}

template <typename T, typename = std::enable_if_t<std::is_unsigned_v<T>>>
bool IsPowerOfTwo(T x) {
  return x > 0 && (!(x & (x - 1)));
}

namespace Print {
inline std::string Hex(const std::uint8_t *v, std::size_t N) {
  std::string buffer;
  for (auto i = 0ull; i < N; ++i) {
    buffer.append(fmt::format("{0:#x} ", v[i]));
  }
  buffer.erase(buffer.end() - 1);  // remove the last whitespace
  return buffer;
}

inline std::string Hex(const std::byte *v, std::size_t N) {
  return Hex(reinterpret_cast<const std::uint8_t *>(v), N);
}

template <std::size_t N>
inline std::string Hex(const std::array<std::byte, N> &v) {
  return Hex(reinterpret_cast<const std::uint8_t *>(v.data()), v.size());
}

template <std::size_t N>
inline std::string Hex(const std::array<std::uint8_t, N> &v) {
  return Hex(v.data(), v.size());
}

inline std::string Hex(const std::vector<std::uint8_t> &v) { return Hex(v.data(), v.size()); }

inline std::string Hex(const std::vector<std::byte> &v) { return Hex(v.data(), v.size()); }

inline std::string Hex(const std::vector<std::uint8_t> &&v) { return Hex(v); }

std::string ToString(MPCProtocol p);

template <typename T>
inline std::string ToString(std::vector<T> vector) {
  std::string result;
  for (auto &v : vector) {
    result.append(std::to_string(v) + " ");
  }
  return result;
}
}  // namespace Print

namespace Compare {
template <typename T>
inline bool Vectors(const std::vector<T> &a, const std::vector<T> &b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (auto i = 0ull; i < a.size(); ++i) {
    if (a.at(i) != b.at(i)) {
      return false;
    }
  }
  return true;
}

template <typename T>
inline bool Dimensions(const std::vector<std::vector<T>> &v) {
  if (v.size() <= 1) {
    return true;
  } else {
    auto first_size = v.at(0).size();
    for (auto i = 1ull; i < v.size(); ++i) {
      if (first_size != v.at(i).size()) {
        return false;
      }
    }
  }
  return true;
}
}  // namespace Compare

std::size_t DivideAndCeil(std::size_t dividend, std::size_t divisor);

namespace Convert {

inline std::size_t BitsToBytes(const std::size_t bits) { return (bits + 7) / 8; }

}  // namespace Convert
}  // namespace MOTION::Helpers
