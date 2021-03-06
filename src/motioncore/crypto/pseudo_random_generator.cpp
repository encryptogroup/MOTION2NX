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

#include "pseudo_random_generator.h"
#include <cstdint>

#include "aes/aesni_primitives.h"

namespace ENCRYPTO {

void PRG::SetKey(const std::uint8_t *key) {
  offset_ = 0;
  std::copy(key, key + AES_BLOCK_SIZE, reinterpret_cast<std::uint8_t *>(key_.data()));
  if (1 != EVP_EncryptInit_ex(ctx_.get(), EVP_aes_128_ecb(), NULL,
                              reinterpret_cast<const unsigned char *>(key_.data()), nullptr)) {
    throw(std::runtime_error(fmt::format("Could not re-initialize EVP context")));
  }
  std::copy(key, key + aes_key_size_128, reinterpret_cast<std::uint8_t *>(round_keys_.data()));
  aesni_key_expansion_128(round_keys_.data());
  contains_key_ = true;
}

void PRG::SetKey(const std::byte *key) { SetKey(reinterpret_cast<const std::uint8_t *>(key)); }

std::vector<std::byte> PRG::Encrypt(const std::size_t bytes) {
  const uint remainder = (bytes & 15u) > 0 ? 1 : 0;
  const std::size_t num_blocks = (bytes / 16) + remainder + 1;
  int len = bytes;
  const std::size_t bytelen = num_blocks * AES_BLOCK_SIZE;

  std::vector<std::byte> output(bytelen), input(bytelen, std::byte(0));

  for (auto i = 0ull; i < input.size() / AES_BLOCK_SIZE; ++i) {
    *reinterpret_cast<uint64_t *>(input.data() + i * AES_BLOCK_SIZE) = i + offset_;
  }

  if (1 != EVP_EncryptUpdate(ctx_.get(), reinterpret_cast<std::uint8_t *>(output.data()), &len,
                             reinterpret_cast<std::uint8_t *>(input.data()),
                             num_blocks * AES_BLOCK_SIZE)) {
    throw(std::runtime_error(fmt::format("Could not EVP_EncryptUpdate")));
  }

  return output;
}

std::vector<std::byte> PRG::Encrypt(const std::byte *input, const std::size_t bytes) {
  const uint remainder = (bytes & 15u) > 0 ? 1 : 0;
  const std::size_t num_blocks = (bytes / 16) + remainder;
  int len = bytes;
  const std::size_t bytelen = num_blocks * AES_BLOCK_SIZE;
  std::vector<std::byte> output(bytelen);

  if (1 != EVP_EncryptUpdate(ctx_.get(), reinterpret_cast<std::uint8_t *>(output.data()), &len,
                             reinterpret_cast<const std::uint8_t *>(input),
                             num_blocks * AES_BLOCK_SIZE)) {
    throw(std::runtime_error(fmt::format("Could not EVP_EncryptUpdate")));
  }

  return output;
}

std::vector<std::byte> PRG::FixedKeyAES(const std::byte *x, const std::uint64_t i,
                                        const std::size_t num) {
  auto aes_x = Encrypt(x, num * AES_BLOCK_SIZE);

  std::size_t j;

  for (j = 0; j < num; ++j) {
    reinterpret_cast<uint64_t *>(aes_x.data())[2 * j] ^= i + j;
  }

  auto output = Encrypt(aes_x.data(), num * AES_BLOCK_SIZE);

  for (j = 0; j < num; ++j) {
    reinterpret_cast<uint128_t *>(output.data())[j] ^=
        reinterpret_cast<const uint128_t *>(aes_x.data())[j];
    reinterpret_cast<uint64_t *>(output.data())[2 * j] ^= i + j;
  }

  return output;
}

std::vector<std::byte> PRG::FixedKeyAES(const std::byte *x, const uint128_t i) {
  auto aes_x = Encrypt(x, AES_BLOCK_SIZE);

  *reinterpret_cast<uint128_t *>(aes_x.data()) ^= i;

  auto output = Encrypt(aes_x.data(), AES_BLOCK_SIZE);

  *reinterpret_cast<uint128_t *>(output.data()) ^=
      *reinterpret_cast<const uint128_t *>(aes_x.data()) ^ i;

  return output;
}

static void encrypt_block(EVP_CIPHER_CTX *ctx, const std::byte *in, std::byte *out) {
  int outl;
  if (1 != EVP_EncryptUpdate(ctx, reinterpret_cast<std::uint8_t *>(out), &outl,
                             reinterpret_cast<const std::uint8_t *>(in), AES_BLOCK_SIZE)) {
    throw(std::runtime_error(fmt::format("Could not EVP_EncryptUpdate")));
  }
}

void PRG::FixedKeyAES(const std::byte *input, const uint128_t tweak, std::byte *output) {
  // TODO: enforce buffer alignment, do byte-wise xor (-> better compiler optimization)
  std::array<std::byte, AES_BLOCK_SIZE> tmp1;
  std::array<std::byte, AES_BLOCK_SIZE> tmp2;
  encrypt_block(ctx_.get(), input, tmp1.data());         // compute \pi(x) ...
  std::copy(std::begin(tmp1), std::end(tmp1), output);   // ... and save it in the output
  *reinterpret_cast<uint128_t *>(tmp1.data()) ^= tweak;  // compute \pi(x) ^ i
  encrypt_block(ctx_.get(), tmp1.data(), tmp2.data());   // compute \pi(\pi(x) ^ i)
  // compute \pi(\pi(x) ^ i) ^ \pi(x):
  *reinterpret_cast<uint128_t *>(output) ^= *reinterpret_cast<const uint128_t *>(tmp2.data());
}

void PRG::MMO(std::byte *input) { aesni_mmo_single(round_keys_.data(), input); }

}  // namespace ENCRYPTO
