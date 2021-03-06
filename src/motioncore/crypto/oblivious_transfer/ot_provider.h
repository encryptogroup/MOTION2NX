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

#include <array>
#include <atomic>
#include <memory>
#include <unordered_map>

#include <flatbuffers/flatbuffers.h>

#include "utility/bit_vector.h"
#include "utility/enable_wait.h"

namespace MOTION {

class BaseOTProvider;
struct BaseOTsData;
struct OTExtensionData;
struct OTExtensionReceiverData;
struct OTExtensionSenderData;
class Logger;

namespace Communication {
class CommunicationLayer;
}

namespace Crypto {
class MotionBaseProvider;
}

namespace Statistics {
struct RunTimeStats;
}

}  // namespace MOTION

namespace ENCRYPTO {

namespace ObliviousTransfer {

enum OTProtocol : uint {
  GOT = 0,   // general OT
  ROT = 1,   // random OT
  XCOT = 2,  // XOR-correlated OT
  ACOT = 3,  // additively-correlated OT
  invalid_OT = 4,
  FixedXCOT128 = 5,
  XCOTBit = 6,
  GOT128 = 7
};

class FixedXCOT128Sender;
class FixedXCOT128Receiver;
class XCOTBitSender;
class XCOTBitReceiver;
template <typename T>
class ACOTSender;
template <typename T>
class ACOTReceiver;
class GOT128Sender;
class GOT128Receiver;
class GOTBitSender;
class GOTBitReceiver;
class ROTSender;
class ROTReceiver;

class OTVector {
 public:
  OTVector() = delete;

  [[nodiscard]] std::size_t GetOtId() const noexcept { return ot_id_; }
  [[nodiscard]] std::size_t GetNumOTs() const noexcept { return num_ots_; }
  [[nodiscard]] std::size_t GetBitlen() const noexcept { return bitlen_; }
  [[nodiscard]] OTProtocol GetProtocol() const noexcept { return p_; }

 protected:
  OTVector(const std::size_t ot_id, const std::size_t num_ots, const std::size_t bitlen,
           const OTProtocol p, const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  const std::size_t ot_id_, num_ots_, bitlen_;
  const OTProtocol p_;

  std::function<void(flatbuffers::FlatBufferBuilder&&)> Send_;
};

class OTVectorSender : public OTVector {
 public:
  [[nodiscard]] const std::vector<BitVector<>>& GetInputs() const { return inputs_; };
  virtual const std::vector<BitVector<>>& GetOutputs();

  virtual void SetInputs(const std::vector<BitVector<>>& v) = 0;
  virtual void SetInputs(std::vector<BitVector<>>&& v) = 0;

  virtual void SendMessages() = 0;

  void WaitSetup();

 protected:
  OTVectorSender(const std::size_t ot_id, const std::size_t num_ots, const std::size_t bitlen,
                 const OTProtocol p, MOTION::OTExtensionSenderData& data,
                 const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  void Reserve(const std::size_t id, const std::size_t num_ots, const std::size_t bitlen);

  MOTION::OTExtensionSenderData& data_;
  std::vector<BitVector<>> inputs_, outputs_;
};

class GOTVectorSender final : public OTVectorSender {
 public:
  GOTVectorSender(const std::size_t ot_id, const std::size_t num_ots, const std::size_t bitlen,
                  MOTION::OTExtensionSenderData& data,
                  const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  void SetInputs(std::vector<BitVector<>>&& v) final;

  void SetInputs(const std::vector<BitVector<>>& v) final;

  // blocking wait for correction bits
  void SendMessages() final;
};

class COTVectorSender final : public OTVectorSender {
 public:
  COTVectorSender(const std::size_t id, const std::size_t num_ots, const std::size_t bitlen,
                  OTProtocol p, MOTION::OTExtensionSenderData& data,
                  const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  void SetInputs(std::vector<BitVector<>>&& v) final;

  void SetInputs(const std::vector<BitVector<>>& v) final;

  const std::vector<BitVector<>>& GetOutputs() final;

  void SendMessages() final;

  void Truncate(std::size_t num_last_bits);

 private:
  std::size_t num_truncated_bits_{0};
};

class ROTVectorSender final : public OTVectorSender {
 public:
  ROTVectorSender(const std::size_t ot_id, const std::size_t num_ots, const std::size_t bitlen,
                  MOTION::OTExtensionSenderData& data,
                  const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  void SetInputs(std::vector<BitVector<>>&& v) final;

  void SetInputs(const std::vector<BitVector<>>& v) final;

  void SendMessages() final;
};

class OTVectorReceiver : public OTVector {
 public:
  virtual void SetChoices(const BitVector<>& v) = 0;

  virtual void SetChoices(BitVector<>&& v) = 0;

  [[nodiscard]] const virtual BitVector<>& GetChoices() = 0;

  [[nodiscard]] const virtual std::vector<BitVector<>>& GetOutputs() = 0;

  virtual void SendCorrections() = 0;

  void WaitSetup();

  bool ChoicesAreSet() { return choices_flag_; }

 protected:
  OTVectorReceiver(const std::size_t ot_id, const std::size_t num_ots, const std::size_t bitlen,
                   const OTProtocol p, MOTION::OTExtensionReceiverData& data,
                   std::function<void(flatbuffers::FlatBufferBuilder&&)> Send);

  void Reserve(const std::size_t id, const std::size_t num_ots, const std::size_t bitlen);

  MOTION::OTExtensionReceiverData& data_;
  BitVector<> choices_;
  std::atomic<bool> choices_flag_{false};
  std::vector<BitVector<>> messages_;
};

class GOTVectorReceiver final : public OTVectorReceiver {
 public:
  GOTVectorReceiver(const std::size_t ot_id, const std::size_t num_ots, const std::size_t bitlen,
                    MOTION::OTExtensionReceiverData& data,
                    const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  void SetChoices(BitVector<>&& v) final;

  void SetChoices(const BitVector<>& v) final;

  const BitVector<>& GetChoices() final { return choices_; }

  void SendCorrections() final;

  const std::vector<BitVector<>>& GetOutputs() final;

 private:
  std::atomic<bool> corrections_sent_ = false;
};

class COTVectorReceiver final : public OTVectorReceiver {
 public:
  COTVectorReceiver(const std::size_t ot_id, const std::size_t num_ots, const std::size_t bitlen,
                    OTProtocol p, MOTION::OTExtensionReceiverData& data,
                    const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  void SendCorrections() final;

  void SetChoices(BitVector<>&& v);

  void SetChoices(const BitVector<>& v);

  const BitVector<>& GetChoices() final { return choices_; }

  const std::vector<BitVector<>>& GetOutputs() final;

 private:
  std::atomic<bool> corrections_sent_ = false;
};

class ROTVectorReceiver final : public OTVectorReceiver {
 public:
  ROTVectorReceiver(const std::size_t ot_id, const std::size_t num_ots, const std::size_t bitlen,
                    MOTION::OTExtensionReceiverData& data,
                    const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  void SetChoices(const BitVector<>& v) final;

  void SetChoices(BitVector<>&& v) final;

  void SendCorrections() final;

  const BitVector<>& GetChoices() final;

  const std::vector<BitVector<>>& GetOutputs() final;
};

class OTProviderSender {
 public:
  OTProviderSender(MOTION::OTExtensionSenderData& data, std::size_t party_id,
                   std::shared_ptr<MOTION::Logger> logger)
      : data_(data), party_id_(party_id), logger_(std::move(logger)) {}

  ~OTProviderSender() = default;

  OTProviderSender(const OTProviderSender&) = delete;

  std::shared_ptr<OTVectorSender>& GetOTs(std::size_t offset);

  std::shared_ptr<OTVectorSender>& RegisterOTs(
      const std::size_t bitlen, const std::size_t num_ots, const OTProtocol p,
      const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<FixedXCOT128Sender> RegisterFixedXCOT128s(
      const std::size_t num_ots, const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<XCOTBitSender> RegisterXCOTBits(
      std::size_t num_ots, std::size_t vector_size,
      const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  template <typename T>
  std::unique_ptr<ACOTSender<T>> RegisterACOT(
      std::size_t num_ots, std::size_t vector_size,
      const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<GOT128Sender> RegisterGOT128(
      const std::size_t num_ots, const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<GOTBitSender> RegisterGOTBit(
      const std::size_t num_ots, const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<ROTSender> RegisterROT(
      std::size_t num_ots, std::size_t vector_size, bool random_choice,
      const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  auto GetNumOTs() const { return total_ots_count_; }

  void Clear();

  void Reset();

 private:
  std::unordered_map<std::size_t, std::shared_ptr<OTVectorSender>> sender_data_;

  std::size_t total_ots_count_{0};

  MOTION::OTExtensionSenderData& data_;

  std::size_t party_id_;

  std::shared_ptr<MOTION::Logger> logger_;
};

class OTProviderReceiver {
 public:
  OTProviderReceiver(MOTION::OTExtensionReceiverData& data, std::size_t party_id,
                     std::shared_ptr<MOTION::Logger> logger)
      : data_(data), party_id_(party_id), logger_(std::move(logger)) {}

  ~OTProviderReceiver() = default;

  OTProviderReceiver(const OTProviderReceiver&) = delete;

  std::shared_ptr<OTVectorReceiver>& GetOTs(const std::size_t offset);

  std::shared_ptr<OTVectorReceiver>& RegisterOTs(
      const std::size_t bitlen, const std::size_t num_ots, const OTProtocol p,
      const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<FixedXCOT128Receiver> RegisterFixedXCOT128s(
      const std::size_t num_ots, const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<XCOTBitReceiver> RegisterXCOTBits(
      std::size_t num_ots, std::size_t vector_size,
      const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  template <typename T>
  std::unique_ptr<ACOTReceiver<T>> RegisterACOT(
      std::size_t num_ots, std::size_t vector_size,
      const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<GOT128Receiver> RegisterGOT128(
      const std::size_t num_ots, const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<GOTBitReceiver> RegisterGOTBit(
      const std::size_t num_ots, const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);
  std::unique_ptr<ROTReceiver> RegisterROT(
      std::size_t num_ots, std::size_t vector_size, bool random_choice,
      const std::function<void(flatbuffers::FlatBufferBuilder&&)>& Send);

  std::size_t GetNumOTs() const { return total_ots_count_; }

  void Clear();
  void Reset();

 private:
  std::unordered_map<std::size_t, std::shared_ptr<OTVectorReceiver>> receiver_data_;

  std::atomic<std::size_t> total_ots_count_{0};

  MOTION::OTExtensionReceiverData& data_;

  std::size_t party_id_;

  std::shared_ptr<MOTION::Logger> logger_;
};

// OTProvider encapsulates both sender and receiver interfaces for simplicity
class OTProvider {
 public:
  virtual ~OTProvider() = default;

  OTProvider(const OTProvider&) = delete;

  /// @param bitlen Bit-length of the messages
  /// @param num_ots Number of OTs
  /// @param p OT protocol from {General OT (GOT), Correlated OT (COT), Random OT (ROT)}
  /// @return Offset to the OT that can be used to set input messages
  [[nodiscard]] std::shared_ptr<OTVectorSender>& RegisterSend(const std::size_t bitlen = 1,
                                                              const std::size_t num_ots = 1,
                                                              const OTProtocol p = GOT) {
    return sender_provider_.RegisterOTs(bitlen, num_ots, p, Send_);
  }

  [[nodiscard]] std::unique_ptr<FixedXCOT128Sender> RegisterSendFixedXCOT128(
      std::size_t num_ots = 1);

  [[nodiscard]] std::unique_ptr<XCOTBitSender> RegisterSendXCOTBit(std::size_t num_ots = 1,
                                                                   std::size_t vector_size = 1);

  template <typename T>
  [[nodiscard]] std::unique_ptr<ACOTSender<T>> RegisterSendACOT(std::size_t num_ots = 1,
                                                                std::size_t vector_size = 1);

  [[nodiscard]] std::unique_ptr<GOT128Sender> RegisterSendGOT128(std::size_t num_ots = 1);

  [[nodiscard]] std::unique_ptr<GOTBitSender> RegisterSendGOTBit(std::size_t num_ots = 1);

  [[nodiscard]] std::unique_ptr<ROTSender> RegisterSendROT(std::size_t num_ots = 1,
                                                           std::size_t vector_size = 1,
                                                           bool random_choice = true);

  /// @param bitlen Bit-length of the messages
  /// @param num_ots Number of OTs
  /// @param p OT protocol from {General OT (GOT), Correlated OT (COT), Random OT (ROT)}
  /// @return Offset to the OT that can be used to retrieve the output of the OT
  [[nodiscard]] std::shared_ptr<OTVectorReceiver>& RegisterReceive(const std::size_t bitlen = 1,
                                                                   const std::size_t num_ots = 1,
                                                                   const OTProtocol p = GOT) {
    return receiver_provider_.RegisterOTs(bitlen, num_ots, p, Send_);
  }

  [[nodiscard]] std::unique_ptr<FixedXCOT128Receiver> RegisterReceiveFixedXCOT128(
      std::size_t num_ots = 1);

  [[nodiscard]] std::unique_ptr<XCOTBitReceiver> RegisterReceiveXCOTBit(
      std::size_t num_ots = 1, std::size_t vector_size = 1);

  template <typename T>
  [[nodiscard]] std::unique_ptr<ACOTReceiver<T>> RegisterReceiveACOT(std::size_t num_ots = 1,
                                                                     std::size_t vector_size = 1);

  [[nodiscard]] std::unique_ptr<GOT128Receiver> RegisterReceiveGOT128(std::size_t num_ots = 1);

  [[nodiscard]] std::unique_ptr<GOTBitReceiver> RegisterReceiveGOTBit(std::size_t num_ots = 1);

  [[nodiscard]] std::unique_ptr<ROTReceiver> RegisterReceiveROT(std::size_t num_ots = 1,
                                                                std::size_t vector_size = 1,
                                                                bool random_choice = true);

  [[nodiscard]] std::size_t GetNumOTsReceiver() const { return receiver_provider_.GetNumOTs(); }

  [[nodiscard]] std::size_t GetNumOTsSender() const { return sender_provider_.GetNumOTs(); }

  virtual void SendSetup() = 0;
  virtual void ReceiveSetup() = 0;

  void WaitSetup() const;

  void Clear() {
    receiver_provider_.Clear();
    sender_provider_.Clear();
  }

  void Reset() {
    receiver_provider_.Reset();
    sender_provider_.Reset();
  }

 protected:
  OTProvider(std::function<void(flatbuffers::FlatBufferBuilder&&)> Send,
             MOTION::OTExtensionData& data, std::size_t party_id,
             std::shared_ptr<MOTION::Logger> logger);

  std::function<void(flatbuffers::FlatBufferBuilder&&)> Send_;
  MOTION::OTExtensionData& data_;
  OTProviderReceiver receiver_provider_;
  OTProviderSender sender_provider_;
  std::shared_ptr<MOTION::Logger> logger_;
};

class OTProviderFromFile : public OTProvider {
  // TODO
};

class OTProviderFromBaseOTs : public OTProvider {
  // TODO
};

class OTProviderFromOTExtension final : public OTProvider {
 public:
  void SendSetup() final;

  void ReceiveSetup() final;

  OTProviderFromOTExtension(std::function<void(flatbuffers::FlatBufferBuilder&&)> Send,
                            MOTION::OTExtensionData& data, const MOTION::BaseOTsData& base_ot_data,
                            MOTION::Crypto::MotionBaseProvider&, std::size_t party_id,
                            std::shared_ptr<MOTION::Logger> logger);

 private:
  const MOTION::BaseOTsData& base_ot_data_;
  MOTION::Crypto::MotionBaseProvider& motion_base_provider_;
};

class OTProviderFromThirdParty : public OTProvider {
  // TODO
};

class OTProviderFromMultipleThirdParties : public OTProvider {
  // TODO
};

class OTProviderManager : public enable_wait_setup {
 public:
  OTProviderManager(MOTION::Communication::CommunicationLayer&, const MOTION::BaseOTProvider&,
                    MOTION::Crypto::MotionBaseProvider&, MOTION::Statistics::RunTimeStats*,
                    std::shared_ptr<MOTION::Logger>);
  ~OTProviderManager();

  std::vector<std::unique_ptr<OTProvider>>& get_providers() { return providers_; }
  OTProvider& get_provider(std::size_t party_id) { return *providers_.at(party_id); }
  void run_setup();

  // reset all data structures for a new round of OTs
  void clear();

 private:
  MOTION::Communication::CommunicationLayer& communication_layer_;
  const MOTION::BaseOTProvider& base_ot_provider_;
  MOTION::Crypto::MotionBaseProvider& motion_base_provider_;
  MOTION::Statistics::RunTimeStats* stats_;
  std::shared_ptr<MOTION::Logger> logger_;
  std::size_t num_parties_;
  std::vector<std::unique_ptr<OTProvider>> providers_;
  std::vector<std::unique_ptr<MOTION::OTExtensionData>> data_;
};

}  // namespace ObliviousTransfer
}  // namespace ENCRYPTO
