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

#include "comm_mixin.h"

#include <cstdint>
#include <unordered_map>

#include <boost/functional/hash.hpp>

#include "communication/communication_layer.h"
#include "communication/fbs_headers/comm_mixin_gate_message_generated.h"
#include "communication/message.h"
#include "communication/message_handler.h"
#include "utility/constants.h"
#include "utility/logger.h"

namespace {

struct SizeTPairHash {
  std::size_t operator()(const std::pair<std::size_t, std::size_t>& p) const {
    std::size_t seed = 0;
    boost::hash_combine(seed, p.first);
    boost::hash_combine(seed, p.second);
    return seed;
  }
};

}  // namespace

namespace MOTION::proto {

struct CommMixin::GateMessageHandler : public Communication::MessageHandler {
  GateMessageHandler(std::size_t num_parties, Communication::MessageType gate_message_type,
                     std::shared_ptr<Logger> logger);
  void received_message(std::size_t, std::vector<std::uint8_t>&& raw_message) override;

  enum class MsgValueType { bit, block, uint8, uint16, uint32, uint64 };

  template <typename T>
  constexpr static CommMixin::GateMessageHandler::MsgValueType get_msg_value_type();

  // KeyType = (gate_id, msg_num)
  using KeyType = std::pair<std::size_t, std::size_t>;

  // KeyType -> (size, type)
  std::unordered_map<KeyType, std::pair<std::size_t, MsgValueType>, SizeTPairHash>
      expected_messages_;

  // [KeyType -> promise]
  std::vector<std::unordered_map<KeyType, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>,
                                 SizeTPairHash>>
      bits_promises_;
  std::vector<std::unordered_map<KeyType, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::block128_vector>,
                                 SizeTPairHash>>
      blocks_promises_;
  std::vector<std::unordered_map<KeyType, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint8_t>>,
                                 SizeTPairHash>>
      uint8_promises_;
  std::vector<std::unordered_map<
      KeyType, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint16_t>>, SizeTPairHash>>
      uint16_promises_;
  std::vector<std::unordered_map<
      KeyType, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint32_t>>, SizeTPairHash>>
      uint32_promises_;
  std::vector<std::unordered_map<
      KeyType, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint64_t>>, SizeTPairHash>>
      uint64_promises_;

  template <typename T>
  std::vector<
      std::unordered_map<KeyType, ENCRYPTO::ReusableFiberPromise<std::vector<T>>, SizeTPairHash>>&
  get_promise_map();

  Communication::MessageType gate_message_type_;
  std::shared_ptr<Logger> logger_;
};

template <typename T>
constexpr CommMixin::GateMessageHandler::MsgValueType
CommMixin::GateMessageHandler::get_msg_value_type() {
  if constexpr (std::is_same_v<T, std::uint8_t>) {
    return CommMixin::GateMessageHandler::MsgValueType::uint8;
  } else if constexpr (std::is_same_v<T, std::uint16_t>) {
    return CommMixin::GateMessageHandler::MsgValueType::uint16;
  } else if constexpr (std::is_same_v<T, std::uint32_t>) {
    return CommMixin::GateMessageHandler::MsgValueType::uint32;
  } else if constexpr (std::is_same_v<T, std::uint64_t>) {
    return CommMixin::GateMessageHandler::MsgValueType::uint64;
  }
}

template <>
std::vector<
    std::unordered_map<CommMixin::GateMessageHandler::KeyType,
                       ENCRYPTO::ReusableFiberPromise<std::vector<std::uint8_t>>, SizeTPairHash>>&
CommMixin::GateMessageHandler::get_promise_map() {
  return uint8_promises_;
}
template <>
std::vector<
    std::unordered_map<CommMixin::GateMessageHandler::KeyType,
                       ENCRYPTO::ReusableFiberPromise<std::vector<std::uint16_t>>, SizeTPairHash>>&
CommMixin::GateMessageHandler::get_promise_map() {
  return uint16_promises_;
}
template <>
std::vector<
    std::unordered_map<CommMixin::GateMessageHandler::KeyType,
                       ENCRYPTO::ReusableFiberPromise<std::vector<std::uint32_t>>, SizeTPairHash>>&
CommMixin::GateMessageHandler::get_promise_map() {
  return uint32_promises_;
}
template <>
std::vector<
    std::unordered_map<CommMixin::GateMessageHandler::KeyType,
                       ENCRYPTO::ReusableFiberPromise<std::vector<std::uint64_t>>, SizeTPairHash>>&
CommMixin::GateMessageHandler::get_promise_map() {
  return uint64_promises_;
}

CommMixin::GateMessageHandler::GateMessageHandler(std::size_t num_parties,
                                                  Communication::MessageType gate_message_type,
                                                  std::shared_ptr<Logger> logger)
    : bits_promises_(num_parties),
      blocks_promises_(num_parties),
      uint8_promises_(num_parties),
      uint16_promises_(num_parties),
      uint32_promises_(num_parties),
      uint64_promises_(num_parties),
      gate_message_type_(gate_message_type),
      logger_(logger) {}

void CommMixin::GateMessageHandler::received_message(std::size_t party_id,
                                                     std::vector<std::uint8_t>&& raw_message) {
  assert(!raw_message.empty());
  auto message = Communication::GetMessage(raw_message.data());
  {
    flatbuffers::Verifier verifier(raw_message.data(), raw_message.size());
    if (!message->Verify(verifier)) {
      throw std::runtime_error("received malformed Message");
      // TODO: log and drop instead
    }
  }

  auto message_type = message->message_type();
  if (message_type != gate_message_type_) {
    throw std::logic_error(
        fmt::format("CommMixin::GateMessageHandler: received unexpected message of type {}",
                    EnumNameMessageType(message_type)));
  }

  auto gate_message =
      flatbuffers::GetRoot<MOTION::Communication::CommMixinGateMessage>(message->payload()->data());
  {
    flatbuffers::Verifier verifier(message->payload()->data(), message->payload()->size());
    if (!gate_message->Verify(verifier)) {
      throw std::runtime_error(
          fmt::format("received malformed {}", EnumNameMessageType(gate_message_type_)));
      // TODO: log and drop instead
    }
  }
  auto gate_id = gate_message->gate_id();
  auto msg_num = gate_message->msg_num();
  auto payload = gate_message->payload();
  auto it = expected_messages_.find({gate_id, msg_num});
  if (it == expected_messages_.end()) {
    logger_->LogError(fmt::format("received unexpected {} for gate {}, dropping",
                                  EnumNameMessageType(gate_message_type_), gate_id));
    return;
  }
  auto expected_size = it->second.first;
  auto type = it->second.second;

  auto set_value_helper = [this, party_id, gate_id, msg_num, expected_size, payload](
                              auto& map_vec, auto type_tag) {
    auto byte_size = expected_size * sizeof(type_tag);
    if (byte_size != payload->size()) {
      logger_->LogError(fmt::format(
          "received {} for gate {} of size {} while expecting size {}, dropping",
          EnumNameMessageType(gate_message_type_), gate_id, payload->size(), byte_size));
      return;
    }
    auto& promise_map = map_vec[party_id];
    auto& promise = promise_map.at({gate_id, msg_num});
    auto ptr = reinterpret_cast<const decltype(type_tag)*>(payload->data());
    promise.set_value(std::vector(ptr, ptr + expected_size));
  };

  switch (type) {
    case MsgValueType::bit: {
      auto byte_size = Helpers::Convert::BitsToBytes(expected_size);
      if (byte_size != payload->size()) {
        logger_->LogError(fmt::format(
            "received {} for gate {} of size {} while expecting size {}, dropping",
            EnumNameMessageType(gate_message_type_), gate_id, payload->size(), byte_size));
        return;
      }
      auto& promise = bits_promises_[party_id].at({gate_id, msg_num});
      promise.set_value(ENCRYPTO::BitVector(payload->data(), expected_size));
      break;
    }
    case MsgValueType::block: {
      auto byte_size = 16 * expected_size;
      if (byte_size != payload->size()) {
        logger_->LogError(fmt::format(
            "received {} for gate {} of size {} while expecting size {}, dropping",
            EnumNameMessageType(gate_message_type_), gate_id, payload->size(), byte_size));
        return;
      }
      auto& promise = blocks_promises_[party_id].at({gate_id, msg_num});
      promise.set_value(ENCRYPTO::block128_vector(expected_size, payload->data()));
      break;
    }
    case MsgValueType::uint8: {
      set_value_helper(uint8_promises_, std::uint8_t{});
      break;
    }
    case MsgValueType::uint16: {
      set_value_helper(uint16_promises_, std::uint16_t{});
      break;
    }
    case MsgValueType::uint32: {
      set_value_helper(uint32_promises_, std::uint32_t{});
      break;
    }
    case MsgValueType::uint64: {
      set_value_helper(uint64_promises_, std::uint64_t{});
      break;
    }
  }
}

CommMixin::CommMixin(Communication::CommunicationLayer& communication_layer,
                     Communication::MessageType gate_message_type, std::shared_ptr<Logger> logger)
    : communication_layer_(communication_layer),
      gate_message_type_(gate_message_type),
      my_id_(communication_layer.get_my_id()),
      num_parties_(communication_layer.get_num_parties()),
      message_handler_(std::make_unique<GateMessageHandler>(communication_layer_.get_num_parties(),
                                                            gate_message_type, logger)),
      logger_(std::move(logger)) {
  // TODO
  communication_layer_.register_message_handler([this](auto) { return message_handler_; },
                                                {gate_message_type});
}

CommMixin::~CommMixin() = default;

flatbuffers::FlatBufferBuilder CommMixin::build_gate_message(std::size_t gate_id,
                                                             std::size_t msg_num,
                                                             const std::uint8_t* message,
                                                             std::size_t size) const {
  flatbuffers::FlatBufferBuilder builder;
  auto vector = builder.CreateVector(message, size);
  auto root = Communication::CreateCommMixinGateMessage(builder, gate_id, msg_num, vector);
  builder.Finish(root);
  return Communication::BuildMessage(gate_message_type_, builder.GetBufferPointer(),
                                     builder.GetSize());
}

template <typename T>
flatbuffers::FlatBufferBuilder CommMixin::build_gate_message(std::size_t gate_id,
                                                             std::size_t msg_num,
                                                             const std::vector<T>& vector) const {
  return build_gate_message(gate_id, msg_num, reinterpret_cast<const std::uint8_t*>(vector.data()),
                            sizeof(T) * vector.size());
}

flatbuffers::FlatBufferBuilder CommMixin::build_gate_message(
    std::size_t gate_id, std::size_t msg_num, const ENCRYPTO::BitVector<>& message) const {
  auto vector = message.GetData();
  return build_gate_message(gate_id, msg_num, reinterpret_cast<const std::uint8_t*>(vector.data()),
                            vector.size());
}

flatbuffers::FlatBufferBuilder CommMixin::build_gate_message(
    std::size_t gate_id, std::size_t msg_num, const ENCRYPTO::block128_vector& message) const {
  auto data = message.data();
  return build_gate_message(gate_id, msg_num, reinterpret_cast<const std::uint8_t*>(data),
                            16 * message.size());
}

void CommMixin::broadcast_bits_message(std::size_t gate_id, const ENCRYPTO::BitVector<>& message,
                                       std::size_t msg_num) const {
  communication_layer_.broadcast_message(build_gate_message(gate_id, msg_num, message));
}

void CommMixin::send_bits_message(std::size_t party_id, std::size_t gate_id,
                                  const ENCRYPTO::BitVector<>& message, std::size_t msg_num) const {
  communication_layer_.send_message(party_id, build_gate_message(gate_id, msg_num, message));
}

[[nodiscard]] std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>>
CommMixin::register_for_bits_messages(std::size_t gate_id, std::size_t num_bits,
                                      std::size_t msg_num) {
  auto& mh = *message_handler_;
  std::vector<ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>> promises(num_parties_);
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>> futures;
  std::transform(std::begin(promises), std::end(promises), std::back_inserter(futures),
                 [](auto& p) { return p.get_future(); });
  auto [_, success] = mh.expected_messages_.insert(
      {std::make_pair(gate_id, msg_num),
       std::make_pair(num_bits, GateMessageHandler::MsgValueType::bit)});
  if (!success) {
    throw std::logic_error(fmt::format("tried to register twice for message for gate {}", gate_id));
  }
  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    auto& promise_map = mh.bits_promises_.at(party_id);
    auto [_, success] =
        promise_map.insert({std::make_pair(gate_id, msg_num), std::move(promises.at(party_id))});
    assert(success);
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for bits messages of size {}", gate_id, num_bits));
    }
  }
  return futures;
}

[[nodiscard]] ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>
CommMixin::register_for_bits_message(std::size_t party_id, std::size_t gate_id,
                                     std::size_t num_bits, std::size_t msg_num) {
  assert(party_id != my_id_);
  auto& mh = *message_handler_;
  ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>> promise;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>> future = promise.get_future();
  auto [_, success] = mh.expected_messages_.insert(
      {std::make_pair(gate_id, msg_num),
       std::make_pair(num_bits, GateMessageHandler::MsgValueType::bit)});
  if (!success) {
    throw std::logic_error(fmt::format("tried to register twice for message for gate {}", gate_id));
  }
  {
    auto& promise_map = mh.bits_promises_.at(party_id);
    auto [_, success] = promise_map.insert({std::make_pair(gate_id, msg_num), std::move(promise)});
    assert(success);
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for bits message of size {}", gate_id, num_bits));
    }
  }
  return future;
}

void CommMixin::broadcast_blocks_message(std::size_t gate_id,
                                         const ENCRYPTO::block128_vector& message,
                                         std::size_t msg_num) const {
  communication_layer_.broadcast_message(build_gate_message(gate_id, msg_num, message));
}

void CommMixin::send_blocks_message(std::size_t party_id, std::size_t gate_id,
                                    const ENCRYPTO::block128_vector& message,
                                    std::size_t msg_num) const {
  communication_layer_.send_message(party_id, build_gate_message(gate_id, msg_num, message));
}

[[nodiscard]] std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector>>
CommMixin::register_for_blocks_messages(std::size_t gate_id, std::size_t num_blocks,
                                        std::size_t msg_num) {
  auto& mh = *message_handler_;
  std::vector<ENCRYPTO::ReusableFiberPromise<ENCRYPTO::block128_vector>> promises(num_parties_);
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector>> futures;
  std::transform(std::begin(promises), std::end(promises), std::back_inserter(futures),
                 [](auto& p) { return p.get_future(); });
  auto [_, success] = mh.expected_messages_.insert(
      {std::make_pair(gate_id, msg_num),
       std::make_pair(num_blocks, GateMessageHandler::MsgValueType::block)});
  if (!success) {
    throw std::logic_error(fmt::format("tried to register twice for message for gate {}", gate_id));
  }
  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    auto& promise_map = mh.blocks_promises_.at(party_id);
    auto [_, success] =
        promise_map.insert({std::make_pair(gate_id, msg_num), std::move(promises.at(party_id))});
    assert(success);
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for blocks messages of size {}", gate_id, num_blocks));
    }
  }
  return futures;
}

[[nodiscard]] ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector>
CommMixin::register_for_blocks_message(std::size_t party_id, std::size_t gate_id,
                                       std::size_t num_blocks, std::size_t msg_num) {
  assert(party_id != my_id_);
  auto& mh = *message_handler_;
  ENCRYPTO::ReusableFiberPromise<ENCRYPTO::block128_vector> promise;
  ENCRYPTO::ReusableFiberFuture<ENCRYPTO::block128_vector> future = promise.get_future();
  auto [_, success] = mh.expected_messages_.insert(
      {std::make_pair(gate_id, msg_num),
       std::make_pair(num_blocks, GateMessageHandler::MsgValueType::block)});
  if (!success) {
    throw std::logic_error(fmt::format("tried to register twice for message for gate {}", gate_id));
  }
  {
    auto& promise_map = mh.blocks_promises_.at(party_id);
    auto [_, success] = promise_map.insert({std::make_pair(gate_id, msg_num), std::move(promise)});
    assert(success);
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for blocks message of size {}", gate_id, num_blocks));
    }
  }
  return future;
}

template <typename T>
void CommMixin::broadcast_ints_message(std::size_t gate_id, const std::vector<T>& message,
                                       std::size_t msg_num) const {
  communication_layer_.broadcast_message(build_gate_message(gate_id, msg_num, message));
}

template void CommMixin::broadcast_ints_message(std::size_t, const std::vector<std::uint8_t>&,
                                                std::size_t) const;
template void CommMixin::broadcast_ints_message(std::size_t, const std::vector<std::uint16_t>&,
                                                std::size_t) const;
template void CommMixin::broadcast_ints_message(std::size_t, const std::vector<std::uint32_t>&,
                                                std::size_t) const;
template void CommMixin::broadcast_ints_message(std::size_t, const std::vector<std::uint64_t>&,
                                                std::size_t) const;

template <typename T>
void CommMixin::send_ints_message(std::size_t party_id, std::size_t gate_id,
                                  const std::vector<T>& message, std::size_t msg_num) const {
  communication_layer_.send_message(party_id, build_gate_message(gate_id, msg_num, message));
}

template void CommMixin::send_ints_message(std::size_t, std::size_t,
                                           const std::vector<std::uint8_t>&, std::size_t) const;
template void CommMixin::send_ints_message(std::size_t, std::size_t,
                                           const std::vector<std::uint16_t>&, std::size_t) const;
template void CommMixin::send_ints_message(std::size_t, std::size_t,
                                           const std::vector<std::uint32_t>&, std::size_t) const;
template void CommMixin::send_ints_message(std::size_t, std::size_t,
                                           const std::vector<std::uint64_t>&, std::size_t) const;

template <typename T>
[[nodiscard]] std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<T>>>
CommMixin::register_for_ints_messages(std::size_t gate_id, std::size_t num_elements,
                                      std::size_t msg_num) {
  auto& mh = *message_handler_;
  std::vector<ENCRYPTO::ReusableFiberPromise<std::vector<T>>> promises(num_parties_);
  std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<T>>> futures;
  std::transform(std::begin(promises), std::end(promises), std::back_inserter(futures),
                 [](auto& p) { return p.get_future(); });
  auto type = GateMessageHandler::get_msg_value_type<T>();
  auto [_, success] = mh.expected_messages_.insert(
      {std::make_pair(gate_id, msg_num), std::make_pair(num_elements, type)});
  if (!success) {
    throw std::logic_error(
        fmt::format("tried to register twice for message {} for gate {}", msg_num, gate_id));
  }
  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    auto& promise_map = mh.get_promise_map<T>().at(party_id);
    auto [_, success] =
        promise_map.insert({std::make_pair(gate_id, msg_num), std::move(promises.at(party_id))});
    assert(success);
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(fmt::format("Gate {}: registered for int messages {} of size {}", gate_id,
                                    msg_num, num_elements));
    }
  }
  return futures;
}

template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint8_t>>>
    CommMixin::register_for_ints_messages(std::size_t, std::size_t, std::size_t);
template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint16_t>>>
    CommMixin::register_for_ints_messages(std::size_t, std::size_t, std::size_t);
template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint32_t>>>
    CommMixin::register_for_ints_messages(std::size_t, std::size_t, std::size_t);
template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>>>
    CommMixin::register_for_ints_messages(std::size_t, std::size_t, std::size_t);

template <typename T>
[[nodiscard]] ENCRYPTO::ReusableFiberFuture<std::vector<T>> CommMixin::register_for_ints_message(
    std::size_t party_id, std::size_t gate_id, std::size_t num_elements, std::size_t msg_num) {
  assert(party_id != my_id_);
  auto& mh = *message_handler_;
  ENCRYPTO::ReusableFiberPromise<std::vector<T>> promise;
  ENCRYPTO::ReusableFiberFuture<std::vector<T>> future = promise.get_future();
  auto type = GateMessageHandler::get_msg_value_type<T>();
  auto [_, success] = mh.expected_messages_.insert(
      {std::make_pair(gate_id, msg_num), std::make_pair(num_elements, type)});
  if (!success) {
    throw std::logic_error(
        fmt::format("tried to register twice for message {} for gate {}", msg_num, gate_id));
  }
  {
    auto& promise_map = mh.get_promise_map<T>().at(party_id);
    auto [_, success] = promise_map.insert({std::make_pair(gate_id, msg_num), std::move(promise)});
    assert(success);
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(fmt::format("Gate {}: registered for int message {} of size {}", gate_id,
                                    msg_num, num_elements));
    }
  }
  return future;
}

template ENCRYPTO::ReusableFiberFuture<std::vector<std::uint8_t>>
    CommMixin::register_for_ints_message(std::size_t, std::size_t, std::size_t, std::size_t);
template ENCRYPTO::ReusableFiberFuture<std::vector<std::uint16_t>>
    CommMixin::register_for_ints_message(std::size_t, std::size_t, std::size_t, std::size_t);
template ENCRYPTO::ReusableFiberFuture<std::vector<std::uint32_t>>
    CommMixin::register_for_ints_message(std::size_t, std::size_t, std::size_t, std::size_t);
template ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>>
    CommMixin::register_for_ints_message(std::size_t, std::size_t, std::size_t, std::size_t);

}  // namespace MOTION::proto
