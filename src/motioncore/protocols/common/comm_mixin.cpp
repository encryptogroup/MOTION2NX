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

#include "communication/communication_layer.h"
#include "communication/fbs_headers/gmw_message_generated.h"
#include "communication/message.h"
#include "communication/message_handler.h"
#include "utility/constants.h"
#include "utility/logger.h"

namespace MOTION::proto {

struct CommMixin::GateMessageHandler : public Communication::MessageHandler {
  GateMessageHandler(std::size_t num_parties, Communication::MessageType gate_message_type,
                     std::shared_ptr<Logger> logger);
  void received_message(std::size_t, std::vector<std::uint8_t>&& raw_message) override;

  enum class MsgValueType { bit, uint8, uint16, uint32, uint64 };

  template <typename T>
  constexpr static CommMixin::GateMessageHandler::MsgValueType get_msg_value_type();

  // gate_id -> (size, type)
  std::unordered_map<std::size_t, std::pair<std::size_t, MsgValueType>> expected_messages_;

  // [gate_id -> promise]
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>>>
      bits_promises_;
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint8_t>>>>
      uint8_promises_;
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint16_t>>>>
      uint16_promises_;
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint32_t>>>>
      uint32_promises_;
  std::vector<
      std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint64_t>>>>
      uint64_promises_;

  template <typename T>
  std::vector<std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<T>>>>&
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
    std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint8_t>>>>&
CommMixin::GateMessageHandler::get_promise_map() {
  return uint8_promises_;
}
template <>
std::vector<
    std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint16_t>>>>&
CommMixin::GateMessageHandler::get_promise_map() {
  return uint16_promises_;
}
template <>
std::vector<
    std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint32_t>>>>&
CommMixin::GateMessageHandler::get_promise_map() {
  return uint32_promises_;
}
template <>
std::vector<
    std::unordered_map<std::size_t, ENCRYPTO::ReusableFiberPromise<std::vector<std::uint64_t>>>>&
CommMixin::GateMessageHandler::get_promise_map() {
  return uint64_promises_;
}

CommMixin::GateMessageHandler::GateMessageHandler(std::size_t num_parties,
                                                  Communication::MessageType gate_message_type,
                                                  std::shared_ptr<Logger> logger)
    : bits_promises_(num_parties),
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
      flatbuffers::GetRoot<MOTION::Communication::GMWGateMessage>(message->payload()->data());
  {
    flatbuffers::Verifier verifier(message->payload()->data(), message->payload()->size());
    if (!gate_message->Verify(verifier)) {
      throw std::runtime_error(
          fmt::format("received malformed {}", EnumNameMessageType(gate_message_type_)));
      // TODO: log and drop instead
    }
  }
  auto gate_id = gate_message->gate_id();
  auto payload = gate_message->payload();
  auto it = expected_messages_.find(gate_id);
  if (it == expected_messages_.end()) {
    logger_->LogError(fmt::format("received unexpected {} for gate {}, dropping",
                                  EnumNameMessageType(gate_message_type_), gate_id));
    return;
  }
  auto expected_size = it->second.first;
  auto type = it->second.second;

  auto set_value_helper = [this, party_id, gate_id, expected_size, payload](auto& map_vec,
                                                                            auto type_tag) {
    auto byte_size = expected_size * sizeof(type_tag);
    if (byte_size != payload->size()) {
      logger_->LogError(fmt::format(
          "received {} for gate {} of size {} while expecting size {}, dropping",
          EnumNameMessageType(gate_message_type_), gate_id, payload->size(), byte_size));
      return;
    }
    auto& promise_map = map_vec[party_id];
    auto& promise = promise_map.at(gate_id);
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
      auto& promise = bits_promises_[party_id].at(gate_id);
      promise.set_value(ENCRYPTO::BitVector(payload->data(), expected_size));
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
                                                             const std::uint8_t* message,
                                                             std::size_t size) const {
  flatbuffers::FlatBufferBuilder builder;
  auto vector = builder.CreateVector(message, size);
  auto root = Communication::CreateGMWGateMessage(builder, gate_id, vector);
  builder.Finish(root);
  return Communication::BuildMessage(gate_message_type_, builder.GetBufferPointer(),
                                     builder.GetSize());
}

template <typename T>
flatbuffers::FlatBufferBuilder CommMixin::build_gate_message(std::size_t gate_id,
                                                             const std::vector<T>& vector) const {
  return build_gate_message(gate_id, reinterpret_cast<const std::uint8_t*>(vector.data()),
                            sizeof(T) * vector.size());
}

flatbuffers::FlatBufferBuilder CommMixin::build_gate_message(
    std::size_t gate_id, const ENCRYPTO::BitVector<>& message) const {
  return build_gate_message(gate_id, message.GetData());
}

void CommMixin::broadcast_bits_message(std::size_t gate_id,
                                       const ENCRYPTO::BitVector<>& message) const {
  communication_layer_.broadcast_message(build_gate_message(gate_id, message));
}

void CommMixin::send_bits_message(std::size_t party_id, std::size_t gate_id,
                                  const ENCRYPTO::BitVector<>& message) const {
  communication_layer_.send_message(party_id, build_gate_message(gate_id, message));
}

[[nodiscard]] std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>>
CommMixin::register_for_bits_messages(std::size_t gate_id, std::size_t num_bits) {
  auto& mh = *message_handler_;
  std::vector<ENCRYPTO::ReusableFiberPromise<ENCRYPTO::BitVector<>>> promises(num_parties_);
  std::vector<ENCRYPTO::ReusableFiberFuture<ENCRYPTO::BitVector<>>> futures;
  std::transform(std::begin(promises), std::end(promises), std::back_inserter(futures),
                 [](auto& p) { return p.get_future(); });
  auto [_, success] = mh.expected_messages_.insert(
      {gate_id, std::make_pair(num_bits, GateMessageHandler::MsgValueType::bit)});
  if (!success) {
    throw std::logic_error(fmt::format("tried to register twice for message for gate {}", gate_id));
  }
  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    auto& promise_map = mh.bits_promises_.at(party_id);
    auto [_, success] = promise_map.insert({gate_id, std::move(promises.at(party_id))});
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

template <typename T>
void CommMixin::broadcast_ints_message(std::size_t gate_id, const std::vector<T>& message) const {
  communication_layer_.broadcast_message(build_gate_message(gate_id, message));
}

template void CommMixin::broadcast_ints_message(std::size_t,
                                                const std::vector<std::uint8_t>&) const;
template void CommMixin::broadcast_ints_message(std::size_t,
                                                const std::vector<std::uint16_t>&) const;
template void CommMixin::broadcast_ints_message(std::size_t,
                                                const std::vector<std::uint32_t>&) const;
template void CommMixin::broadcast_ints_message(std::size_t,
                                                const std::vector<std::uint64_t>&) const;

template <typename T>
void CommMixin::send_ints_message(std::size_t party_id, std::size_t gate_id,
                                  const std::vector<T>& message) const {
  communication_layer_.send_message(party_id, build_gate_message(gate_id, message));
}

template void CommMixin::send_ints_message(std::size_t, std::size_t,
                                           const std::vector<std::uint8_t>&) const;
template void CommMixin::send_ints_message(std::size_t, std::size_t,
                                           const std::vector<std::uint16_t>&) const;
template void CommMixin::send_ints_message(std::size_t, std::size_t,
                                           const std::vector<std::uint32_t>&) const;
template void CommMixin::send_ints_message(std::size_t, std::size_t,
                                           const std::vector<std::uint64_t>&) const;

template <typename T>
[[nodiscard]] std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<T>>>
CommMixin::register_for_ints_messages(std::size_t gate_id, std::size_t num_elements) {
  auto& mh = *message_handler_;
  std::vector<ENCRYPTO::ReusableFiberPromise<std::vector<T>>> promises(num_parties_);
  std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<T>>> futures;
  std::transform(std::begin(promises), std::end(promises), std::back_inserter(futures),
                 [](auto& p) { return p.get_future(); });
  auto type = GateMessageHandler::get_msg_value_type<T>();
  auto [_, success] = mh.expected_messages_.insert({gate_id, std::make_pair(num_elements, type)});
  if (!success) {
    throw std::logic_error(fmt::format("tried to register twice for message for gate {}", gate_id));
  }
  for (std::size_t party_id = 0; party_id < num_parties_; ++party_id) {
    if (party_id == my_id_) {
      continue;
    }
    auto& promise_map = mh.get_promise_map<T>().at(party_id);
    auto [_, success] = promise_map.insert({gate_id, std::move(promises.at(party_id))});
    assert(success);
  }
  if constexpr (MOTION_VERBOSE_DEBUG) {
    if (logger_) {
      logger_->LogTrace(
          fmt::format("Gate {}: registered for int messages of size {}", gate_id, num_elements));
    }
  }
  return futures;
}

template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint8_t>>>
    CommMixin::register_for_ints_messages(std::size_t, std::size_t);
template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint16_t>>>
    CommMixin::register_for_ints_messages(std::size_t, std::size_t);
template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint32_t>>>
    CommMixin::register_for_ints_messages(std::size_t, std::size_t);
template std::vector<ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>>>
    CommMixin::register_for_ints_messages(std::size_t, std::size_t);

}  // namespace MOTION::proto
