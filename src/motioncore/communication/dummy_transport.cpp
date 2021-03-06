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

#include "dummy_transport.h"

namespace MOTION::Communication {

DummyTransport::DummyTransport(DummyTransport&& other)
    : Transport(std::move(other)),
      send_queue_(std::move(other.send_queue_)),
      receive_queue_(std::move(other.receive_queue_)) {}

DummyTransport::DummyTransport(std::shared_ptr<message_queue_t> send_queue,
                               std::shared_ptr<message_queue_t> receive_queue) noexcept
    : send_queue_(send_queue), receive_queue_(receive_queue) {}

std::pair<std::unique_ptr<DummyTransport>, std::unique_ptr<DummyTransport>>
DummyTransport::make_transport_pair() {
  auto queue_0 = std::make_shared<message_queue_t>();
  auto queue_1 = std::make_shared<message_queue_t>();
  auto transport_0 = std::unique_ptr<DummyTransport>(new DummyTransport(queue_0, queue_1));
  auto transport_1 = std::unique_ptr<DummyTransport>(new DummyTransport(queue_1, queue_0));
  return std::make_pair(std::move(transport_0), std::move(transport_1));
}

void DummyTransport::send_message(std::vector<std::uint8_t>&& message) {
  auto message_size = message.size();
  send_queue_->enqueue(std::move(message));
  statistics_.num_messages_sent += 1;
  statistics_.num_bytes_sent += message_size;
}

void DummyTransport::send_message(const std::vector<std::uint8_t>& message) {
  auto message_size = message.size();
  send_queue_->enqueue(message);
  statistics_.num_messages_sent += 1;
  statistics_.num_bytes_sent += message_size;
}

void DummyTransport::send_message(const std::uint8_t* message, std::size_t size) {
  send_message(std::vector<std::uint8_t>(message, message + size));
}

bool DummyTransport::available() const { return !receive_queue_->empty(); }

std::optional<std::vector<std::uint8_t>> DummyTransport::receive_message() {
  auto message_opt = receive_queue_->dequeue();
  if (!message_opt.has_value()) {
    // transport has been closed
    assert(receive_queue_->closed());
    return std::nullopt;
  }
  statistics_.num_messages_received += 1;
  statistics_.num_bytes_received += message_opt->size();
  return message_opt;
}

void DummyTransport::shutdown_send() { send_queue_->close(); }

void DummyTransport::shutdown() {
  shutdown_send();
  receive_queue_->close();
}

}  // namespace MOTION::Communication
