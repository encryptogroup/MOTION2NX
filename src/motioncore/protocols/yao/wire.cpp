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

#include "wire.h"

#include <mutex>

namespace MOTION::proto::yao {

YaoWire::YaoWire(std::size_t num_simd)
    : NewWire(num_simd),
      keys_(num_simd),
      setup_done_(false),
      setup_cond_([this] { return setup_done_; }) {}

void YaoWire::wait_setup() const { setup_cond_.Wait(); }

void YaoWire::set_setup_ready() {
  {
    std::scoped_lock lock(setup_cond_.GetMutex());
    setup_done_ = true;
  }
  setup_cond_.NotifyAll();
}

}  // namespace MOTION::proto::yao
