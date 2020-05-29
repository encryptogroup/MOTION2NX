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

#include <mutex>
#include "fiber_condition.h"

namespace ENCRYPTO {

class enable_wait_online {
 public:
  void set_online_ready() noexcept {
    {
      std::scoped_lock lock(online_cond_.GetMutex());
      online_ready_ = true;
    }
    online_cond_.NotifyAll();
  }
  void wait_online() const noexcept { online_cond_.Wait(); }

 private:
  bool online_ready_ = false;
  ENCRYPTO::FiberCondition online_cond_ =
      ENCRYPTO::FiberCondition([this] { return online_ready_; });
};

class enable_wait_setup {
 public:
  void set_setup_ready() noexcept {
    {
      std::scoped_lock lock(setup_cond_.GetMutex());
      setup_ready_ = true;
    }
    setup_cond_.NotifyAll();
  }
  void wait_setup() const noexcept { setup_cond_.Wait(); }

 private:
  bool setup_ready_ = false;
  ENCRYPTO::FiberCondition setup_cond_ = ENCRYPTO::FiberCondition([this] { return setup_ready_; });
};

}  // namespace ENCRYPTO
