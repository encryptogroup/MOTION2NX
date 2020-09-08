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

#include "network_builder.h"

#include <fmt/format.h>
#include <stdexcept>

#include "tensor.h"
#include "tensor_op_factory.h"
#include "utility/typedefs.h"

namespace MOTION::tensor {

TensorCP NetworkBuilder::convert(MPCProtocol dst_proto, const TensorCP tensor_in) {
  const auto src_proto = tensor_in->get_protocol();
  const auto convert_f = [dst_proto, tensor_in](auto& factory) -> std::pair<TensorCP, bool> {
    try {
      auto tensor_out = factory.make_tensor_conversion(dst_proto, tensor_in);
      return {std::move(tensor_out), true};
    } catch (std::exception& e) {
      return {{}, false};
    }
  };
  auto via_protocol = convert_via(src_proto, dst_proto);
  if (via_protocol.has_value()) {
    // implicit conversion via third protocol
    auto tmp = convert(*via_protocol, tensor_in);
    return convert(dst_proto, tmp);
  } else {
    // direct conversion
    {
      auto& factory = get_tensor_op_factory(src_proto);
      auto [output_wires, success] = convert_f(factory);
      if (success) {
        return output_wires;
      }
    }
    {
      auto& factory = get_tensor_op_factory(dst_proto);
      auto [output_wires, success] = convert_f(factory);
      if (success) {
        return output_wires;
      }
    }
  }
  throw std::runtime_error(fmt::format("no conversion from {} to {} supported", ToString(src_proto),
                                       ToString(dst_proto)));
}

std::optional<MPCProtocol> NetworkBuilder::convert_via(MPCProtocol, MPCProtocol) {
  return std::nullopt;
}

}  // namespace MOTION::tensor
