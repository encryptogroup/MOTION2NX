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

#include <unordered_map>
#include <unordered_set>

#include "onnx_visitor.h"
#include "tensor/tensor.h"
#include "utility/reusable_future.h"
#include "utility/typedefs.h"

namespace MOTION {

namespace tensor {
class NetworkBuilder;
}

namespace onnx {

class OnnxAdapter : public OnnxVisitor {
 public:
  OnnxAdapter(tensor::NetworkBuilder& network_builder, MPCProtocol arithmetic_protocol,
              MPCProtocol boolean_protocol, bool is_model_provider);
  void visit_initializer(const ::onnx::TensorProto&) override;
  void visit_input(const ::onnx::ValueInfoProto&) override;
  void visit_output(const ::onnx::ValueInfoProto&) override;
  void visit_gemm(const ::onnx::NodeProto&) override;
  void visit_conv(const ::onnx::NodeProto&) override;
  void visit_mul(const ::onnx::NodeProto&) override;
  void visit_relu(const ::onnx::NodeProto&) override;
  void visit_maxpool(const ::onnx::NodeProto&) override;
  void visit_flatten(const ::onnx::NodeProto&) override;
  tensor::TensorCP get_as_arithmetic_tensor(const std::string&);
  tensor::TensorCP get_as_boolean_tensor(const std::string&);

 private:
  tensor::NetworkBuilder& network_builder_;
  MPCProtocol arithmetic_protocol_;
  MPCProtocol boolean_protocol_;
  bool is_model_provider_;

  std::unordered_set<std::string> initializer_set_;
  std::unordered_map<std::string, tensor::TensorCP> arithmetic_tensor_map_;
  std::unordered_map<std::string, tensor::TensorCP> boolean_tensor_map_;
  std::unordered_map<std::string, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>>>
      input_promises_;
  std::unordered_map<std::string, ENCRYPTO::ReusableFiberFuture<std::vector<std::uint64_t>>>
      output_futures_;
};

}  // namespace onnx
}  // namespace MOTION
