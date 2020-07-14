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

#include "onnx_visitor.h"

#include <onnx/onnx_pb.h>

namespace MOTION::onnx {

void OnnxVisitor::visit_model(const ::onnx::ModelProto& model) { visit_graph(model.graph()); }

void OnnxVisitor::visit_graph(const ::onnx::GraphProto& graph) {
  for (const auto& initializer : graph.initializer()) {
    visit_initializer(initializer);
  }
  for (const auto& input : graph.input()) {
    visit_input(input);
  }
  for (const auto& output : graph.output()) {
    visit_output(output);
  }
  for (const auto& node : graph.node()) {
    visit_node(node);
  }
}

void OnnxVisitor::visit_node(const ::onnx::NodeProto& node) {
  const auto& op_type = node.op_type();
  if (op_type == "Gemm") {
    visit_gemm(node);
  } else if (op_type == "Conv") {
    visit_conv(node);
  } else if (op_type == "Relu") {
    visit_relu(node);
  } else if (op_type == "MaxPool") {
    visit_maxpool(node);
  } else {
    // TODO: warn unsupported node
  }
}

}  // namespace MOTION::onnx
