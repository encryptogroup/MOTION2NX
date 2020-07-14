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

namespace onnx {
class GraphProto;
class ModelProto;
class NodeProto;
class TensorProto;
class ValueInfoProto;
}  // namespace onnx

namespace MOTION::onnx {

class OnnxVisitor {
 public:
  virtual void visit_model(const ::onnx::ModelProto&);
  virtual void visit_graph(const ::onnx::GraphProto&);

  virtual void visit_initializer(const ::onnx::TensorProto&) = 0;
  virtual void visit_input(const ::onnx::ValueInfoProto&) = 0;
  virtual void visit_output(const ::onnx::ValueInfoProto&) = 0;

  virtual void visit_node(const ::onnx::NodeProto&);
  virtual void visit_conv(const ::onnx::NodeProto&) = 0;
  virtual void visit_flatten(const ::onnx::NodeProto&) = 0;
  virtual void visit_gemm(const ::onnx::NodeProto&) = 0;
  virtual void visit_maxpool(const ::onnx::NodeProto&) = 0;
  virtual void visit_mul(const ::onnx::NodeProto&) = 0;
  virtual void visit_relu(const ::onnx::NodeProto&) = 0;
};

}  // namespace MOTION::onnx
