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

#define FMT_HEADER_ONLY 1

#include "onnx_adapter.h"

#include <fmt/format.h>
#include <onnx/onnx_pb.h>
#include <stdexcept>

#include "tensor/network_builder.h"
#include "tensor/tensor_op_factory.h"

namespace MOTION::onnx {

OnnxAdapter::OnnxAdapter(tensor::NetworkBuilder& network_builder, MPCProtocol arithmetic_protocol,
                         MPCProtocol boolean_protocol, bool is_model_provider)
    : network_builder_(network_builder),
      arithmetic_protocol_(arithmetic_protocol),
      boolean_protocol_(boolean_protocol),
      is_model_provider_(is_model_provider) {}

void OnnxAdapter::visit_initializer(const ::onnx::TensorProto& tensor) {
  if (tensor.dims_size() > 4) {
    throw std::invalid_argument("tensors with > 4 dimensions are not yet supported");
  }
  if (tensor.dims_size() < 1) {
    throw std::invalid_argument("tensors needs at least 1 dimension");
  }

  const auto to_tensor_dims = [](const auto& dims) {
    tensor::TensorDimensions tensor_dims = {1, 1, 1, 1};
    auto n = dims.size();
    switch (n) {
      case 4:
        tensor_dims.batch_size_ = dims[n - 4];
        [[fallthrough]];
      case 3:
        tensor_dims.num_channels_ = dims[n - 3];
        [[fallthrough]];
      case 2:
        tensor_dims.height_ = dims[n - 2];
        [[fallthrough]];
      case 1:
        tensor_dims.width_ = dims[n - 1];
        break;
      default:
        throw std::logic_error("invalid number of dimensions");
    }
    return tensor_dims;
  };

  tensor::TensorDimensions tensor_dims = to_tensor_dims(tensor.dims());
  tensor::TensorCP tensor_share;
  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  if (is_model_provider_) {
    auto result = tensor_op_factory.make_arithmetic_64_tensor_input_my(tensor_dims);
    tensor_share = std::move(result.second);
  } else {
    tensor_share = tensor_op_factory.make_arithmetic_64_tensor_input_other(tensor_dims);
  }
  arithmetic_tensor_map_.insert({tensor.name(), std::move(tensor_share)});
  initializer_set_.insert(tensor.name());
}

void OnnxAdapter::visit_input(const ::onnx::ValueInfoProto& value_info) {
  if (initializer_set_.count(value_info.name())) {
    // already handled in visit_initializer
    return;
  }

  const auto& type = value_info.type();
  if (type.value_case() != ::onnx::TypeProto::kTensorType) {
    throw std::invalid_argument("unsupported type");
  }
  const auto& tensor_type = type.tensor_type();
  const auto& elem_type = tensor_type.elem_type();
  if (elem_type != ::onnx::TensorProto::FLOAT) {
    throw std::invalid_argument("unsupported element type");
  }
  const auto& tensor_shape = tensor_type.shape();
  if (tensor_shape.dim_size() > 4) {
    throw std::invalid_argument("tensors with > 4 dimensions are not yet supported");
  }
  if (tensor_shape.dim_size() < 1) {
    throw std::invalid_argument("tensors needs at least 1 dimension");
  }

  const auto to_tensor_dims = [](const auto& dims) {
    if (std::any_of(std::begin(dims), std::end(dims), [](const auto& d) {
          return d.value_case() != ::onnx::TensorShapeProto_Dimension::kDimValue;
        })) {
      throw std::invalid_argument("only explicit (not named) dimensions supported");
    }
    tensor::TensorDimensions tensor_dims = {1, 1, 1, 1};
    auto n = dims.size();
    switch (n) {
      case 4:
        tensor_dims.batch_size_ = dims[n - 4].dim_value();
        [[fallthrough]];
      case 3:
        tensor_dims.num_channels_ = dims[n - 3].dim_value();
        [[fallthrough]];
      case 2:
        tensor_dims.height_ = dims[n - 2].dim_value();
        [[fallthrough]];
      case 1:
        tensor_dims.width_ = dims[n - 1].dim_value();
        break;
      default:
        throw std::logic_error("invalid number of dimensions");
    }
    return tensor_dims;
  };

  tensor::TensorDimensions tensor_dims = to_tensor_dims(tensor_shape.dim());
  tensor::TensorCP tensor_share;
  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  if (is_model_provider_) {
    tensor_share = tensor_op_factory.make_arithmetic_64_tensor_input_other(tensor_dims);
  } else {
    auto result = tensor_op_factory.make_arithmetic_64_tensor_input_my(tensor_dims);
    tensor_share = std::move(result.second);
  }
  arithmetic_tensor_map_[value_info.name()] = std::move(tensor_share);
}

void OnnxAdapter::visit_output(const ::onnx::ValueInfoProto& value_info) {
  const auto& name = value_info.name();
  auto tensor_share = get_as_arithmetic_tensor(name);
  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  if (is_model_provider_) {
    tensor_op_factory.make_arithmetic_tensor_output_other(tensor_share);
  } else {
    auto future = tensor_op_factory.make_arithmetic_64_tensor_output_my(tensor_share);
    output_futures_[name] = std::move(future);
  }
}

void OnnxAdapter::visit_gemm(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Gemm");
  assert(node.input_size() == 2);  // XXX: ignore bias for now
  assert(node.output_size() == 1);
  const auto& input_a_name = node.input(0);
  const auto& input_b_name = node.input(1);
  const auto& output_name = node.output(0);

  std::unordered_map<std::string, std::reference_wrapper<const ::onnx::AttributeProto>>
      attribute_map;
  for (const auto& attr : node.attribute()) {
    attribute_map.emplace(attr.name(), std::cref(attr));
  }
  assert(attribute_map.count("alpha") == 0);
  assert(attribute_map.count("beta") == 0);
  assert(attribute_map.count("transA") == 0);
  assert(attribute_map.count("transB") == 0);

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  const auto input_a_tensor = get_as_arithmetic_tensor(input_a_name);
  const auto input_b_tensor = get_as_arithmetic_tensor(input_b_name);
  tensor::GemmOp gemm_op;
  {
    const auto& dims_a = input_a_tensor->get_dimensions();
    gemm_op.input_A_shape_[0] = dims_a.height_;
    gemm_op.input_A_shape_[1] = dims_a.width_;
    const auto& dims_b = input_b_tensor->get_dimensions();
    gemm_op.input_B_shape_[0] = dims_b.height_;
    gemm_op.input_B_shape_[1] = dims_b.width_;
  }
  gemm_op.output_shape_ = gemm_op.compute_output_shape();
  assert(gemm_op.verify());
  const auto output_tensor =
      tensor_op_factory.make_tensor_gemm_op(gemm_op, input_a_tensor, input_b_tensor);
  arithmetic_tensor_map_[output_name] = output_tensor;
}

void OnnxAdapter::visit_conv(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Conv");
  assert(node.input_size() == 2);  // XXX: ignore bias for now
  assert(node.output_size() == 1);
  const auto& input_name = node.input(0);
  const auto& kernel_name = node.input(1);
  const auto& output_name = node.output(0);

  std::unordered_map<std::string, std::reference_wrapper<const ::onnx::AttributeProto>>
      attribute_map;
  for (const auto& attr : node.attribute()) {
    attribute_map.emplace(attr.name(), std::cref(attr));
  }
  assert(attribute_map.count("auto_pad") == 0);
  assert(attribute_map.count("group") == 0);
  assert(attribute_map.count("pads") == 1);

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  const auto input_tensor = get_as_arithmetic_tensor(input_name);
  const auto kernel_tensor = get_as_arithmetic_tensor(input_name);
  tensor::Conv2DOp conv_op;
  if (attribute_map.count("dilations") == 1) {
    auto it = attribute_map.find("dilations");
    assert(it != std::end(attribute_map));
    const auto& dilations_attr = it->second.get();
    assert(dilations_attr.name() == "dilations");
    assert(dilations_attr.has_type() && dilations_attr.type() == ::onnx::AttributeProto::INTS);
    assert(dilations_attr.ints_size() == 2);
    conv_op.dilations_[0] = dilations_attr.ints(0);
    conv_op.dilations_[1] = dilations_attr.ints(1);
  } else {
    conv_op.dilations_[0] = 1;
    conv_op.dilations_[1] = 1;
  }
  if (attribute_map.count("kernel_shape") == 1) {
    auto it = attribute_map.find("kernel_shape");
    assert(it != std::end(attribute_map));
    const auto& kernel_shape_attr = it->second.get();
    assert(kernel_shape_attr.name() == "kernel_shape");
    assert(kernel_shape_attr.has_type() &&
           kernel_shape_attr.type() == ::onnx::AttributeProto::INTS);
    assert(kernel_shape_attr.ints_size() == 4);
    conv_op.kernel_shape_[0] = kernel_shape_attr.ints(0);
    conv_op.kernel_shape_[1] = kernel_shape_attr.ints(1);
    conv_op.kernel_shape_[2] = kernel_shape_attr.ints(2);
    conv_op.kernel_shape_[3] = kernel_shape_attr.ints(3);
  }
  assert(attribute_map.count("pads") == 1);
  {
    auto it = attribute_map.find("pads");
    assert(it != std::end(attribute_map));
    const auto& pads_attr = it->second.get();
    assert(pads_attr.name() == "pads");
    assert(pads_attr.has_type() && pads_attr.type() == ::onnx::AttributeProto::INTS);
    assert(pads_attr.ints_size() == 4);
    conv_op.pads_[0] = pads_attr.ints(0);
    conv_op.pads_[1] = pads_attr.ints(1);
    conv_op.pads_[2] = pads_attr.ints(2);
    conv_op.pads_[3] = pads_attr.ints(3);
  }
  if (attribute_map.count("strides") == 1) {
    auto it = attribute_map.find("strides");
    assert(it != std::end(attribute_map));
    const auto& strides_attr = it->second.get();
    assert(strides_attr.name() == "strides");
    assert(strides_attr.has_type() && strides_attr.type() == ::onnx::AttributeProto::INTS);
    assert(strides_attr.ints_size() == 2);
    conv_op.strides_[0] = strides_attr.ints(0);
    conv_op.strides_[1] = strides_attr.ints(1);
  } else {
    conv_op.strides_[0] = 1;
    conv_op.strides_[1] = 1;
  }
  {
    const auto& input_dims = input_tensor->get_dimensions();
    conv_op.input_shape_[0] = input_dims.num_channels_;
    conv_op.input_shape_[1] = input_dims.height_;
    conv_op.input_shape_[2] = input_dims.width_;
    conv_op.output_shape_ = conv_op.compute_output_shape();
    assert(conv_op.verify());
  }
  const auto output_tensor =
      tensor_op_factory.make_tensor_conv2d_op(conv_op, input_tensor, kernel_tensor);
  arithmetic_tensor_map_[output_name] = output_tensor;
}

void OnnxAdapter::visit_mul(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Mul");
  assert(node.input_size() == 2);
  assert(node.output_size() == 1);
  assert(node.input(0) == node.input(1));  // XXX: assume squaring for now
  const auto& input_name = node.input(0);
  const auto& output_name = node.output(0);

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  const auto input_tensor = get_as_arithmetic_tensor(input_name);
  const auto output_tensor = tensor_op_factory.make_tensor_sqr_op(input_tensor);
  arithmetic_tensor_map_[output_name] = output_tensor;
}

void OnnxAdapter::visit_relu(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Relu");
  assert(node.input_size() == 1);
  assert(node.output_size() == 1);
  const auto& input_name = node.input(0);
  const auto& output_name = node.output(0);

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(boolean_protocol_);
  const auto input_tensor = get_as_boolean_tensor(input_name);
  const auto output_tensor = tensor_op_factory.make_tensor_relu_op(input_tensor);
  boolean_tensor_map_[output_name] = output_tensor;
}

void OnnxAdapter::visit_maxpool(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "MaxPool");
  assert(node.input_size() == 1);
  assert(node.output_size() == 1);
  const auto& input_name = node.input(0);
  const auto& output_name = node.output(0);

  std::unordered_map<std::string, std::reference_wrapper<const ::onnx::AttributeProto>>
      attribute_map;
  for (const auto& attr : node.attribute()) {
    attribute_map.emplace(attr.name(), std::cref(attr));
  }
  assert(attribute_map.count("auto_pad") == 0);
  assert(attribute_map.count("ceil_mode") == 0);
  assert(attribute_map.count("dilations") == 0);
  assert(attribute_map.count("kernel_shape") == 1);
  assert(attribute_map.count("pads") == 0);

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(boolean_protocol_);
  const auto input_tensor = get_as_boolean_tensor(input_name);
  tensor::MaxPoolOp maxpool_op;
  {
    auto it = attribute_map.find("kernel_shape");
    assert(it != std::end(attribute_map));
    const auto& kernel_shape_attr = it->second.get();
    assert(kernel_shape_attr.name() == "kernel_shape");
    assert(kernel_shape_attr.has_type() &&
           kernel_shape_attr.type() == ::onnx::AttributeProto::INTS);
    assert(kernel_shape_attr.ints_size() == 2);
    maxpool_op.kernel_shape_[0] = kernel_shape_attr.ints(0);
    maxpool_op.kernel_shape_[1] = kernel_shape_attr.ints(1);
  }
  if (attribute_map.count("strides") == 1) {
    auto it = attribute_map.find("strides");
    assert(it != std::end(attribute_map));
    const auto& strides_attr = it->second.get();
    assert(strides_attr.name() == "strides");
    assert(strides_attr.has_type() && strides_attr.type() == ::onnx::AttributeProto::INTS);
    assert(strides_attr.ints_size() == 2);
    maxpool_op.strides_[0] = strides_attr.ints(0);
    maxpool_op.strides_[1] = strides_attr.ints(1);
  } else {
    maxpool_op.strides_[0] = 1;
    maxpool_op.strides_[1] = 1;
  }
  {
    const auto& input_dims = input_tensor->get_dimensions();
    maxpool_op.input_shape_[0] = input_dims.num_channels_;
    maxpool_op.input_shape_[1] = input_dims.height_;
    maxpool_op.input_shape_[2] = input_dims.width_;
    maxpool_op.output_shape_ = maxpool_op.compute_output_shape();
    assert(maxpool_op.verify());
  }
  const auto output_tensor = tensor_op_factory.make_tensor_maxpool_op(maxpool_op, input_tensor);
  boolean_tensor_map_[output_name] = output_tensor;
}

void OnnxAdapter::visit_flatten(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Flatten");
  assert(node.attribute_size() == 1);
  assert(node.input_size() == 1);
  assert(node.output_size() == 1);
  const auto& input_name = node.input(0);
  const auto& output_name = node.output(0);

  const auto& axis_attr = node.attribute(0);
  assert(axis_attr.name() == "axis");
  assert(axis_attr.has_type() && axis_attr.type() == ::onnx::AttributeProto::INT);
  assert(axis_attr.has_i() && axis_attr.i() >= 0);
  auto axis = static_cast<std::size_t>(axis_attr.i());

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  const auto input_tensor = get_as_arithmetic_tensor(input_name);
  const auto output_tensor = tensor_op_factory.make_tensor_flatten_op(input_tensor, axis);
  arithmetic_tensor_map_[output_name] = output_tensor;
}

tensor::TensorCP OnnxAdapter::get_as_arithmetic_tensor(const std::string& name) {
  auto it = arithmetic_tensor_map_.find(name);
  if (it != std::end(arithmetic_tensor_map_)) {
    return it->second;
  }
  it = boolean_tensor_map_.find(name);
  if (it != std::end(boolean_tensor_map_)) {
    auto tensor = network_builder_.convert(arithmetic_protocol_, it->second);
    arithmetic_tensor_map_[name] = tensor;
    return tensor;
  }
  throw std::runtime_error(fmt::format("cannot find tensor of name: {}", name));
}

tensor::TensorCP OnnxAdapter::get_as_boolean_tensor(const std::string& name) {
  auto it = boolean_tensor_map_.find(name);
  if (it != std::end(boolean_tensor_map_)) {
    return it->second;
  }
  it = arithmetic_tensor_map_.find(name);
  if (it != std::end(arithmetic_tensor_map_)) {
    auto tensor = network_builder_.convert(boolean_protocol_, it->second);
    boolean_tensor_map_[name] = tensor;
    return tensor;
  }
  throw std::runtime_error(fmt::format("cannot find tensor of name: {}", name));
}

}  // namespace MOTION::onnx
