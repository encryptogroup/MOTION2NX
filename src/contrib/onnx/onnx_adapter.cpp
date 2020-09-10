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

#include "onnx_adapter.h"

#include <exception>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>
#include <onnx/onnx_pb.h>

#include "tensor/network_builder.h"
#include "tensor/tensor_op.h"
#include "tensor/tensor_op_factory.h"

namespace MOTION::onnx {

struct OnnxAdapter::OnnxAdapter::OnnxAdapterImpl {
  ::onnx::ModelProto model;
};

OnnxAdapter::OnnxAdapter(tensor::NetworkBuilder& network_builder, MPCProtocol arithmetic_protocol,
                         MPCProtocol boolean_protocol, std::size_t bit_size,
                         std::size_t fractional_bits, bool is_model_provider)
    : network_builder_(network_builder),
      arithmetic_protocol_(arithmetic_protocol),
      boolean_protocol_(boolean_protocol),
      bit_size_(bit_size),
      fractional_bits_(fractional_bits),
      is_model_provider_(is_model_provider),
      impl_(std::make_unique<OnnxAdapterImpl>()) {
  if (bit_size_ != 64 && bit_size_ != 32) {
    throw std::invalid_argument(fmt::format("unsupported bit size: {}", bit_size_));
  }
}

OnnxAdapter::~OnnxAdapter() = default;

void OnnxAdapter::load_model(const std::string& path) {
  {
    std::ifstream in(path, std::ios_base::binary);
    impl_->model.ParseFromIstream(&in);
  }
  visit_model(impl_->model);
}

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
  if (bit_size_ == 64) {
    if (is_model_provider_) {
      auto result = tensor_op_factory.make_arithmetic_64_tensor_input_my(tensor_dims);
      tensor_share = std::move(result.second);
      input_promises_64_.emplace(tensor.name(),
                                 std::make_pair(tensor_dims, std::move(result.first)));
    } else {
      tensor_share = tensor_op_factory.make_arithmetic_64_tensor_input_other(tensor_dims);
    }
  } else {
    if (is_model_provider_) {
      auto result = tensor_op_factory.make_arithmetic_32_tensor_input_my(tensor_dims);
      tensor_share = std::move(result.second);
      input_promises_32_.emplace(tensor.name(),
                                 std::make_pair(tensor_dims, std::move(result.first)));
    } else {
      tensor_share = tensor_op_factory.make_arithmetic_32_tensor_input_other(tensor_dims);
    }
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
  if (bit_size_ == 64) {
    if (is_model_provider_) {
      tensor_share = tensor_op_factory.make_arithmetic_64_tensor_input_other(tensor_dims);
    } else {
      auto result = tensor_op_factory.make_arithmetic_64_tensor_input_my(tensor_dims);
      tensor_share = std::move(result.second);
      input_promises_64_.emplace(value_info.name(),
                                 std::make_pair(tensor_dims, std::move(result.first)));
    }
  } else {
    if (is_model_provider_) {
      tensor_share = tensor_op_factory.make_arithmetic_32_tensor_input_other(tensor_dims);
    } else {
      auto result = tensor_op_factory.make_arithmetic_32_tensor_input_my(tensor_dims);
      tensor_share = std::move(result.second);
      input_promises_32_.emplace(value_info.name(),
                                 std::make_pair(tensor_dims, std::move(result.first)));
    }
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
    if (bit_size_ == 64) {
      auto future = tensor_op_factory.make_arithmetic_64_tensor_output_my(tensor_share);
      output_futures_64_[name] = std::make_pair(tensor_share->get_dimensions(), std::move(future));
    } else {
      auto future = tensor_op_factory.make_arithmetic_32_tensor_output_my(tensor_share);
      output_futures_32_[name] = std::make_pair(tensor_share->get_dimensions(), std::move(future));
    }
  }
}

void OnnxAdapter::visit_gemm(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Gemm");
  assert(node.input_size() == 2 || node.input_size() == 3);
  assert(node.output_size() == 1);
  const auto& input_a_name = node.input(0);
  const auto& input_b_name = node.input(1);
  const auto& output_name = node.output(0);

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  const auto input_a_tensor = get_as_arithmetic_tensor(input_a_name);
  const auto input_b_tensor = get_as_arithmetic_tensor(input_b_name);
  tensor::TensorCP input_c_tensor = nullptr;
  if (node.input_size() == 3) {
    const auto& input_c_name = node.input(2);
    input_c_tensor = get_as_arithmetic_tensor(input_c_name);
  }

  std::unordered_map<std::string, std::reference_wrapper<const ::onnx::AttributeProto>>
      attribute_map;
  for (const auto& attr : node.attribute()) {
    attribute_map.emplace(attr.name(), std::cref(attr));
  }

  tensor::GemmOp gemm_op;
  if (attribute_map.count("alpha") == 1) {
    auto it = attribute_map.find("alpha");
    assert(it != std::end(attribute_map));
    const auto& alpha_attr = it->second.get();
    assert(alpha_attr.name() == "alpha");
    assert(alpha_attr.has_type() && alpha_attr.type() == ::onnx::AttributeProto::FLOAT);
    gemm_op.alpha_ = alpha_attr.f();
  }
  if (attribute_map.count("beta") == 1) {
    auto it = attribute_map.find("beta");
    assert(it != std::end(attribute_map));
    const auto& beta_attr = it->second.get();
    assert(beta_attr.name() == "beta");
    assert(beta_attr.has_type() && beta_attr.type() == ::onnx::AttributeProto::FLOAT);
    gemm_op.beta_ = beta_attr.f();
  }
  if (attribute_map.count("transB") == 1) {
    auto it = attribute_map.find("transB");
    assert(it != std::end(attribute_map));
    const auto& transB_attr = it->second.get();
    assert(transB_attr.name() == "transB");
    assert(transB_attr.has_type() && transB_attr.type() == ::onnx::AttributeProto::INT);
    gemm_op.transB_ = (transB_attr.i() != 0);
  }
  if (attribute_map.count("transA") == 1) {
    auto it = attribute_map.find("transA");
    assert(it != std::end(attribute_map));
    const auto& transA_attr = it->second.get();
    assert(transA_attr.name() == "transA");
    assert(transA_attr.has_type() && transA_attr.type() == ::onnx::AttributeProto::INT);
    gemm_op.transA_ = (transA_attr.i() != 0);
  }
  if (attribute_map.count("transB") == 1) {
    auto it = attribute_map.find("transB");
    assert(it != std::end(attribute_map));
    const auto& transB_attr = it->second.get();
    assert(transB_attr.name() == "transB");
    assert(transB_attr.has_type() && transB_attr.type() == ::onnx::AttributeProto::INT);
    gemm_op.transB_ = (transB_attr.i() != 0);
  }
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
  const auto output_tensor = tensor_op_factory.make_tensor_gemm_op(
      gemm_op, input_a_tensor, input_b_tensor, fractional_bits_);
  arithmetic_tensor_map_[output_name] = output_tensor;
}

void OnnxAdapter::visit_conv(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Conv");
  assert(node.input_size() == 2 || node.input_size() == 3);
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
  assert(attribute_map.count("pads") == 1);

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  const auto input_tensor = get_as_arithmetic_tensor(input_name);
  const auto kernel_tensor = get_as_arithmetic_tensor(kernel_name);
  tensor::TensorCP bias_tensor = nullptr;
  if (node.input_size() == 3) {
    const auto& bias_name = node.input(2);
    bias_tensor = get_as_arithmetic_tensor(bias_name);
  }
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
  if (attribute_map.count("group") == 1) {
    auto it = attribute_map.find("group");
    assert(it != std::end(attribute_map));
    const auto& group_attr = it->second.get();
    assert(group_attr.name() == "group");
    assert(group_attr.has_type() && group_attr.type() == ::onnx::AttributeProto::INT);
    assert(group_attr.i() == 1);
  }
  {
    const auto& kernel_dims = kernel_tensor->get_dimensions();
    conv_op.kernel_shape_ = {kernel_dims.batch_size_, kernel_dims.num_channels_,
                             kernel_dims.height_, kernel_dims.width_};
  }
  if (attribute_map.count("kernel_shape") == 1) {
    auto it = attribute_map.find("kernel_shape");
    assert(it != std::end(attribute_map));
    const auto& kernel_shape_attr = it->second.get();
    assert(kernel_shape_attr.name() == "kernel_shape");
    assert(kernel_shape_attr.has_type() &&
           kernel_shape_attr.type() == ::onnx::AttributeProto::INTS);
    assert(kernel_shape_attr.ints_size() == 2);
    assert(conv_op.kernel_shape_[2] == kernel_shape_attr.ints(0));
    assert(conv_op.kernel_shape_[3] == kernel_shape_attr.ints(1));
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
  const auto output_tensor = tensor_op_factory.make_tensor_conv2d_op(
      conv_op, input_tensor, kernel_tensor, fractional_bits_);
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
  const auto output_tensor = tensor_op_factory.make_tensor_sqr_op(input_tensor, fractional_bits_);
  arithmetic_tensor_map_[output_name] = output_tensor;
}

void OnnxAdapter::visit_relu(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Relu");
  assert(node.input_size() == 1);
  assert(node.output_size() == 1);
  const auto& input_name = node.input(0);
  const auto& output_name = node.output(0);

  const bool use_mixed_protocol_relu = [this, &input_name, &output_name] {
    // check if input is available in arithmetic sharing
    if (arithmetic_tensor_map_.count(input_name) == 0) {
      return false;
    }
    // check if output is needed only in arithmetic sharing
    const auto& graph = impl_->model.graph();
    for (const auto& node : graph.node()) {
      for (const auto& input : node.input()) {
        if (input == output_name) {
          const auto& op_type = node.op_type();
          // assume flatten only appear in front of Gemm
          if (op_type != "Gemm" && op_type != "Mul" && op_type != "Conv" && op_type != "Flatten") {
            return false;
          }
        }
      }
    }
    return true;
  }();

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(boolean_protocol_);
  const auto input_tensor = get_as_boolean_tensor(input_name);
  if (use_mixed_protocol_relu) {
    try {
      const auto input_arith_tensor = get_as_arithmetic_tensor(input_name);
      const auto output_tensor =
          tensor_op_factory.make_tensor_relu_op(input_tensor, input_arith_tensor);
      arithmetic_tensor_map_[output_name] = output_tensor;
      return;
    } catch (std::exception&) {
      // operation not supported
    }
  }
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
  if (attribute_map.count("pads") == 1) {
    auto it = attribute_map.find("pads");
    assert(it != std::end(attribute_map));
    const auto& pads_attr = it->second.get();
    assert(pads_attr.name() == "pads");
    assert(pads_attr.has_type() && pads_attr.type() == ::onnx::AttributeProto::INTS);
    assert(pads_attr.ints_size() == 4);
    assert(pads_attr.ints(0) == 0);
    assert(pads_attr.ints(1) == 0);
    assert(pads_attr.ints(2) == 0);
    assert(pads_attr.ints(3) == 0);
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

void OnnxAdapter::visit_avgpool(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "AveragePool");
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

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  const auto input_tensor = get_as_arithmetic_tensor(input_name);
  tensor::AveragePoolOp avgpool_op;
  {
    auto it = attribute_map.find("kernel_shape");
    assert(it != std::end(attribute_map));
    const auto& kernel_shape_attr = it->second.get();
    assert(kernel_shape_attr.name() == "kernel_shape");
    assert(kernel_shape_attr.has_type() &&
           kernel_shape_attr.type() == ::onnx::AttributeProto::INTS);
    if (kernel_shape_attr.ints_size() == 2) {
      avgpool_op.kernel_shape_[0] = kernel_shape_attr.ints(0);
      avgpool_op.kernel_shape_[1] = kernel_shape_attr.ints(1);
    } else if (kernel_shape_attr.ints_size() == 3) {
      assert(kernel_shape_attr.ints(0) == 1);
      avgpool_op.kernel_shape_[0] = kernel_shape_attr.ints(1);
      avgpool_op.kernel_shape_[1] = kernel_shape_attr.ints(2);
    } else {
      throw std::runtime_error("unexpected kernel dimension");
    }
  }
  if (attribute_map.count("pads") == 1) {
    auto it = attribute_map.find("pads");
    assert(it != std::end(attribute_map));
    const auto& pads_attr = it->second.get();
    assert(pads_attr.name() == "pads");
    assert(pads_attr.has_type() && pads_attr.type() == ::onnx::AttributeProto::INTS);
    assert(pads_attr.ints_size() == 4);
    assert(pads_attr.ints(0) == 0);
    assert(pads_attr.ints(1) == 0);
    assert(pads_attr.ints(2) == 0);
    assert(pads_attr.ints(3) == 0);
  }
  if (attribute_map.count("strides") == 1) {
    auto it = attribute_map.find("strides");
    assert(it != std::end(attribute_map));
    const auto& strides_attr = it->second.get();
    assert(strides_attr.name() == "strides");
    assert(strides_attr.has_type() && strides_attr.type() == ::onnx::AttributeProto::INTS);
    assert(strides_attr.ints_size() == 2);
    avgpool_op.strides_[0] = strides_attr.ints(0);
    avgpool_op.strides_[1] = strides_attr.ints(1);
  } else {
    avgpool_op.strides_[0] = 1;
    avgpool_op.strides_[1] = 1;
  }
  {
    const auto& input_dims = input_tensor->get_dimensions();
    avgpool_op.input_shape_[0] = input_dims.num_channels_;
    avgpool_op.input_shape_[1] = input_dims.height_;
    avgpool_op.input_shape_[2] = input_dims.width_;
    avgpool_op.output_shape_ = avgpool_op.compute_output_shape();
    assert(avgpool_op.verify());
  }
  const auto output_tensor =
      tensor_op_factory.make_tensor_avgpool_op(avgpool_op, input_tensor, fractional_bits_);
  arithmetic_tensor_map_[output_name] = output_tensor;
}

void OnnxAdapter::visit_flatten(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Flatten");
  assert(node.input_size() == 1);
  assert(node.output_size() == 1);
  const auto& input_name = node.input(0);
  const auto& output_name = node.output(0);

  std::unordered_map<std::string, std::reference_wrapper<const ::onnx::AttributeProto>>
      attribute_map;
  for (const auto& attr : node.attribute()) {
    attribute_map.emplace(attr.name(), std::cref(attr));
  }

  std::size_t axis = 1;
  if (attribute_map.count("axis") == 1) {
    const auto& axis_attr = node.attribute(0);
    assert(axis_attr.name() == "axis");
    assert(axis_attr.has_type() && axis_attr.type() == ::onnx::AttributeProto::INT);
    assert(axis_attr.has_i() && axis_attr.i() >= 0);
    axis = static_cast<std::size_t>(axis_attr.i());
  }

  auto& tensor_op_factory = network_builder_.get_tensor_op_factory(arithmetic_protocol_);
  const auto input_tensor = get_as_arithmetic_tensor(input_name);
  const auto output_tensor = tensor_op_factory.make_tensor_flatten_op(input_tensor, axis);
  arithmetic_tensor_map_[output_name] = output_tensor;
}

void OnnxAdapter::visit_dropout(const ::onnx::NodeProto& node) {
  assert(node.op_type() == "Dropout");
  assert(node.input_size() >= 1);
  assert(node.output_size() >= 1);
  const auto& input_name = node.input(0);
  const auto& output_name = node.output(0);

  if (arithmetic_tensor_map_.count(input_name) == 1) {
    arithmetic_tensor_map_[output_name] = arithmetic_tensor_map_[input_name];
  }
  if (boolean_tensor_map_.count(input_name) == 1) {
    boolean_tensor_map_[output_name] = boolean_tensor_map_[input_name];
  }
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
