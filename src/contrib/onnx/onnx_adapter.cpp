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

#include "tensor/tensor_op_factory.h"

namespace MOTION::onnx {

OnnxAdapter::OnnxAdapter(tensor::TensorOpFactory& tensor_op_factory,
                         MPCProtocol arithmetic_protocol, MPCProtocol boolean_protocol,
                         bool is_model_provider)
    : tensor_op_factory_(tensor_op_factory),
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
  if (is_model_provider_) {
    auto result = tensor_op_factory_.make_arithmetic_64_tensor_input_my(tensor_dims);
    tensor_share = std::move(result.second);
  } else {
    tensor_share = tensor_op_factory_.make_arithmetic_64_tensor_input_other(tensor_dims);
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
  if (is_model_provider_) {
    tensor_share = tensor_op_factory_.make_arithmetic_64_tensor_input_other(tensor_dims);
  } else {
    auto result = tensor_op_factory_.make_arithmetic_64_tensor_input_my(tensor_dims);
    tensor_share = std::move(result.second);
  }
  arithmetic_tensor_map_[value_info.name()] = std::move(tensor_share);
}

void OnnxAdapter::visit_output(const ::onnx::ValueInfoProto& value_info) {
  const auto& name = value_info.name();
  auto tensor_share = get_as_arithmetic_tensor(name);
  if (is_model_provider_) {
    tensor_op_factory_.make_arithmetic_tensor_output_other(tensor_share);
  } else {
    auto future = tensor_op_factory_.make_arithmetic_64_tensor_output_my(tensor_share);
    output_futures_[name] = std::move(future);
  }
}

tensor::TensorCP OnnxAdapter::get_as_arithmetic_tensor(const std::string& name) {
  auto it = arithmetic_tensor_map_.find(name);
  if (it != std::end(arithmetic_tensor_map_)) {
    return it->second;
  }
  it = boolean_tensor_map_.find(name);
  if (it != std::end(boolean_tensor_map_)) {
    // TODO: convert boolean -> arithmetic
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
    // TODO: convert arithmetic -> boolean
  }
  throw std::runtime_error(fmt::format("cannot find tensor of name: {}", name));
}

}  // namespace MOTION::onnx
