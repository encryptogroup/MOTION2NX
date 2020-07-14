add_library(motion_onnx
  onnx_adapter.cpp
  onnx_visitor.cpp
)
target_compile_features(motion_onnx PRIVATE cxx_std_17)
target_include_directories(motion_onnx PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(motion_onnx PRIVATE FMT_HEADER_ONLY)
add_library(MOTION::motion_onnx ALIAS motion_onnx)


find_package(ONNX REQUIRED)
find_package(Protobuf REQUIRED)

target_link_libraries(motion_onnx PUBLIC motion)
target_link_libraries(motion_onnx PRIVATE onnx)
