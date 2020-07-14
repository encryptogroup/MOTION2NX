add_executable(onnx2motion onnx2motion.cpp)
target_compile_features(onnx2motion PRIVATE cxx_std_17)

find_package(Boost COMPONENTS program_options REQUIRED)

target_link_libraries(onnx2motion
  MOTION::motion
  MOTION::motion_onnx
  Boost::program_options
)
