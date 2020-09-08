add_executable(benchmark_nn_layers benchmark_nn_layers.cpp)
target_compile_features(benchmark_nn_layers PRIVATE cxx_std_17)

find_package(Boost COMPONENTS program_options REQUIRED)

target_link_libraries(benchmark_nn_layers
  MOTION::motion
  Boost::program_options
)
