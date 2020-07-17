add_executable(benchmark_garbling benchmark_garbling.cpp)
target_compile_features(benchmark_garbling PRIVATE cxx_std_17)

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(benchmark REQUIRED)

target_link_libraries(benchmark_garbling
  MOTION::motion
  Boost::program_options
  benchmark::benchmark_main
  benchmark::benchmark
)
