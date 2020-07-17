add_executable(benchmark_operations benchmark_main.cpp common/benchmark.cpp)

find_package(Boost
        COMPONENTS
        program_options
        REQUIRED)

target_compile_features(benchmark_operations PRIVATE cxx_std_17)

target_link_libraries(benchmark_operations
        MOTION::motion
        Boost::program_options
        )
