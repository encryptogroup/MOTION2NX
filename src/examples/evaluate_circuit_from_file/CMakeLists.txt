add_executable(evaluate_circuit_from_file evaluate_circuit_from_file.cpp)

find_package(Boost
        COMPONENTS
        program_options
        REQUIRED)

target_compile_features(evaluate_circuit_from_file PRIVATE cxx_std_20)

target_link_libraries(evaluate_circuit_from_file
        MOTION::motion
        Boost::program_options
        )
