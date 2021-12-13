add_executable(millionaires_problem millionaires_problem.cpp)

find_package(Boost COMPONENTS json log program_options REQUIRED)

target_compile_features(millionaires_problem PRIVATE cxx_std_20)

target_link_libraries(millionaires_problem
    MOTION::motion
    Boost::json
    Boost::log
    Boost::program_options
)
