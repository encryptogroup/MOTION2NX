add_executable(cryptonets cryptonets.cpp)
target_compile_features(cryptonets PRIVATE cxx_std_17)

find_package(Boost COMPONENTS program_options REQUIRED)

target_link_libraries(cryptonets
  MOTION::motion
  Boost::program_options
)
