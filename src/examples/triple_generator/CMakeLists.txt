add_executable(triple_generator
  main.cpp
  ot_backend.cpp
  triple_generator.cpp
)
target_compile_features(triple_generator PRIVATE cxx_std_17)

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(triple_generator
  MOTION::motion
  Boost::program_options
  OpenMP::OpenMP_CXX
)
