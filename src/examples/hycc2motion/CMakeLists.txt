add_executable(hycc2motion hycc2motion.cpp)
target_compile_features(hycc2motion PRIVATE cxx_std_17)

find_package(Boost COMPONENTS program_options REQUIRED)

target_link_libraries(hycc2motion
  MOTION::motion
  MOTION::motion_hycc
  Boost::program_options
)
