if (NOT DEFINED MOTION_HYCC_PATH OR NOT EXISTS "${MOTION_HYCC_PATH}")
  message(FATAL_ERROR "Path MOTION_HYCC_PATH='${MOTION_HYCC_PATH}' does not exist")
endif ()

get_filename_component(MOTION_HYCC_PATH "${MOTION_HYCC_PATH}"
                       REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
message("Using HyCC from '${MOTION_HYCC_PATH}'")

add_library(motion_hycc
  hycc_adapter.cpp
)
target_compile_features(motion_hycc PRIVATE cxx_std_17)
target_include_directories(motion_hycc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(motion_hycc PRIVATE FMT_HEADER_ONLY)
add_library(MOTION::motion_hycc ALIAS motion_hycc)

if (CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(motion_hycc PRIVATE "-femit-class-debug-always")
endif ()

add_library(hycc_libcircuit STATIC IMPORTED)
set_target_properties(hycc_libcircuit PROPERTIES
  IMPORTED_LOCATION "${MOTION_HYCC_PATH}/src/libcircuit/libcircuit.a"
  INTERFACE_INCLUDE_DIRECTORIES ${MOTION_HYCC_PATH}/src
)

target_link_libraries(motion_hycc PUBLIC motion)
target_link_libraries(motion_hycc PRIVATE hycc_libcircuit)
