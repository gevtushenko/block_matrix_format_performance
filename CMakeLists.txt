cmake_minimum_required(VERSION 3.15)
project(block_matrix_format_performance)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(gpu)
add_subdirectory(common)
add_subdirectory(external/fmt)
add_subdirectory(external/cuda_jit)

include_directories(external)

add_executable(block_matrix_format_performance main.cpp fem_2d/golden_gate_bridge.h)
target_link_libraries(block_matrix_format_performance common gpu fmt)