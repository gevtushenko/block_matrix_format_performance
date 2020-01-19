project(gpu)
set(CMAKE_CXX_STANDARD 17)

set(GPU_SOURCES
        gpu_matrix_multiplier.h
        gpu_matrix_multiplier.cu
        bicgstab.h
        bicgstab.cu)

find_package(CUDA QUIET)

# set cuda debug flags
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -gencode arch=compute_75,code=sm_75")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -lineinfo -gencode arch=compute_75,code=sm_75")
endif()

cuda_add_library(gpu ${GPU_SOURCES})
target_include_directories(gpu PUBLIC . ${CUDA_INCLUDE_DIRS} ../external/int_fastdiv)
target_link_libraries(gpu common ${CUDA_cusparse_LIBRARY} cuda_jit)
