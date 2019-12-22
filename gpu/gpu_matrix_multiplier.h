//
// Created by egi on 11/24/19.
//

#ifndef BLOCK_MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H
#define BLOCK_MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H

#include <vector>

#include "matrix_converters.h"
#include "measurement_class.h"

template <typename data_type, typename index_type>
measurement_class gpu_csr_spmv (
  const csr_matrix_class<data_type, index_type> &matrix,
  const data_type *reference_y);

template <typename data_type, typename index_type>
measurement_class gpu_csr_vector_spmv (
  const csr_matrix_class<data_type, index_type> &matrix,
  const data_type *reference_y);

template <typename data_type, typename index_type>
std::vector<measurement_class> gpu_bcsr_spmv (
  bcsr_matrix_class<data_type, index_type> &matrix,
  const data_type *transpose_matrix_data,
  const data_type *reference_y);

#endif //BLOCK_MATRIX_FORMAT_PERFORMANCE_GPU_MATRIX_MULTIPLIER_H
