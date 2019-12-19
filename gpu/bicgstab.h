//
// Created by egi on 12/4/19.
//

#ifndef BLOCK_MATRIX_FORMAT_PERFORMANCE_BICGSTAB_H
#define BLOCK_MATRIX_FORMAT_PERFORMANCE_BICGSTAB_H

#include <memory>

#include "matrix_converters.h"

template <class T, class C=std::size_t>
class gpu_bicgstab
{
public:
  gpu_bicgstab () = delete;
  explicit gpu_bicgstab (const csr_matrix_class<T, C> &A_b, bool use_precond);
  ~gpu_bicgstab ();

  T *solve (const csr_matrix_class<T, C> &A, const T* b, T epsilon, unsigned int max_iterations);

private:
  unsigned int n_rows = 0;
  const C n_elements = 0;

  /// Host data
  T *h_x = nullptr; /// Pinned memory

  /// Device data
  T *x = nullptr;
  T *r = nullptr;
  T *v = nullptr;
  T *p = nullptr;
  T *h = nullptr;
  T *s = nullptr;
  T *t = nullptr;
  T *rh = nullptr;
  T *tmp = nullptr;
  T *rhs = nullptr;
  T *A = nullptr;
  C *column_indices = nullptr; // TODO Move to matrix
  C *offsets_to_rows_begin = nullptr;

  T *P = nullptr;
  T *q = nullptr;
  T *z = nullptr;
};

#endif //BLOCK_MATRIX_FORMAT_PERFORMANCE_BICGSTAB_H
