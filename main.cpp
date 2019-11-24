#include "measurement_class.h"
#include "matrix_converters.h"

#include "gpu_matrix_multiplier.h"

#include <cuda_runtime.h>

#include <iostream>
#include <chrono>
#include <memory>

template<typename data_type, typename index_type>
measurement_class cpu_csr_spmv_single_thread_naive (
  const csr_matrix_class<data_type, index_type> &matrix,
  data_type *x,
  data_type *y)
{
  fill_n (x, matrix.meta.cols_count, 1.0);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.data.get ();

  auto begin = std::chrono::system_clock::now ();

  for (index_type row = 0; row < matrix.meta.rows_count; row++)
    {
      const auto row_start = row_ptr[row];
      const auto row_end = row_ptr[row + 1];

      data_type dot = 0;
      for (auto element = row_start; element < row_end; element++)
        dot += data[element] * x[col_ids[element]];
      y[row] = dot;
    }

  auto end = std::chrono::system_clock::now ();
  const double elapsed = std::chrono::duration<double> (end - begin).count ();

  const size_t data_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t x_bytes = matrix.meta.non_zero_count * sizeof (data_type);
  const size_t col_ids_bytes = matrix.meta.non_zero_count * sizeof (index_type);
  const size_t row_ids_bytes = 2 * matrix.meta.rows_count * sizeof (index_type);
  const size_t y_bytes = matrix.meta.rows_count * sizeof (data_type);

  const size_t operations_count = matrix.meta.non_zero_count * 2;

  return measurement_class (
    "CPU CSR",
    elapsed,
    data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
    operations_count);
}

double size_to_gb (size_t size)
{
  return static_cast<double> (size) / 1024 / 1024/ 1024;
}

template <typename data_type, typename index_type>
void perform_measurements (
  index_type bs,
  index_type n_rows,
  index_type blocks_per_row
  )
{
  const size_t nnz = n_rows * blocks_per_row * bs * bs;
  const double matrix_and_vectors_data_size = static_cast<double> (nnz + 2 * n_rows * bs) * sizeof (data_type);

  const size_t csr_extra_data_size = (nnz + n_rows * bs + bs) * sizeof (index_type);
  const size_t bcsr_extra_data_size = (n_rows * blocks_per_row + n_rows + 1) * sizeof (index_type);

  std::cout << "Required memory: \n"
            << "\tCSR  => DATA: " << size_to_gb (matrix_and_vectors_data_size) << " GB; EXTRA: " << size_to_gb (csr_extra_data_size) << "\n"
            << "\tBCSR => DATA: " << size_to_gb (matrix_and_vectors_data_size) << " GB; EXTRA: " << size_to_gb (bcsr_extra_data_size) << std::endl;
  auto block_matrix = gen_n_diag_bcsr<data_type, index_type> (n_rows, blocks_per_row, bs);
  auto matrix = std::make_unique<csr_matrix_class<data_type, index_type>> (*block_matrix);

  auto elapsed_csr = gpu_csr_spmv<data_type, index_type> (*matrix, nullptr);
  std::cout << "GPU CSR: " << elapsed_csr.get_elapsed () << "s" << std::endl;

  auto bcsr_elapsed = gpu_bcsr_spmv<data_type, index_type> (*block_matrix, nullptr);

  for (auto &elapsed: bcsr_elapsed)
    std::cout << elapsed.get_format () << " " << elapsed.get_elapsed () << "s" << std::endl;
}

int main ()
{
  cudaSetDevice (1);
  perform_measurements<float, int> (32, 100'000, 6);

  return 0;
}