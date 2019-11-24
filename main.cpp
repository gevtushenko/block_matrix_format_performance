#include "measurement_class.h"
#include "matrix_converters.h"

#include "gpu_matrix_multiplier.h"

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

int main ()
{
  const unsigned int n_rows = 100'000;
  auto block_matrix = gen_n_diag_bcsr<float, int> (n_rows, 6, 16);
  auto matrix = std::make_unique<csr_matrix_class<float, int>> (*block_matrix);

  auto elapsed_csr = gpu_csr_spmv<float, int> (*matrix, nullptr);
  std::cout << "GPU CSR: " << elapsed_csr.get_elapsed () << "s" << std::endl;

  auto bcsr_elapsed = gpu_bcsr_spmv<float, int> (*block_matrix, nullptr);

  for (auto &elapsed: bcsr_elapsed)
    std::cout << elapsed.get_format () << " " << elapsed.get_elapsed () << "s" << std::endl;

  return 0;
}