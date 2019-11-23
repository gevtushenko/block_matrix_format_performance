#include "measurement_class.h"

#include <iostream>
#include <chrono>
#include <memory>

template <typename data_type, typename index_type>
class bcsr_matrix_class
{
public:
  bcsr_matrix_class (
      index_type n_rows_arg,
      index_type n_cols_arg,
      index_type bs_arg,
      index_type nnzb_arg)
    : n_rows (n_rows_arg)
    , n_cols (n_cols_arg)
    , bs (bs_arg)
    , nnzb (nnzb_arg)
    , values (new data_type[nnzb * bs * bs])
    , columns (new index_type[nnzb])
    , row_ptr (new index_type[n_rows + 1])
  {
  }

public:
  const index_type n_rows {};
  const index_type n_cols {};

  const index_type bs {};
  const index_type nnzb {};

  const std::unique_ptr<data_type> values;
  const std::unique_ptr<index_type[]> columns;
  const std::unique_ptr<index_type[]> row_ptr;
};

template <typename data_type, typename index_type>
class csr_matrix_class
{
public:
  explicit csr_matrix_class (const bcsr_matrix_class<data_type, index_type> &matrix)
    : n_rows (matrix.n_rows)
    , n_cols (matrix.n_cols)
    , nnz (matrix.nnzb * matrix.bs * matrix.bs)
    , values (new data_type[nnz])
    , columns (new index_type[nnz])
    , row_ptr (new index_type[n_rows + 1])
  {
    auto brow_ptr = matrix.row_ptr.get ();
    for (index_type row = 0; row <= n_rows; row++)
      row_ptr[row] = brow_ptr[row] * matrix.bs;
  }

public:
  const index_type n_rows {};
  const index_type n_cols {};

  const index_type nnz {};

  const std::unique_ptr<data_type> values;
  const std::unique_ptr<index_type[]> columns;
  const std::unique_ptr<index_type[]> row_ptr;
};

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

  const size_t operations_count = matrix.meta.non_zero_count * 2; // + and * per element

  return measurement_class (
    "CPU CSR",
    elapsed,
    data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
    operations_count);
}

template <typename data_type, typename index_type>
std::unique_ptr<bcsr_matrix_class<data_type, index_type>> gen_n_diag_bcsr (
  index_type n_rows_arg,
  index_type blocks_per_row,
  index_type bs_arg)
{
  const index_type nnzb_arg = blocks_per_row * n_rows_arg;
  std::unique_ptr<bcsr_matrix_class<data_type, index_type>> matrix (
    new bcsr_matrix_class<data_type, index_type> (
      n_rows_arg, n_rows_arg, bs_arg, nnzb_arg));

  index_type block_id = 0;
  auto row_ptr = matrix->row_ptr.get ();
  auto columns = matrix->columns.get ();
  auto values = matrix->values.get ();
  for (index_type row = 0; row < n_rows_arg; row++)
    {
      row_ptr[row] = row * blocks_per_row;

      const index_type first_column = row > blocks_per_row / 2 ? row - blocks_per_row / 2 : 0;
      for (index_type element = 0; element < blocks_per_row; element++)
        {
          const index_type element_column = first_column + element;
          auto block_data = values + block_id * bs_arg * bs_arg;
          for (unsigned int i = 0; i < bs_arg * bs_arg; i++)
            block_data[i] = (static_cast<data_type> (element_column) + i) / n_rows_arg;
          columns[block_id++] = element_column;
        }
    }
  row_ptr[n_rows_arg] = n_rows_arg * blocks_per_row;

  return matrix;
}

int main ()
{
  auto block_matrix = gen_n_diag_bcsr<float, int> (8, 5, 2);
  auto matrix = std::make_unique<csr_matrix_class<float, int>> (*block_matrix);
  return 0;
}