//
// Created by egi on 11/24/19.
//

#ifndef BLOCK_MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTERS_H
#define BLOCK_MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTERS_H

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

  const std::unique_ptr<data_type[]> values;
  const std::unique_ptr<index_type[]> columns;
  const std::unique_ptr<index_type[]> row_ptr;
};

template <typename data_type, typename index_type>
class csr_matrix_class
{
public:
  explicit csr_matrix_class (const bcsr_matrix_class<data_type, index_type> &matrix)
    : n_rows (matrix.n_rows * matrix.bs)
    , n_cols (matrix.n_cols * matrix.bs)
    , nnz (matrix.nnzb * matrix.bs * matrix.bs)
    , values (new data_type[nnz])
    , columns (new index_type[nnz])
    , row_ptr (new index_type[n_rows + 1])
  {
    size_t offset = 0;
    for (index_type block_row = 0; block_row < matrix.n_rows; block_row++)
      {
        for (index_type row = 0; row < matrix.bs; row++)
          {
            row_ptr[block_row * matrix.bs + row] = offset;
            for (index_type block = matrix.row_ptr[block_row]; block < matrix.row_ptr[block_row + 1]; block++)
              {
                for (index_type column = 0; column < matrix.bs; column++)
                  {
                    index_type actual_column = matrix.columns[block] * matrix.bs + column;
                    data_type value = matrix.values[block * matrix.bs * matrix.bs + row * matrix.bs + column];

                    columns[offset] = actual_column;
                    values[offset++] = value;
                  }
              }
          }
      }

    row_ptr[n_rows] = offset;
  }

public:
  const index_type n_rows {};
  const index_type n_cols {};

  const index_type nnz {};

  const std::unique_ptr<data_type[]> values;
  const std::unique_ptr<index_type[]> columns;
  const std::unique_ptr<index_type[]> row_ptr;
};

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

#endif //BLOCK_MATRIX_FORMAT_PERFORMANCE_MATRIX_CONVERTERS_H
