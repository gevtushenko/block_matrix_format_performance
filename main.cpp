#include <iostream>
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
  const std::unique_ptr<index_type> columns;
  const std::unique_ptr<index_type> row_ptr;
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
  std::fill_n (matrix->values.get (), nnzb_arg * bs_arg * bs_arg, 1.0);

  index_type block_id = 0;
  auto row_ptr = matrix->row_ptr.get ();
  auto columns = matrix->columns.get ();
  for (index_type row = 0; row < n_rows_arg; row++)
    {
      row_ptr[row] = row * blocks_per_row;

      const index_type first_column = row > blocks_per_row / 2 ? row - blocks_per_row / 2 : 0;
      for (index_type element = 0; element < blocks_per_row; element++)
        columns[block_id++] = first_column + element;
    }
  row_ptr[n_rows_arg] = n_rows_arg * blocks_per_row;

  return matrix;
}

int main ()
{
  auto matrix = gen_n_diag_bcsr<float, int> (8, 5, 2);
  return 0;
}