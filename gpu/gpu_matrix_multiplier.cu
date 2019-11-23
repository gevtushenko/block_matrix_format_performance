#include "gpu_matrix_multiplier.h"

template <typename data_type, typename index_type>
__global__ void csr_spmv_kernel (
  index_type n_rows,
  const index_type *col_ids,
  const index_type *row_ptr,
  const data_type *data,
  const data_type *x,
  data_type *y)
{
  index_type row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n_rows)
    {
      const index_type row_start = row_ptr[row];
      const index_type row_end = row_ptr[row + 1];

      data_type sum = 0;
      for (index_type element = row_start; element < row_end; element++)
        sum += data[element] * x[col_ids[element]];
      y[row] = sum;
    }
}

template <typename data_type, typename index_type>
measurement_class gpu_csr_spmv (
  const csr_matrix_class<data_type, index_type> &matrix,
  const data_type *reference_y)
{
  const index_type matrix_size = matrix.nnz;
  const index_type columns_size = matrix_size;
  const index_type row_ptr_size = matrix.n_rows + 1;
  const index_type x_size = matrix.n_cols;
  const index_type y_size = matrix.n_rows;

  data_type *d_values {};
  data_type *d_y {};
  data_type *d_x {};

  index_type *d_row_ptr {};
  index_type *d_columns {};

  cudaMalloc (&d_values, matrix_size * sizeof (data_type));
  cudaMalloc (&d_x, x_size * sizeof (data_type));
  cudaMalloc (&d_y, y_size * sizeof (data_type));

  cudaMalloc (&d_row_ptr, row_ptr_size * sizeof (index_type));
  cudaMalloc (&d_columns, columns_size * sizeof (index_type));

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;

    csr_spmv_kernel<data_type, index_type> <<<grid_size, block_size>>> (matrix.n_rows, d_columns, d_row_ptr, d_values, d_x, d_y);
  }

  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);
  const double elapsed = milliseconds / 1000;

  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  cudaFree (d_values);
  cudaFree (d_x);
  cudaFree (d_y);
  cudaFree (d_row_ptr);
  cudaFree (d_columns);

  return measurement_class ("GPU CSR", elapsed, 0, 0);
}

#define INSTANTIATE(DTYPE,ITYPE) template measurement_class gpu_csr_spmv (const csr_matrix_class<DTYPE, ITYPE> &matrix, const DTYPE *reference_y);

INSTANTIATE (float,int)

#undef INSTANTIATE
