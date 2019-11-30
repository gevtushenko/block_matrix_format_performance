#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "gpu_matrix_multiplier.h"

template<class T>
struct shared_memory
{
  __device__ inline operator T *()
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

template<>
struct shared_memory<double>
{
  __device__ inline operator double *()
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const
  {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

#define FULL_WARP_MASK 0xFFFFFFFF

template <class T>
__device__ T warp_reduce (T val)
{
  /**
   *  For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset) gets
   *  the value of the val variable from the thread at lane X+offset of the same warp.
   *  The data exchange is performed between registers, and more efficient than going
   *  through shared memory, which requires a load, a store and an extra register to
   *  hold the address.
   */
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync (FULL_WARP_MASK, val, offset);

  return val;
}

template <typename data_type>
__global__ void fill_vector (unsigned int n, data_type *vec, data_type value)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    vec[i] = value;
}

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

  cudaMemcpy (d_values, matrix.values.get (), matrix_size * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (d_columns, matrix.columns.get (), columns_size * sizeof (index_type), cudaMemcpyHostToDevice);
  cudaMemcpy (d_row_ptr, matrix.row_ptr.get (), row_ptr_size * sizeof (index_type), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, d_x, 1.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (matrix.n_rows + block_size.x - 1) / block_size.x;

    csr_spmv_kernel<data_type, index_type> <<<grid_size, block_size>>> (matrix.n_rows, d_columns, d_row_ptr, d_values, d_x, d_y);
  }

  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);
  const double elapsed = milliseconds / 1000;

  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  std::unique_ptr<data_type[]> cpu_y (new data_type[y_size]);
  cudaMemcpy (cpu_y.get (), d_y, y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reference_y, cpu_y.get ());

  cudaFree (d_values);
  cudaFree (d_x);
  cudaFree (d_y);
  cudaFree (d_row_ptr);
  cudaFree (d_columns);

  return measurement_class ("GPU CSR", elapsed, 0, 0);
}

template <typename data_type, typename index_type>
__global__ void csr_spmv_vector_kernel (
  index_type n_rows,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type * __restrict__ y)
{
  const index_type thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type warp_id = thread_id / 32;
  const index_type lane = thread_id % 32;

  const index_type row = warp_id; ///< One warp per row

  data_type dot = 0;
  if (row < n_rows)
    {
      const index_type row_start = row_ptr[row];
      const index_type row_end = row_ptr[row + 1];

      for (index_type element = row_start + lane; element < row_end; element += 32)
        dot += data[element] * x[col_ids[element]];
    }

  dot = warp_reduce (dot);

  if (lane == 0 && row < n_rows)
    {
      y[row] = dot;
    }
}

template <typename data_type, typename index_type>
measurement_class gpu_csr_vector_spmv (
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

  cudaMemcpy (d_values, matrix.values.get (), matrix_size * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (d_columns, matrix.columns.get (), columns_size * sizeof (index_type), cudaMemcpyHostToDevice);
  cudaMemcpy (d_row_ptr, matrix.row_ptr.get (), row_ptr_size * sizeof (index_type), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, d_x, 1.0);
  }

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (matrix.n_rows * 32 + block_size.x - 1) / block_size.x;

    csr_spmv_vector_kernel<data_type, index_type> <<<grid_size, block_size>>> (matrix.n_rows, d_columns, d_row_ptr, d_values, d_x, d_y);
  }

  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);
  const double elapsed = milliseconds / 1000;

  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  std::unique_ptr<data_type[]> cpu_y (new data_type[y_size]);
  cudaMemcpy (cpu_y.get (), d_y, y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reference_y, cpu_y.get ());

  cudaFree (d_values);
  cudaFree (d_x);
  cudaFree (d_y);
  cudaFree (d_row_ptr);
  cudaFree (d_columns);

  return measurement_class ("GPU CSR-Vector", elapsed, 0, 0);
}

template <typename data_type, typename index_type>
__global__ void bcsr_spmv_kernel_block_per_block_row_thread_per_row_row_major_matrix (
  index_type bs,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type *y)
{
  const index_type row = threadIdx.x;
  const index_type block_row = blockIdx.x;
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];

  if (row < bs)
    {
      data_type local_out = 0.0;

      for (index_type block = first_block; block < last_block; block++)
        for (index_type col = 0; col < bs; col++)
          local_out += x[col_ids[block] * bs + col] * data[block * bs * bs + row * bs + col];

      y[block_row * bs + row] = local_out;
    }
}

template <typename data_type, typename index_type>
__global__ void bcsr_spmv_kernel_block_per_block_row_thread_per_row_column_major_matrix (
  index_type bs,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type * __restrict__ y)
{
  const index_type row = threadIdx.x;
  const index_type block_row = blockIdx.x;
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];

  if (row < bs)
    {
      data_type local_out = 0.0;

      for (index_type block = first_block; block < last_block; block++)
        for (index_type col = 0; col < bs; col++)
          local_out += x[col_ids[block] * bs + col] * data[block * bs * bs + col * bs + row];

      y[block_row * bs + row] = local_out;
    }
}

template <typename data_type, typename index_type>
__global__ void bcsr_spmv_kernel_block_per_block_row_thread_per_row_column_major_matrix_coal_x (
  index_type bs,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type *y)
{
  const index_type row = threadIdx.x;
  const index_type block_row = blockIdx.x;
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];

  data_type *cache_x = shared_memory<data_type> ();

  cache_x[threadIdx.x] = 0.0;
  data_type local_out = 0.0;

  for (index_type block = first_block; block < last_block; block++)
    {
      __syncthreads ();
      if (threadIdx.x < bs)
        cache_x[threadIdx.x] = x[col_ids[block] * bs + threadIdx.x];
      __syncthreads ();

      for (index_type col = 0; col < bs; col++)
        local_out += cache_x[col] * data[block * bs * bs + col * bs + row];
    }

  y[block_row * bs + row] = local_out;
}

void cusparse_bsrmv (
  cusparseHandle_t  &handle,
  cusparseMatDescr_t  &descr_A,
  cusparseDirection_t direction,

  int n_rows,
  int n_cols,
  int nnzb,
  int bs,

  const float *A,
  const int *row_ptr,
  const int *columns,
  const float *x,
  float *y
  )
{
  const float alpha = 1.0;
  const float beta = 0.0;

  cusparseSbsrmv (
    handle,
    direction,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    n_rows, n_cols, nnzb,
    &alpha, descr_A, A,
    row_ptr, columns, bs,
    x, &beta, y);
}

void cusparse_bsrmv (
  cusparseHandle_t  &handle,
  cusparseMatDescr_t  &descr_A,
  cusparseDirection_t direction,

  int n_rows,
  int n_cols,
  int nnzb,
  int bs,

  const double *A,
  const int *row_ptr,
  const int *columns,
  const double *x,
  double *y
)
{
  const double alpha = 1.0;
  const double beta = 0.0;

  cusparseDbsrmv (
    handle,
    direction,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    n_rows, n_cols, nnzb,
    &alpha, descr_A, A,
    row_ptr, columns, bs,
    x, &beta, y);
}

template <typename data_type, typename index_type>
std::vector<measurement_class> gpu_bcsr_spmv (
  bcsr_matrix_class<data_type, index_type> &matrix,
  const data_type *reference_y)
{
  std::vector<measurement_class> results;

  const index_type matrix_size = matrix.nnzb * matrix.bs * matrix.bs;
  const index_type columns_size = matrix.nnzb;
  const index_type row_ptr_size = matrix.n_rows + 1;
  const index_type x_size = matrix.n_cols * matrix.bs;
  const index_type y_size = matrix.n_rows * matrix.bs;

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

  cudaMemcpy (d_values, matrix.values.get (), matrix_size * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (d_columns, matrix.columns.get (), columns_size * sizeof (index_type), cudaMemcpyHostToDevice);
  cudaMemcpy (d_row_ptr, matrix.row_ptr.get (), row_ptr_size * sizeof (index_type), cudaMemcpyHostToDevice);

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (x_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (x_size, d_x, 1.0);
  }

  {
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    cudaDeviceSynchronize ();
    cudaEventRecord (start);

    {
      dim3 block_size = dim3 (matrix.bs);
      dim3 grid_size {};

      grid_size.x = (matrix.n_rows * matrix.bs + block_size.x - 1) / block_size.x;

      bcsr_spmv_kernel_block_per_block_row_thread_per_row_row_major_matrix<data_type, index_type> <<<grid_size, block_size>>> (
        matrix.bs, d_columns, d_row_ptr, d_values, d_x, d_y);
    }

    cudaEventRecord (stop);
    cudaEventSynchronize (stop);

    float milliseconds = 0;
    cudaEventElapsedTime (&milliseconds, start, stop);
    const double elapsed = milliseconds / 1000;

    cudaEventDestroy (start);
    cudaEventDestroy (stop);

    results.emplace_back ("GPU BCSR (row major, block per block row, thread per row)", elapsed, 0, 0);
  }

  std::unique_ptr<data_type[]> cpu_y (new data_type[y_size]);
  cudaMemcpy (cpu_y.get (), d_y, y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reference_y, cpu_y.get ());

  /// cuSPARSE Row major
  {
    cusparseHandle_t handle;
    cusparseCreate (&handle);

    cusparseMatDescr_t descr_A;
    cusparseCreateMatDescr (&descr_A);
    cusparseSetMatType (descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase (descr_A, CUSPARSE_INDEX_BASE_ZERO);

    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    cudaDeviceSynchronize ();
    cudaEventRecord (start);

    cusparse_bsrmv (handle, descr_A, CUSPARSE_DIRECTION_ROW, matrix.n_rows, matrix.n_cols, matrix.nnzb, matrix.bs, d_values, d_row_ptr, d_columns, d_x, d_y);

    cudaEventRecord (stop);
    cudaEventSynchronize (stop);

    float milliseconds = 0;
    cudaEventElapsedTime (&milliseconds, start, stop);
    const double elapsed = milliseconds / 1000;

    cudaEventDestroy (start);
    cudaEventDestroy (stop);

    cusparseDestroyMatDescr (descr_A);
    cusparseDestroy (handle);

    results.emplace_back ("GPU BSR (cuSPARSE, row major)", elapsed, 0, 0);

    cudaMemcpy (cpu_y.get (), d_y, y_size * sizeof (data_type), cudaMemcpyDeviceToHost);
    compare_results (y_size, reference_y, cpu_y.get ());
  }

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, d_y, 1.0);
  }

  matrix.transpose_blocks ();
  cudaMemcpy (d_values, matrix.values.get (), matrix_size * sizeof (data_type), cudaMemcpyHostToDevice);

  /// cuSPARSE Column major
  {
    cusparseHandle_t handle;
    cusparseCreate (&handle);

    cusparseMatDescr_t descr_A;
    cusparseCreateMatDescr (&descr_A);
    cusparseSetMatType (descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase (descr_A, CUSPARSE_INDEX_BASE_ZERO);

    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    cudaDeviceSynchronize ();
    cudaEventRecord (start);

    cusparse_bsrmv (handle, descr_A, CUSPARSE_DIRECTION_COLUMN, matrix.n_rows, matrix.n_cols, matrix.nnzb, matrix.bs, d_values, d_row_ptr, d_columns, d_x, d_y);

    cudaEventRecord (stop);
    cudaEventSynchronize (stop);

    float milliseconds = 0;
    cudaEventElapsedTime (&milliseconds, start, stop);
    const double elapsed = milliseconds / 1000;

    cudaEventDestroy (start);
    cudaEventDestroy (stop);

    cusparseDestroyMatDescr (descr_A);
    cusparseDestroy (handle);

    results.emplace_back ("GPU BSR (cuSPARSE, column major)", elapsed, 0, 0);

    cudaMemcpy (cpu_y.get (), d_y, y_size * sizeof (data_type), cudaMemcpyDeviceToHost);
    compare_results (y_size, reference_y, cpu_y.get ());
  }

  {
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    cudaDeviceSynchronize ();
    cudaEventRecord (start);

    {
      dim3 block_size = dim3 (matrix.bs);
      dim3 grid_size {};

      grid_size.x = (matrix.n_rows * matrix.bs + block_size.x - 1) / block_size.x;

      bcsr_spmv_kernel_block_per_block_row_thread_per_row_column_major_matrix<data_type, index_type> <<<grid_size, block_size>>> (
        matrix.bs, d_columns, d_row_ptr, d_values, d_x, d_y);
    }

    cudaEventRecord (stop);
    cudaEventSynchronize (stop);

    float milliseconds = 0;
    cudaEventElapsedTime (&milliseconds, start, stop);
    const double elapsed = milliseconds / 1000;

    cudaEventDestroy (start);
    cudaEventDestroy (stop);

    results.emplace_back ("GPU BCSR (column major, block per block row, thread per row)", elapsed, 0, 0);
  }

  std::fill_n (cpu_y.get (), y_size, 0.0);
  cudaMemcpy (cpu_y.get (), d_y, y_size * sizeof (data_type), cudaMemcpyDeviceToHost);
  compare_results (y_size, reference_y, cpu_y.get ());

  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (y_size + block_size.x - 1) / block_size.x;
    fill_vector<data_type><<<grid_size, block_size>>> (y_size, d_y, 1.0);
  }

  {
    cudaEvent_t start, stop;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    cudaDeviceSynchronize ();
    cudaEventRecord (start);

    {
      dim3 block_size = dim3 (matrix.bs);
      dim3 grid_size {};

      grid_size.x = (matrix.n_rows * matrix.bs  + block_size.x - 1) / block_size.x;

      bcsr_spmv_kernel_block_per_block_row_thread_per_row_column_major_matrix_coal_x<data_type, index_type> <<<grid_size, block_size, block_size.x * sizeof (data_type)>>> (
        matrix.bs, d_columns, d_row_ptr, d_values, d_x, d_y);
    }

    cudaEventRecord (stop);
    cudaEventSynchronize (stop);

    float milliseconds = 0;
    cudaEventElapsedTime (&milliseconds, start, stop);
    const double elapsed = milliseconds / 1000;

    cudaEventDestroy (start);
    cudaEventDestroy (stop);

    results.emplace_back ("GPU BCSR (column major, block per block row, thread per row, coal x)", elapsed, 0, 0);
  }

  std::fill_n (cpu_y.get (), y_size, 0.0);
  cudaMemcpy (cpu_y.get (), d_y, y_size * sizeof (data_type), cudaMemcpyDeviceToHost);
  compare_results (y_size, reference_y, cpu_y.get ());

  cudaFree (d_values);
  cudaFree (d_x);
  cudaFree (d_y);
  cudaFree (d_row_ptr);
  cudaFree (d_columns);

  return results;
}

#define INSTANTIATE(DTYPE,ITYPE) \
  template measurement_class gpu_csr_spmv (const csr_matrix_class<DTYPE, ITYPE> &matrix, const DTYPE *reference_y); \
  template measurement_class gpu_csr_vector_spmv (const csr_matrix_class<DTYPE, ITYPE> &matrix, const DTYPE *reference_y); \
  template std::vector<measurement_class> gpu_bcsr_spmv (bcsr_matrix_class<DTYPE, ITYPE> &matrix, const DTYPE *reference_y);

INSTANTIATE (float,int)

#undef INSTANTIATE
