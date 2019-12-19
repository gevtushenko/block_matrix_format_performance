#include "bicgstab.h"

#include <cmath>
#include <iostream>
#include <cuda_runtime.h>

#define TMP_ARRAY_SIZE 5
#define THREADS_PER_BLOCK 512
#define FULL_WARP_MASK 0xFFFFFFFF

// Common funcs
#define gpuCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr,"GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      cudaDeviceReset ();
      if (abort) exit(code);
    }
}

template <class T, class C>
gpu_bicgstab<T, C>::gpu_bicgstab (const csr_matrix_class<T, C> &A_cpu, bool use_precond)
  : n_rows (A_cpu.n_rows)
  , n_elements (A_cpu.nnz)
{
  const std::size_t offsets_vector_size = (n_rows + 1) * sizeof (C);
  const std::size_t single_vectors_size = n_rows * sizeof (T);
  const std::size_t double_vectors_size = 2 * single_vectors_size;

  cudaDeviceProp device_prop {};
  gpuCheck(cudaGetDeviceProperties (&device_prop, 0));

  const std::size_t total_memory_required = double_vectors_size * 4     /// x, r, v, p
                                          + single_vectors_size * 5     /// h, s, t, rh, rhs
                                          + n_elements * sizeof (T)     /// A;
                                          + offsets_vector_size         /// offsets_to_rows_begin
                                          + n_elements * sizeof (C);    /// column_indices

  if (total_memory_required > 0.8 * device_prop.totalGlobalMem)
    throw std::runtime_error ("Error! There is not enough memory on gpu (" + std::to_string (total_memory_required / 1024 / 1024) + "MB)!");

  if (device_prop.concurrentKernels == 0)
    throw std::runtime_error ("Error! Concurrent kernels is not supported!");

  std::cout << "Allocate " << static_cast<double> (total_memory_required) / 1024 / 1024 << "MB on GPU" << std::endl;

  gpuCheck(cudaMalloc ((void **) &x, double_vectors_size));
  gpuCheck(cudaMalloc ((void **) &r, double_vectors_size));
  gpuCheck(cudaMalloc ((void **) &v, double_vectors_size));
  gpuCheck(cudaMalloc ((void **) &p, double_vectors_size));

  gpuCheck(cudaMalloc ((void **) &h,  single_vectors_size));
  gpuCheck(cudaMalloc ((void **) &s,  single_vectors_size));
  gpuCheck(cudaMalloc ((void **) &t,  single_vectors_size));
  gpuCheck(cudaMalloc ((void **) &rh, single_vectors_size));

  gpuCheck(cudaMalloc ((void **) &tmp, TMP_ARRAY_SIZE * sizeof (T)));
  gpuCheck(cudaMalloc ((void **) &rhs, single_vectors_size));
  gpuCheck(cudaMalloc ((void **) &A, n_elements * sizeof (T)));

  gpuCheck(cudaMalloc ((void **) &column_indices, n_elements * sizeof (C)));
  gpuCheck(cudaMalloc ((void **) &offsets_to_rows_begin, offsets_vector_size));

  gpuCheck(cudaMallocHost ((void**) &h_x, double_vectors_size));

  if (use_precond)
    {
      gpuCheck(cudaMalloc ((void **) &P, n_rows * sizeof (double)));
      gpuCheck(cudaMalloc ((void **) &q, n_rows * sizeof (double)));
      gpuCheck(cudaMalloc ((void **) &z, n_rows * sizeof (double)));
    }
}

template <class T, class C>
gpu_bicgstab<T, C>::~gpu_bicgstab()
{
  gpuCheck(cudaFree (x));
  gpuCheck(cudaFree (r));
  gpuCheck(cudaFree (v));
  gpuCheck(cudaFree (p));

  gpuCheck(cudaFree (h));
  gpuCheck(cudaFree (s));
  gpuCheck(cudaFree (t));
  gpuCheck(cudaFree (rh));

  gpuCheck(cudaFree (tmp));

  gpuCheck(cudaFree (rhs));
  gpuCheck(cudaFree (A));

  gpuCheck(cudaFree (column_indices));
  gpuCheck(cudaFree (offsets_to_rows_begin));

  gpuCheck(cudaFreeHost (h_x));

  if (P)
    {
      gpuCheck (cudaFree (P));
      gpuCheck (cudaFree (q));
      gpuCheck (cudaFree (z));
    }
}

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

/**
 * @brief Reduce value within block
 * @tparam T type of data to reduce
 * @tparam op reduce operation
 * @tparam warps_count blockDim.x / warpSize
 * @param val value to reduce from each fiber
 * @return reduced value on first lane of first warp
 */
template <class T, int warps_count>
__device__ T block_reduce (T val)
{
  static __shared__ T shared[warps_count]; /// Shared memory for partial results

  const int lane = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;

  val = warp_reduce<T> (val);

  if (lane == 0) /// Main fiber stores value from it's warp
    shared[wid] = val;

  __syncthreads ();

  /// block thread id < warps count in block
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0;

  if (wid == 0) /// Reduce within first warp
    val = warp_reduce <T> (val);

  return val;
}


template <class T, int warps_count>
__device__ void dot_product (const T *a, const T *b, T *result, unsigned int n)
{
  T sum = T{};

  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  while (index < n)
    {
      sum += a[index] * b[index];
      index += stride;
    }

  sum = block_reduce <T, warps_count> (sum);

  if (threadIdx.x == 0)
    atomicAdd (result, sum);
}

template <class T, int warps_count>
__global__ void dot_product_kernel (const T *a, const T *b, T *result, unsigned int n)
{
  dot_product<T, warps_count> (a, b, result, n);
}

template <class T, class C>
__device__ void matrix_vector_multiplication (const T * __restrict__ a, const T * __restrict__ x, T * __restrict__ y, const C * __restrict__ offsets_to_rows_begin, const C * __restrict__ column_indices, C n)
{
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < n)
    {
      C row_begin = offsets_to_rows_begin[row];
      C row_end = offsets_to_rows_begin[row + 1];

      T sum = T {};

      for (C element = row_begin; element < row_end; element++)
        sum += a[element] * x[column_indices[element]];

      y[row] = sum;
    }
}

template <typename T, typename C>
__global__ void calculate_jacobi_preconditioner (
  const T * __restrict__ a,
  const C * __restrict__ offsets_to_rows_begin,
  const C * __restrict__ column_indices,
  T * __restrict__ P,
  C n)
{
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < n)
    {
      C row_begin = offsets_to_rows_begin[row];
      C row_end = offsets_to_rows_begin[row + 1];

      for (C element = row_begin; element < row_end; element++)
        {
          if (column_indices[element] == row)
            {
              P[row] = fabs (a[element]) < 1e-20 ? 1.0 : 1.0 / a[element];
              return;
            }
        }
    }
}

template <typename T, typename C>
__global__ void apply_preconditioner (
  const T * __restrict__ P,
  const T * __restrict__ v,
        T * __restrict__ q,
  C n)
{
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;

  if (row < n)
    q[row] = P[row] * v[row];
}

template <class T, class C>
__global__ void matrix_vector_multiplication_kernel (const T * __restrict__ a, const T * __restrict__ x, T * __restrict__ y, const C * __restrict__ offsets_to_rows_begin, const C * __restrict__ column_indices, C n)
{
  matrix_vector_multiplication <T> (a, x, y, offsets_to_rows_begin, column_indices, n);
}

template <class T, class C, int block_size>
void gpu_matrix_vector_multiplication (const T * __restrict__ a, const T * __restrict__ x, T * __restrict__ y, const C * __restrict__ offsets_to_rows_begin, const C * __restrict__ column_indices, C n)
{
  // int threads = THREADS_PER_BLOCK;
  // int blocks = std::min ((n + threads - 1) / threads, C (1024));
  int threads = block_size;
  int blocks = (n + threads - 1) / threads;

  matrix_vector_multiplication_kernel<T, C> <<<blocks, threads>>> (a, x, y, offsets_to_rows_begin, column_indices, n);
}

template <class T>
__global__ void bicgstab_init_kernel (
  const T *b,
  T *x,
  T *r,
  T *v,
  T *p,
  T *h,
  T *s,
  T *t,
  T *rh,
  const unsigned int n)
{
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  /// 0 - Clear arrays
  /// 1 - r0 = b - A * x0 (x0 = 0 => r0 = b)
  while (row < n)
    {
      x[row] = r[row] = v[row] = p[row] = 0.0;
      h[row] = s[row] = t[row] = 0.0;

      rh[row] = r[row] = b[row];

      row += stride;
    }
}

template <class T>
__global__ void bicgstab_23_kernel (
  T *pc,
  T *pp,
  T *rp,
  T *vp,
  T beta,
  T omegap,
  const unsigned int n)
{
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  while (row < n)
    {
      pc[row] = rp[row] + beta * (pp[row] - omegap * vp[row]);
      row += stride;
    }
}

template <class T>
__global__ void bicgstab_67_kernel (
  T *h,
  T *s,
  T *xp,
  T *rp,
  T *pc,
  T *vc,
  T alpha,
  const unsigned int n)
{
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  while (row < n)
    {
      h[row] = xp[row] + alpha * pc[row];
      s[row] = rp[row] - alpha * vc[row];

      row += stride;
    }
}

template <class T>
__global__ void bicgstab_9_kernel (
  T *t,
  T *s,
  T *tmp,
  const unsigned int n)
{
  dot_product<T, 32> (t, s, tmp + 0, n);
  dot_product<T, 32> (t, t, tmp + 1, n);
}

template <class T>
__global__ void bicgstab_1011_kernel (
  T *h,
  T *s,
  T *t,
  T *rc,
  T omegac,
  const unsigned int n)
{
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  while (row < n)
    {
      rc[row] = s[row] - (omegac) * t[row];
      row += stride;
    }
}

template <class T>
__global__ void bicgstab_1012_kernel (
  T *h,
  T *s,
  T *xc,
  T omegac,
  const unsigned int n)
{
  unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  while (row < n)
    {
      xc[row] = h[row] + (omegac) * s[row];
      row += stride;
    }
}

template <typename T>
void dbg_print (const T *data, unsigned int n, const std::string &label)
{
  std::unique_ptr<T[]> cpu_data (new T[n]);
  cudaMemcpy (cpu_data.get(), data, n * sizeof (T), cudaMemcpyDeviceToHost);

  for (unsigned int i = 0; i < n; i++)
    std::cout << label << "[" << i << "] = " << cpu_data[i] << std::endl;
}

#define matrix_block_size 18
template <class T, class C>
T *gpu_bicgstab<T, C>::solve (const csr_matrix_class<T, C> &A_cpu, const T* b, T epsilon, unsigned int max_iterations)
{
  cudaEvent_t start, stop;

  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaEventRecord (start);

  T zeros [TMP_ARRAY_SIZE] = { T {} };

  int threads = THREADS_PER_BLOCK;
  int blocks = std::min ((n_rows + threads - 1) / threads, 1024u);

  gpuCheck(cudaMemcpy (A, A_cpu.values.get (), n_elements * sizeof (T), cudaMemcpyDefault));
  gpuCheck(cudaMemcpy (column_indices, A_cpu.columns.get (), n_elements * sizeof (C), cudaMemcpyDefault));
  gpuCheck(cudaMemcpy (offsets_to_rows_begin, A_cpu.row_ptr.get (), (n_rows + 1) * sizeof (C), cudaMemcpyDefault));
  gpuCheck(cudaMemcpy (rhs, b, n_rows * sizeof (T), cudaMemcpyDefault));

  T norm_b = T {};
  cudaMemcpy (tmp, zeros, TMP_ARRAY_SIZE * sizeof (T), cudaMemcpyDefault);
  dot_product_kernel<T, 32> <<<blocks, threads>>> (rhs, rhs, tmp, n_rows);
  gpuCheck(cudaMemcpy (&norm_b, tmp, sizeof (T), cudaMemcpyDefault));
  norm_b = std::sqrt (norm_b);

  T omega[2];
  omega[0] = omega[1] = 1.0;

  T rho[2];
  rho[0] = rho[1] = 1.0;

  T alpha = 1.0;

  if (P)
    calculate_jacobi_preconditioner<T,C> <<<blocks, threads>>> (A, offsets_to_rows_begin, column_indices, P, n_rows);
  bicgstab_init_kernel<T> <<<blocks, threads>>> (rhs, x, r, v, p, h, s, t, rh, n_rows);

  for (unsigned int i = 0; i < max_iterations; )
    {
      cudaMemcpy (tmp, zeros, TMP_ARRAY_SIZE * sizeof (T), cudaMemcpyDefault);

      T &rhop = rho[i % 2];
      T &omegap = omega[i % 2];
      T *rp = r + n_rows * (i % 2);
      T *xp = x + n_rows * (i % 2);
      T *pp = p + n_rows * (i % 2);
      T *vp = v + n_rows * (i % 2);

      i++;

      T &rhoc = rho[i % 2];
      T &omegac = omega[i % 2];
      T *rc = r + n_rows * (i % 2);
      T *xc = x + n_rows * (i % 2);
      T *pc = p + n_rows * (i % 2);
      T *vc = v + n_rows * (i % 2);

      /// 1 - rho_i = (rh0, rp)
      dot_product_kernel<T, 32> <<<blocks, threads>>> (rh, rp, tmp + 2, n_rows);
      cudaMemcpy (&rhoc, tmp + 2, sizeof (T), cudaMemcpyDefault);

      /// 2 - beta = ...; 3 - pi = rp + beta (pp - omega * vp)
      T beta = rhoc / rhop * alpha / omegap;
      bicgstab_23_kernel<T> <<<blocks, threads>>> (pc, pp, rp, vp, beta, omegap, n_rows);

      if (P)
        apply_preconditioner<T, C> <<<blocks, threads>>> (P, pc, q, n_rows);
      else
        q = pc;

      /// 4 - vi = A pi
      gpu_matrix_vector_multiplication<T, C, matrix_block_size> (A, q, vc, offsets_to_rows_begin, column_indices, n_rows);

      /// 5 - alpha = rhoi / (rh0, vi)
      dot_product_kernel<T, 32> <<<blocks, threads>>> (rh, vc, tmp + 3, n_rows);
      T rh_vi_prod = T {};
      cudaMemcpy (&rh_vi_prod, tmp + 3, sizeof (T), cudaMemcpyDefault);
      alpha = rhoc / rh_vi_prod;

      /// 6 - h = xp + alpha * pi
      /// 7 - s = rp - alpha * vi
      bicgstab_67_kernel<T> <<<blocks, threads>>> (h, s, xp, rp, q, vc, alpha, n_rows);

      if (P)
        apply_preconditioner<T, C> <<<blocks, threads>>> (P, s, z, n_rows);
      else
        z = s;

      /// 8 - t = A s
      gpu_matrix_vector_multiplication<T, C, matrix_block_size> (A, z, t, offsets_to_rows_begin, column_indices, n_rows);

      cudaMemset (tmp, 0, sizeof (T) * 2);

      /// 9 - omegai = (t, s) / (t, t)
      bicgstab_9_kernel<T> <<<blocks, threads>>> (t, s, tmp, n_rows);

      T ts_and_tt_prod[2];
      cudaMemcpy (ts_and_tt_prod, tmp, 2 * sizeof (T), cudaMemcpyDefault);

      omegac = ts_and_tt_prod[0] / ts_and_tt_prod[1];

      /// 10 - ri = s - imegai * t
      bicgstab_1011_kernel<T> <<<blocks, threads>>> (h, s, t, rc, omegac, n_rows);

      /// 11 - xi = h + omegai * s
      bicgstab_1012_kernel<T> <<<blocks, threads>>> (h, z, xc, omegac, n_rows);

      /// 12 Check norms
      dot_product_kernel<T, 32> <<<blocks, threads>>> (rc, rc, tmp + 4, n_rows);

      T norm_r = T {};
      gpuCheck(cudaMemcpy (&norm_r, tmp + 4, sizeof (T), cudaMemcpyDefault));
      norm_r = std::sqrt (norm_r);

      std::cout << "i: " << i
                << "; rhs norm: " << norm_r / norm_b
                << "; rho: " << rhoc
                << "; beta: " << beta
                << "; alpha: " << alpha
                << "; omegac: " << omegac
                << "; tc: " << ts_and_tt_prod[0]
                << "; tt: " << ts_and_tt_prod[1] << "\n";

      if (norm_r / norm_b < epsilon) break;
    }

  gpuCheck(cudaMemcpy (h_x, x, n_rows * sizeof (T), cudaMemcpyDefault));

  cudaEventRecord (stop);

  float milliseconds = 0;
  cudaEventSynchronize (stop);
  cudaEventElapsedTime (&milliseconds, start, stop);

  std::cout << "\nCalculation complete in " << milliseconds/1000<< "s" << std::endl;

  return h_x;
}

template class gpu_bicgstab<float,  int>;
template class gpu_bicgstab<double, int>;
template class gpu_bicgstab<float,  unsigned int>;
template class gpu_bicgstab<double, unsigned int>;

