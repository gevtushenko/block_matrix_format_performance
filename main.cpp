#include "measurement_class.h"
#include "matrix_converters.h"

#include "gpu_matrix_multiplier.h"
#include "bicgstab.h"

#include "fem_2d/golden_gate_bridge.h"

#include <cuda_runtime.h>

#include <functional>
#include <iostream>
#include <optional>
#include <chrono>
#include <memory>

#include "fmt/format.h"
#include "fmt/color.h"
#include "fmt/core.h"

template<typename data_type, typename index_type>
measurement_class cpu_csr_spmv_single_thread_naive (
  const csr_matrix_class<data_type, index_type> &matrix,
  data_type *x,
  data_type *y)
{
  std::fill_n (x, matrix.n_cols, 1.0);
  std::fill_n (y, matrix.n_rows, 0.0);

  const auto row_ptr = matrix.row_ptr.get ();
  const auto col_ids = matrix.columns.get ();
  const auto data = matrix.values.get ();

  auto begin = std::chrono::system_clock::now ();

  for (index_type row = 0; row < matrix.n_rows; row++)
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

  const size_t data_bytes = matrix.nnz * sizeof (data_type);
  const size_t x_bytes = matrix.nnz * sizeof (data_type);
  const size_t col_ids_bytes = matrix.nnz * sizeof (index_type);
  const size_t row_ids_bytes = 2 * matrix.n_rows * sizeof (index_type);
  const size_t y_bytes = matrix.n_rows * sizeof (data_type);

  const size_t operations_count = matrix.nnz * 2;

  return measurement_class (
    "CPU CSR",
    elapsed,
    data_bytes + x_bytes + col_ids_bytes + row_ids_bytes + y_bytes,
    operations_count);
}

double size_to_gb (size_t size)
{
  return static_cast<double> (size) / 1024 / 1024 / 1024;
}

class time_printer
{
  double reference {};
  std::optional<double> parallel_reference;

  /// Settings
  const unsigned int time_width = 20;
  const unsigned int time_precision = 6;

public:
  explicit time_printer (
    double reference_time,
    std::optional<double> parallel_ref = std::nullopt)
    : reference (reference_time), parallel_reference (move (parallel_ref))
  {
  }

  void add_time (double time, fmt::color color) const
  {
    fmt::print (fmt::fg (color), "{2:<{0}.{1}g}   ", time_width, time_precision, time);
  }

  void print_time (const measurement_class &measurement) const
  {
    const double time = measurement.get_elapsed ();
    fmt::print (fmt::fg (fmt::color::yellow), "\t{0:<80}", measurement.get_format ());
    fmt::print (":  ");
    add_time (time, fmt::color::white);
    add_time (speedup (time), fmt::color::green);
    if (parallel_reference)
      add_time (parallel_speedup (time), fmt::color::green_yellow);
    fmt::print ("\n");
  }

  double speedup (double time) const
  {
    return reference / time;
  }

  double parallel_speedup (double time) const
  {
    return *parallel_reference / time;
  }
};

#include "cuda_jit.h"

template <typename index_type>
index_type round_up_to_power_of_two (index_type v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}

template<typename data_type, typename index_type>
std::unordered_map<std::string, double> perform_measurements (
  csr_matrix_class<data_type, index_type> &matrix,
  bcsr_matrix_class<data_type, index_type> &block_matrix
)
{
  std::unordered_map<std::string, double> results;
  const unsigned int measurements_count = 10;
  auto measure_multiple_times = [&] (const std::function<measurement_class (bool)> &action)
  {
    measurement_class result;
    for (unsigned int measurement_id = 0; measurement_id < measurements_count; measurement_id++)
      result += action (measurement_id == 0);
    result.finalize ();

    results[result.get_format ()] = result.get_elapsed ();

    return result;
  };

  const index_type n_rows = block_matrix.n_rows;
  const index_type bs = block_matrix.bs;
  std::unique_ptr<data_type> reference_answer (new data_type[n_rows * bs]);
  std::unique_ptr<data_type> x (new data_type[n_rows * bs]);
  auto cpu_naive = measure_multiple_times ([&] (bool)
                                           {
                                             return cpu_csr_spmv_single_thread_naive (matrix, x.get (), reference_answer.get ());
                                           });

  time_printer single_core_timer (cpu_naive.get_elapsed ());
  single_core_timer.print_time (cpu_naive);

  auto gpu_elapsed_csr = measure_multiple_times ([&] (bool) { return gpu_csr_spmv<data_type, index_type> (matrix, reference_answer.get ()); });
  single_core_timer.print_time (gpu_elapsed_csr);

  auto gpu_elapsed_csr_vector = measure_multiple_times ([&] (bool) { return gpu_csr_vector_spmv<data_type, index_type> (matrix, reference_answer.get ()); });
  single_core_timer.print_time (gpu_elapsed_csr_vector);


  std::unique_ptr<data_type[]> transposed_matrix_data (new data_type[block_matrix.size ()]);
  block_matrix.transpose_blocks (transposed_matrix_data.get ());

  dim3 block_size = 32;
  dim3 grid_size {};

  grid_size.x = (block_matrix.n_rows * 32 + block_size.x - 1) / block_size.x;

  jit(bcsr_jit,
  {
    const int bs = {{ bs }};

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = idx % 32;
    const int block_row = idx / 32; ///< Warp per block row
    const int first_block = row_ptr[block_row];
    const int last_block = row_ptr[block_row + 1];

    int col = first_block * bs + lane / bs;
    int r = lane % bs;

    __shared__ float partial_sums[{{ shared_size }}]; // = shared_memory<float> (); ///< Size is equal to blockDim.x * sizeof(float)

    float local_out = 0.0;

    for (; col < last_block * bs; col += 32 / bs)
      {
        const int block = col / bs;
        const int c = col % bs;

        const float value = data[block * bs * bs + c * bs + r];
        const float x_value = x[col_ids[block] * bs + c];
        local_out += x_value * value;
      }

    partial_sums[threadIdx.x] = local_out;

    for (int stride = {{ stride_begin }} ; stride > 0; stride /= 2)
      {
        __syncthreads ();
        if ((lane < stride * bs) && ((threadIdx.x + stride * bs) < 32))
          {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride * bs];
          }
      }

    if (lane < bs)
      {
        y[block_row * bs + lane] = partial_sums[threadIdx.x];
      }
  },
    (const int *, col_ids),
    (const int *, row_ptr),
    (const float *, data),
    (const float *, x),
    (float*, y));
  nlohmann::json json;
  json["bs"] = bs;
  json["stride_begin"] = round_up_to_power_of_two((32 / bs) / 2);
  json["shared_size"] = block_size.x;
  auto bcsr_kernel = bcsr_jit.compile (json);

  const index_type matrix_size = block_matrix.nnzb * block_matrix.bs * block_matrix.bs;
  const index_type columns_size = block_matrix.nnzb;
  const index_type row_ptr_size = block_matrix.n_rows + 1;
  const index_type x_size = block_matrix.n_cols * block_matrix.bs;
  const index_type y_size = block_matrix.n_rows * block_matrix.bs;

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

  cudaMemcpy (d_values, transposed_matrix_data.get (), matrix_size * sizeof (data_type), cudaMemcpyHostToDevice);
  cudaMemcpy (d_columns, block_matrix.columns.get (), columns_size * sizeof (index_type), cudaMemcpyHostToDevice);
  cudaMemcpy (d_row_ptr, block_matrix.row_ptr.get (), row_ptr_size * sizeof (index_type), cudaMemcpyHostToDevice);

  std::unique_ptr<float[]> h_x (new float[x_size]);
  std::fill_n (h_x.get (), x_size, 1.0);

  cudaMemcpy (d_x, h_x.get (), x_size * sizeof (float), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);

  bcsr_kernel.launch (grid_size, block_size, d_columns, d_row_ptr, d_values, d_x, d_y);

  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  std::unique_ptr<data_type[]> cpu_y (new data_type[y_size]);
  cudaMemcpy (cpu_y.get (), d_y, y_size * sizeof (data_type), cudaMemcpyDeviceToHost);

  compare_results (y_size, reference_answer.get (), cpu_y.get ());

  cudaFree (d_values);
  cudaFree (d_x);
  cudaFree (d_y);
  cudaFree (d_row_ptr);
  cudaFree (d_columns);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);
  const double elapsed = milliseconds / 1000;

  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  results["jit"] = elapsed;

  measurement_class jit_measure ("jit", elapsed, 0.0, 0.0);
  single_core_timer.print_time (jit_measure);

  std::vector<measurement_class> multiple_measurements = gpu_bcsr_spmv<data_type, index_type> (block_matrix, transposed_matrix_data.get (), reference_answer.get ());

  for (unsigned int measurement_i = 0; measurement_i < measurements_count; measurement_i++)
    {
      auto new_result = gpu_bcsr_spmv<data_type, index_type> (block_matrix, transposed_matrix_data.get (), reference_answer.get ());

      for (unsigned int i = 0; i < multiple_measurements.size(); i++)
        multiple_measurements[i] += new_result[i];
    }

  for (auto &measure: multiple_measurements)
    measure.finalize ();

  for (auto &elapsed: multiple_measurements)
    {
      results[elapsed.get_format ()] = elapsed.get_elapsed ();
      single_core_timer.print_time (elapsed);
    }

  return results;
}

template<typename data_type, typename index_type>
auto measure_diag_matrices (
  index_type bs,
  index_type n_rows,
  index_type blocks_per_row,
  bool debug_info = false
)
{
  const size_t nnz = n_rows * blocks_per_row * bs * bs;

  const double matrix_and_vectors_data_size = static_cast<double> (nnz + 2 * n_rows * bs) * sizeof (data_type);
  const size_t csr_extra_data_size = (nnz + n_rows * bs + bs) * sizeof (index_type);
  const size_t bcsr_extra_data_size = (n_rows * blocks_per_row + n_rows + 1) * sizeof (index_type);

  if (debug_info)
    {
      std::cout << "Required memory: \n"
                << "\tCSR  => DATA: " << size_to_gb (matrix_and_vectors_data_size) << " GB; EXTRA: "
                << size_to_gb (csr_extra_data_size) << "\n"
                << "\tBCSR => DATA: " << size_to_gb (matrix_and_vectors_data_size) << " GB; EXTRA: "
                << size_to_gb (bcsr_extra_data_size) << std::endl;
    }

  fmt::print (fmt::fg (fmt::color::tomato), "\nBS: {}\n", bs);

  auto block_matrix = gen_n_diag_bcsr<data_type, index_type> (n_rows, blocks_per_row, bs);
  auto matrix = std::make_unique<csr_matrix_class<data_type, index_type>> (*block_matrix);

  return perform_measurements (*matrix, *block_matrix);
}

template<typename data_type, typename index_type>
void measure_golden_bridge (
  bool solve = false
)
{
  const data_type side_length = 345.0; ///< Size from bridge tower to bank in meters
  const data_type main_part_length = 100 * 1280.0; ///< Size from tower to tower in meters

  auto load = [=] (data_type x) -> std::pair<data_type, data_type>
  {
    const data_type mid_poindex_type = (main_part_length + side_length * 2) / 2;
    const data_type window = 450;
    if (x > mid_poindex_type - window && x < mid_poindex_type + window)
      return {0, -2000000.0};
    return {0, 0};
  };

  golden_gate_bridge_2d<data_type, index_type, false> bridge_2d (load, main_part_length, side_length, 260, 7.62);
  auto matrix = std::make_unique<csr_matrix_class<data_type, index_type>> (*bridge_2d.matrix);

  if (solve)
    {
      matrix->write_mm ("matrix.mtx");
      bridge_2d.write_vtk ("output_1.vtk");
      gpu_bicgstab<data_type, index_type> solver (*matrix, true);
      auto solution = solver.solve (*matrix, bridge_2d.forces_rhs.get (), 0.8, 1000);
      bridge_2d.write_vtk ("output_2.vtk", solution);
    }
  else
    {
      perform_measurements (*matrix, *bridge_2d.matrix);
    }
}

#include "json.hpp"
#include <fstream>

int main ()
{
  cudaSetDevice (1);

  nlohmann::json json;
  for (auto bs: {2, 4, 8, 16, 32})
    {
      auto result = measure_diag_matrices<float, int> (bs, 50'000, 6);
      json[std::to_string(bs)] = result;
    }

  std::ofstream os ("result.json");
  os << json.dump (2) << std::endl;

  #if 0
    measure_golden_bridge<double, int> (false);
  #endif

  return 0;
}