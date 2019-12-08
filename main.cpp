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
  return static_cast<double> (size) / 1024 / 1024/ 1024;
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
    : reference (reference_time)
    , parallel_reference (move (parallel_ref))
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

template <typename data_type, typename index_type>
void perform_measurements (
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
                << "\tCSR  => DATA: " << size_to_gb (matrix_and_vectors_data_size) << " GB; EXTRA: " << size_to_gb (csr_extra_data_size) << "\n"
                << "\tBCSR => DATA: " << size_to_gb (matrix_and_vectors_data_size) << " GB; EXTRA: " << size_to_gb (bcsr_extra_data_size) << std::endl;
    }

  fmt::print (fmt::fg (fmt::color::tomato), "\nBS: {}\n", bs);

  auto block_matrix = gen_n_diag_bcsr<data_type, index_type> (n_rows, blocks_per_row, bs);
  auto matrix = std::make_unique<csr_matrix_class<data_type, index_type>> (*block_matrix);

  auto measure_multiple_times = [&] (const std::function<measurement_class(bool)> &action)
  {
    measurement_class result;
    const unsigned int measurements_count = 20;
    for (unsigned int measurement_id = 0; measurement_id < measurements_count; measurement_id++)
      result += action (measurement_id == 0);
    result.finalize ();
    return result;
  };

  std::unique_ptr<data_type> reference_answer (new data_type[n_rows * bs]);
  std::unique_ptr<data_type> x (new data_type[n_rows * bs]);
  auto cpu_naive = measure_multiple_times ([&] (bool) {
    return cpu_csr_spmv_single_thread_naive (*matrix, x.get (), reference_answer.get ());
  });

  time_printer single_core_timer (cpu_naive.get_elapsed ());
  single_core_timer.print_time (cpu_naive);

  auto gpu_elapsed_csr = gpu_csr_spmv<data_type, index_type> (*matrix, reference_answer.get ());
  single_core_timer.print_time (gpu_elapsed_csr);

  auto gpu_elapsed_csr_vector = gpu_csr_vector_spmv<data_type, index_type> (*matrix, reference_answer.get ());
  single_core_timer.print_time (gpu_elapsed_csr_vector);

  auto bcsr_elapsed = gpu_bcsr_spmv<data_type, index_type> (*block_matrix, reference_answer.get ());

  for (auto &elapsed: bcsr_elapsed)
    single_core_timer.print_time (elapsed);
}

int main ()
{
  if (0)
    {
      cudaSetDevice (1);

      for (auto &bs: {2, 4, 8, 16, 32})
        perform_measurements<float, int> (bs, 70'000, 6);
    }

  auto load = [] (double x) -> std::pair<double, double> {
    if (x > 345 && x < 345 + 1280)
      return {0, -0.1};
    return {0, 0};
  };

  golden_gate_bridge_2d<double, int> bridge_2d (load, 17.62);
  bridge_2d.write_vtk ("output_1.vtk");

  auto matrix = std::make_unique<csr_matrix_class<double , int>> (*bridge_2d.matrix);

  gpu_bicgstab<double, int> solver (*matrix);
  auto solution = solver.solve (*matrix, bridge_2d.forces_rhs.get (), 1.e-5, 10000);
  bridge_2d.write_vtk ("output_2.vtk", solution);

  return 0;
}