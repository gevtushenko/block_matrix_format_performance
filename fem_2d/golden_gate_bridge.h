//
// Created by egi on 12/1/19.
//

#ifndef BLOCK_MATRIX_FORMAT_PERFORMANCE_GOLDEN_GATE_BRIDGE_H
#define BLOCK_MATRIX_FORMAT_PERFORMANCE_GOLDEN_GATE_BRIDGE_H

#include "matrix_converters.h"

#include <fstream>
#include <memory>
#include <cmath>
#include <array>

/// C = A * B
template <class data_type, class index_type>
void matrix_mult_matrix (const data_type *a, const data_type *b, data_type *c, const index_type n)
{
  for (index_type i = 0; i < n; i++)
    {
      for (index_type j = 0; j < n; j++)
        {
          data_type sum = 0.0;
          for (index_type k = 0; k < n; k++)
            sum += a[i * n + k] * b[k * n + j];
          c[i * n + j] = sum;
        }
    }
}

/// C = A^T * B
template <class data_type, class index_type>
void matrix_transponse_and_mult (const data_type *a, const data_type *b, data_type *c, const index_type n)
{
  for (index_type i = 0; i < n; i++)
    {
      for (index_type j = 0; j < n; j++)
        {
          data_type sum = 0.0;
          for (index_type k = 0; k < n; k++)
            sum += a[k * n + i] * b[k * n + j];
          c[i * n + j] = sum;
        }
    }
}

template <typename data_type, typename index_type>
class golden_gate_bridge_2d
{
  /**
   * Segment:
   *
   *       0    1    2
   *       +----+----+
   *       |   /|\   |
   *       |  / | \  |
   *       | /  |  \ |
   *       |/   |   \|
   *       +----+----+
   *       3    4    5
   */

  constexpr static const index_type block_size = 4; ///< Local stiffness matrix size
  const data_type segment_length = 10; ///< Size of segment in meters
  const data_type side_length = 345.0; ///< Size from bridge tower to bank in meters
  const data_type main_part_length = 1280.0; ///< Size from tower to tower in meters
  const data_type tower_height = 230.0; ///< Height of tower in meters (from water level)
  const data_type bridge_height = 78.0;
  const data_type section_height = 7.62; ///< In meters
  const index_type segments_count {};

  index_type elements_count {};

  index_type first_available_node_id {};
  index_type first_available_element_id {};

  index_type left_tower_bottom {};
  index_type left_tower_top {};

  index_type right_tower_bottom {};
  index_type right_tower_top {};

  const data_type rope_e = 190; // steel?
  const data_type tower_e = 190; // steel?
  const data_type segment_e = 190; // steel?
  const data_type spin_e = 190; // steel?

  const data_type spin_a = 0.924; /// meters
  const data_type rope_a = 0.3; /// meters
  const data_type tower_a = 5.0; /// meters
  const data_type segment_a = 0.3; /// meters

public:
  explicit golden_gate_bridge_2d (data_type segment_length_arg = 7.62)
    : segment_length (segment_length_arg)
    , segments_count (calculate_segments_count())
    , elements_count (calculate_elements_count())
    , nodes_count (calculate_nodes_count())
    , nodes_xs (new data_type[nodes_count])
    , nodes_ys (new data_type[nodes_count])
    , elements_areas (new data_type[elements_count])
    , elements_e (new data_type[elements_count])
    , dxs (new data_type[elements_count])
    , dys (new data_type[elements_count])
    , lens (new data_type[elements_count])
    , elements_starts (new index_type[elements_count])
    , elements_ends (new index_type[elements_count])
  {
    std::fill_n (elements_starts.get (), elements_count, 0);
    std::fill_n (elements_ends.get (), elements_count, 0);
    std::fill_n (nodes_xs.get (), nodes_count, 0.0);
    std::fill_n (nodes_ys.get (), nodes_count, 0.0);

    fill_road_part ();
    fill_tower_part ();
    fill_side_spin_and_ropes ();
    fill_main_spin_and_ropes ();

    elements_count = first_available_element_id;
    nodes_count = first_available_node_id;

    finalize_elements ();
    calculate_local_stiffness_matrices ();

    matrix = std::make_unique<bcsr_matrix_class<data_type, index_type>> (nodes_count, nodes_count, block_size, elements_count);
  }

  void write_vtk (const std::string &filename)
  {
    std::ofstream vtk (filename);

    vtk << "# vtk DataFile Version 3.0\n";
    vtk << "vtk output\n";
    vtk << "ASCII\n";
    vtk << "DATASET UNSTRUCTURED_GRID\n";

    vtk << "POINTS " << nodes_count << " double\n";

    for (index_type node_id = 0; node_id < nodes_count; node_id++)
      vtk << nodes_xs[node_id] << " "
          << nodes_ys[node_id] << " 0\n";

    vtk << "CELLS " << elements_count << " " << 3 * elements_count << "\n";

    for (unsigned int element_id = 0; element_id < elements_count; element_id++)
      vtk << "2 " << elements_starts[element_id] << " " << elements_ends[element_id] << "\n";

    vtk << "CELL_TYPES " << elements_count << "\n";

    for (unsigned int element_id = 0; element_id < elements_count; element_id++)
      vtk << "3\n"; ///< VTK_LINE
  }

private:
  void set_element (index_type id, index_type begin, index_type end, data_type area, data_type e /* young's modulus */)
  {
    elements_starts[id] = begin;
    elements_ends[id] = end;
    elements_areas[id] = area;
    elements_e[id] = e;
  }

  index_type calculate_segments_count ()
  {
    const data_type total_length = main_part_length + 2 * side_length;
    return total_length / segment_length;
  }

  index_type calculate_elements_count ()
  {
    const index_type elements_count_in_road_part = segments_count * 8 + 4;
    const index_type towers_part = 2;
    const index_type main_spin_elements_count = segments_count;
    const index_type ropes_elements_count = segments_count;

    return elements_count_in_road_part + towers_part + ropes_elements_count + main_spin_elements_count;
  }

  index_type calculate_nodes_count ()
  {
    const index_type nodes_count_in_road_part = segments_count * 4 + 2;
    const index_type ropes_top_nodes_count = segments_count;
    const index_type main_spin_elements_count = segments_count;
    const index_type towers_part = 4;

    return nodes_count_in_road_part + towers_part + ropes_top_nodes_count + main_spin_elements_count;
  }

  void fill_road_part ()
  {
    const data_type dx = segment_length / 2;
    for (index_type segment_id = 0; segment_id < segments_count; segment_id++)
      {
        /// n_1 < n_2 < n_3 < n_4
        const index_type n_1 = segment_id * 4 + 0;
        const index_type n_2 = segment_id * 4 + 1;
        const index_type n_3 = segment_id * 4 + 2;
        const index_type n_4 = segment_id * 4 + 3;

        const index_type n_1_n = (segment_id + 1) * 4 + 0;
        const index_type n_3_n = segment_id == segments_count - 1 ? n_1_n + 1 : (segment_id + 1) * 4 + 2;

        /// Points 1 - 2
        nodes_xs[n_1] = segment_length * segment_id;
        nodes_xs[n_2] = segment_length * segment_id + dx;

        nodes_ys[n_1] = bridge_height;
        nodes_ys[n_2] = bridge_height;

        /// Points 4 - 5
        nodes_xs[n_3] = segment_length * segment_id;
        nodes_xs[n_4] = segment_length * segment_id + dx;

        nodes_ys[n_3] = bridge_height - section_height;
        nodes_ys[n_4] = bridge_height - section_height;

        const index_type e_1 = segment_id * 8 + 0;
        const index_type e_2 = segment_id * 8 + 1;
        const index_type e_3 = segment_id * 8 + 2;
        const index_type e_4 = segment_id * 8 + 3;
        const index_type e_5 = segment_id * 8 + 4;
        const index_type e_6 = segment_id * 8 + 5;
        const index_type e_7 = segment_id * 8 + 6;
        const index_type e_8 = segment_id * 8 + 7;

        set_element (e_1, n_1, n_3,   segment_a, segment_e);
        set_element (e_2, n_2, n_4,   segment_a, segment_e);
        set_element (e_3, n_1, n_2,   segment_a, segment_e);
        set_element (e_4, n_3, n_4,   segment_a, segment_e);
        set_element (e_5, n_3, n_2,   segment_a, segment_e);
        set_element (e_6, n_2, n_3_n, segment_a, segment_e);
        set_element (e_7, n_2, n_1_n, segment_a, segment_e);
        set_element (e_8, n_4, n_3_n, segment_a, segment_e);
      }

    const index_type n_1 = segments_count * 4 + 0;
    const index_type n_3 = segments_count * 4 + 1;

    nodes_xs[n_1] = segment_length * segments_count;
    nodes_xs[n_3] = segment_length * segments_count;

    nodes_ys[n_1] = bridge_height;
    nodes_ys[n_3] = bridge_height - section_height;

    const index_type e = segments_count * 8 + 0;
    set_element (e, n_1, n_3, segment_a, segment_e);

    first_available_node_id = n_3 + 1;
    first_available_element_id = e + 1;
  }

  void fill_tower_part ()
  {
    left_tower_bottom = first_available_node_id++;
    left_tower_top = first_available_node_id++;

    right_tower_bottom = first_available_node_id++;
    right_tower_top = first_available_node_id++;

    const index_type left_tower = first_available_element_id++;
    const index_type right_tower = first_available_element_id++;

    nodes_xs[left_tower_bottom]  = nodes_xs[left_tower_top]  = side_length;
    nodes_xs[right_tower_bottom] = nodes_xs[right_tower_top] = side_length + main_part_length;

    nodes_ys[right_tower_bottom] = nodes_ys[left_tower_bottom] = 0.0;
    nodes_ys[right_tower_top] = nodes_ys[left_tower_top] = tower_height;

    set_element (left_tower, left_tower_bottom, left_tower_top, tower_a, tower_e);
    set_element (right_tower, right_tower_bottom, right_tower_top, tower_a, tower_e);
  }

  void fill_side_spin_and_ropes ()
  {
    auto get_line_eq = [&] (const data_type y_1, const data_type y_2, const data_type x_1, const data_type x_2)
    {
      const data_type a = (y_1 - y_2);
      const data_type b = (x_2 - x_1);
      const data_type c = y_1 * x_2 - x_1 * y_2;

      return [=] (data_type x)
      {
        // Ax + By = C
        return (c - a * x) / b;
      };
    };

    /// Left
    auto get_y_left = get_line_eq (bridge_height, tower_height, 0, side_length);
    const index_type first_left_side_spin_segment = first_available_element_id++;

    set_element (first_left_side_spin_segment, 0, first_available_node_id, rope_a, rope_e);

    for (index_type segment_id = 0; segment_id < (side_length - segment_length) / segment_length; segment_id++)
      {
        const index_type rope_bottom = segment_id * 4 + 1;
        const index_type rope_top = first_available_node_id++;
        const index_type rope = first_available_element_id++;

        nodes_xs[rope_top] = segment_length * segment_id + segment_length / 2;
        nodes_ys[rope_top] = get_y_left (nodes_xs[rope_top]);

        set_element (rope, rope_bottom, rope_top, rope_a, rope_e);

        if (segment_id > 0)
          {
            const index_type spin = first_available_element_id++;
            set_element (spin, rope_top - 1, rope_top, spin_a, spin_e);
          }
      }

    const index_type last_left_side_spin_segment = first_available_element_id++;
    set_element (last_left_side_spin_segment, first_available_node_id - 1, left_tower_top, spin_a, spin_e);

    /// Left
    auto get_y_right = get_line_eq (tower_height, bridge_height, side_length + main_part_length, main_part_length + 2 * side_length);

    const index_type first_right_side_spin_segment = first_available_element_id++;
    set_element (first_right_side_spin_segment, right_tower_top, first_available_node_id, spin_a, spin_e);

    bool first_right_spine_segment = true;
    for (index_type segment_id = (side_length + main_part_length + segment_length) / segment_length;
         segment_id < (main_part_length + 2 * side_length - segment_length) / segment_length;
         segment_id++)
      {
        const index_type rope_bottom = segment_id * 4 + 1;
        const index_type rope_top = first_available_node_id++;
        const index_type rope = first_available_element_id++;

        nodes_xs[rope_top] = segment_length * segment_id + segment_length / 2;
        nodes_ys[rope_top] = get_y_right (nodes_xs[rope_top]);

        set_element (rope, rope_bottom, rope_top, rope_a, rope_e);

        if (first_right_spine_segment)
          {
            first_right_spine_segment = false;
          }
        else
          {
            const index_type spin = first_available_element_id++;
            set_element (spin, rope_top - 1, rope_top, spin_a, spin_e);
          }
      }

    const index_type last_right_side_spin_segment = first_available_element_id++;
    set_element (last_right_side_spin_segment, first_available_node_id - 1, segments_count * 4 + 0, spin_a, spin_e);
  }

  void fill_main_spin_and_ropes ()
  {
    auto get_y = [this] (data_type x_arg)
    {
      /// y = A*x^2 + B*x + C
      const data_type x = x_arg - side_length - main_part_length / 2;
      const data_type a = 0.0003662109375;
      const data_type c = bridge_height + 4;

      return a * x * x + c;
    };

    const index_type first_spin_segment = first_available_element_id++;
    set_element (first_spin_segment, left_tower_top, first_available_node_id, spin_a, spin_e);

    bool first_right_spine_segment = true;
    for (index_type segment_id = (side_length + segment_length) / segment_length;
         segment_id < (side_length + main_part_length - segment_length) / segment_length;
         segment_id++)
      {
        const index_type rope_bottom = segment_id * 4 + 1;
        const index_type rope_top = first_available_node_id++;
        const index_type rope = first_available_element_id++;

        nodes_xs[rope_top] = segment_length * segment_id + segment_length / 2;
        nodes_ys[rope_top] = get_y (nodes_xs[rope_top]);

        set_element (rope, rope_bottom, rope_top, rope_a, rope_e);

        if (first_right_spine_segment)
          {
            first_right_spine_segment = false;
          }
        else
          {
            const index_type spin = first_available_element_id++;
            set_element (spin, rope_top - 1, rope_top, spin_a, spin_e);
          }
      }

    const index_type last_spin_segment = first_available_element_id++;
    set_element (last_spin_segment, right_tower_top, first_available_node_id  - 1, spin_a, spin_e);
  }

  void finalize_elements ()
  {
    for (index_type element = 0; element < elements_count; element++)
      {
        dxs[element] = nodes_xs[elements_ends[element]] - nodes_xs[elements_starts[element]];
        dys[element] = nodes_ys[elements_ends[element]] - nodes_ys[elements_starts[element]];
        lens[element] = std::sqrt (dxs[element] * dxs[element] + dys[element] * dys[element]);
      }
  }

  void calculate_local_stiffness_matrices ()
  {
    std::array<data_type, block_size * block_size> k_prime;
    std::array<data_type, block_size * block_size> beta_prime;
    std::array<data_type, block_size * block_size> tmp_block;
    std::array<data_type, block_size * block_size> stiffness_matrix;

    for (index_type element = 0; element < elements_count; element++)
      {
        k_prime.fill (0.0);
        beta_prime.fill (0.0);

        const data_type cos_theta = dxs[element] / lens[element];
        const data_type sin_theta = dys[element] / lens[element];
        const data_type ae_over_l = elements_areas[element] * elements_e[element] / lens[element];

        /// 0) We need to fill follow elements of local stiffness matrix
        ///
        ///   +-----------------------------+
        ///   | i\j |    0 |  1 |    2 |  3 |
        ///   +-----+---- -+----+------+----+
        ///   | 0   | AE/L |  0 |-AE/L |  0 |
        ///   +-----+------+----+------+----+
        ///   | 1   |    0 |  0 |    0 |  0 |
        ///   +-----+------+----+------+----+
        ///   | 2   |-AE/L |  0 | AE/L |  0 |
        ///   +-----+------+----+------+----+
        ///   | 3   |    0 |  0 |    0 |  0 |
        ///   +-----------------------------+

        k_prime[0 * block_size + 0] =  ae_over_l;
        k_prime[0 * block_size + 2] = -ae_over_l;
        k_prime[2 * block_size + 0] = -ae_over_l;
        k_prime[2 * block_size + 2] =  ae_over_l;

        /// 1) We need to fill elements of rotation matrix beta prime
        ///
        ///   +-----------------------------+
        ///   | i\j |   0 |  1  |   2 |   3 |
        ///   +-----+-----+-----+-----+-----+
        ///   | 0   | cos | sin |   0 |   0 |
        ///   +-----+-----+-----+-----+-----+
        ///   | 1   |-sin | cos |   0 |   0 |
        ///   +-----+-----+-----+-----+-----+
        ///   | 2   |   0 |   0 | cos | sin |
        ///   +-----+-----+-----+-----+-----+
        ///   | 3   |   0 |   0 |-sin | cos |
        ///   +-----------------------------+

        beta_prime[0 * block_size + 0] =  cos_theta;
        beta_prime[0 * block_size + 1] =  sin_theta;
        beta_prime[1 * block_size + 0] = -sin_theta;
        beta_prime[1 * block_size + 1] =  cos_theta;

        beta_prime[2 * block_size + 2] =  cos_theta;
        beta_prime[2 * block_size + 3] =  sin_theta;
        beta_prime[3 * block_size + 2] = -sin_theta;
        beta_prime[3 * block_size + 3] =  cos_theta;

        /// 2) We need to calculate global stiffness matrix for this element: [k_e] = [Beta_e]^T [k^{'}_{e}] [Beta_e]
        matrix_mult_matrix (k_prime.data(), beta_prime.data(), tmp_block.data(), block_size);
        matrix_transponse_and_mult (beta_prime.data(), tmp_block.data(), stiffness_matrix.data(), block_size);
      }
  }

private:
  index_type nodes_count {};
  std::unique_ptr<data_type[]> nodes_xs;
  std::unique_ptr<data_type[]> nodes_ys;

  std::unique_ptr<data_type[]> elements_areas;
  std::unique_ptr<data_type[]> elements_e;

  std::unique_ptr<data_type[]> dxs;  ///< Higher node id minus lower node id!
  std::unique_ptr<data_type[]> dys;  ///< Higher node id minus lower node id!
  std::unique_ptr<data_type[]> lens; ///< Higher node id minus lower node id!

  std::unique_ptr<index_type[]> elements_starts;
  std::unique_ptr<index_type[]> elements_ends;

public:
  std::unique_ptr<bcsr_matrix_class<data_type, index_type>> matrix;
};

#endif //BLOCK_MATRIX_FORMAT_PERFORMANCE_GOLDEN_GATE_BRIDGE_H
