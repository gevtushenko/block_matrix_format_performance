//
// Created by egi on 12/1/19.
//

#ifndef BLOCK_MATRIX_FORMAT_PERFORMANCE_GOLDEN_GATE_BRIDGE_H
#define BLOCK_MATRIX_FORMAT_PERFORMANCE_GOLDEN_GATE_BRIDGE_H

#include "matrix_converters.h"

#include <fstream>
#include <memory>

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

  const index_type block_size = 4; ///< Local stiffness matrix size
  const data_type segment_length = 10; ///< Size of segment in meters
  const data_type side_length = 345.0; ///< Size from bridge tower to bank in meters
  const data_type main_part_length = 1280.0; ///< Size from tower to tower in meters
  const data_type tower_height = 230.0; ///< Height of tower in meters (from water level)
  const data_type bridge_height = 78.0;
  const data_type section_height = 7.62; ///< In meters
  const index_type segments_count {};

  const index_type elements_count {};

public:
  explicit golden_gate_bridge_2d (data_type segment_length_arg = 7.64)
    : segment_length (segment_length_arg)
    , segments_count (calculate_segments_count())
    , elements_count (calculate_elements_count())
    , nodes_count (calculate_nodes_count())
    , nodes_xs (new data_type[nodes_count])
    , nodes_ys (new data_type[nodes_count])
    , elements_starts (new index_type[elements_count])
    , elements_ends (new index_type[elements_count])
  {

    std::fill_n (elements_starts.get (), elements_count, 0);
    std::fill_n (elements_ends.get (), elements_count, 0);
    std::fill_n (nodes_xs.get (), nodes_count, 0.0);
    std::fill_n (nodes_ys.get (), nodes_count, 0.0);

    // matrix (new bcsr_matrix_class<data_type, index_type> (elements_count, elements_count, block_size, ))

    fill_road_part ();
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
  index_type calculate_segments_count ()
  {
    const data_type total_length = main_part_length + 2 * side_length;
    return total_length / segment_length;
  }

  index_type calculate_elements_count ()
  {
    const index_type elements_count_in_road_part = segments_count * 8 + 4;
    const index_type main_spin_elements_count = segments_count;
    const index_type ropes_elements_count = segments_count;

    return elements_count_in_road_part; // + main_spin_elements_count + ropes_elements_count;
  }

  index_type calculate_nodes_count ()
  {
    const index_type nodes_count_in_road_part = segments_count * 4 + 2;

    return nodes_count_in_road_part; // + main_spin_elements_count + ropes_elements_count;
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

        elements_starts[e_1] = n_1; elements_ends[e_1] = n_3;
        elements_starts[e_2] = n_2; elements_ends[e_2] = n_4;
        elements_starts[e_3] = n_1; elements_ends[e_3] = n_2;
        elements_starts[e_4] = n_3; elements_ends[e_4] = n_4;
        elements_starts[e_5] = n_3; elements_ends[e_5] = n_2;
        elements_starts[e_6] = n_2; elements_ends[e_6] = n_3_n;
        elements_starts[e_7] = n_2; elements_ends[e_7] = n_1_n;
        elements_starts[e_8] = n_4; elements_ends[e_8] = n_3_n;
      }

    const index_type n_1 = segments_count * 4 + 0;
    const index_type n_3 = segments_count * 4 + 1;

    nodes_xs[n_1] = segment_length * segments_count;
    nodes_xs[n_3] = segment_length * segments_count;

    nodes_ys[n_1] = bridge_height;
    nodes_ys[n_3] = bridge_height - section_height;

    const index_type e = segments_count * 8 + 0;
    elements_starts[e] = n_1; elements_ends[e] = n_3;
  }

private:
  index_type nodes_count {};
  std::unique_ptr<data_type[]> nodes_xs;
  std::unique_ptr<data_type[]> nodes_ys;

  std::unique_ptr<index_type[]> elements_starts;
  std::unique_ptr<index_type[]> elements_ends;

public:
  std::unique_ptr<bcsr_matrix_class<data_type, index_type>> matrix;
};

#endif //BLOCK_MATRIX_FORMAT_PERFORMANCE_GOLDEN_GATE_BRIDGE_H
