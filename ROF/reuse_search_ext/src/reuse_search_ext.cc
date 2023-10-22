#include "reuse_search_ext.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <set>
#include <fstream>
#include <string>

namespace py = pybind11;

// inline function to print current time
inline void print_current_time(std::string msg = "")
{
    time_t now = time(0);
    char *dt = ctime(&now);
    cout << msg << " " << dt << endl;
}

// inline function to convert std::vector to py::array_t
template <typename Sequence>
inline py::array_t<typename Sequence::value_type> as_pyarray(Sequence &&seq)
{
    auto size = seq.size();
    auto data = seq.data();
    std::unique_ptr<Sequence> seq_ptr =
        std::make_unique<Sequence>(std::move(seq));
    auto capsule = py::capsule(seq_ptr.get(), [](void *p)
                               { std::unique_ptr<Sequence>(reinterpret_cast<Sequence *>(p)); });
    seq_ptr.release();
    return py::array(size, data, capsule);
}

// inline function to convert py::array_t to std::vector
template <typename T>
inline std::vector<T> array_to_vector(const py::array_t<T> &input_array)
{
    const T *input_ptr = input_array.data();
    py::ssize_t array_size = input_array.size();
    return std::vector<T>(input_ptr, input_ptr + array_size);
}

////////////////////////////////////////////////////////////////
// following functions are just wrappers of the original cpp functions
// Do array to vector on input
// Do vector to array on output
////////////////////////////////////////////////////////////////

// return {indptr_src, indices_dst};
// note: we warp it as `py_inverse_csc_or_csr` in py11bind
static vector<py::array_t<uint32_t>>
py_inverse_csc(py::array_t<uint32_t> &indptr_dst,
               py::array_t<uint32_t> &indices_src,
               py::array_t<uint32_t> &indptr_src,
               py::array_t<uint32_t> &indices_dst)
{
    // convert py::array_t to std::vector
    auto indptr_dst_vec = array_to_vector(indptr_dst);
    auto indices_src_vec = array_to_vector(indices_src);
    auto indptr_src_vec = array_to_vector(indptr_src);
    auto indices_dst_vec = array_to_vector(indices_dst);

    // if the length of indptr_src and indices_dst not 0,assert an error and interrupt the program
    if (indptr_src_vec.size() != 0 || indices_dst_vec.size() != 0)
    {
        std::cout << "indptr_src and indices_dst should be empty" << std::endl;
        assert(0);
    }

    // call inverse_csc
    inverse_csc(indptr_dst_vec, indices_src_vec, indptr_src_vec, indices_dst_vec);

    // convert std::vector to py::array_t
    indptr_src = as_pyarray(std::move(indptr_src_vec));
    indices_dst = as_pyarray(std::move(indices_dst_vec));

    return {indptr_src, indices_dst};
}

// return {part_candidates};
// note: `part_candidates` and `indices`, decide all candidates of part
static vector<py::array_t<uint32_t>>
py_split_part(py::array_t<uint32_t> &part_candidates,
                py::array_t<uint32_t> &part2Node,
              py::array_t<uint32_t> &indptr_dst, 
              int part_size)
{
    // convert py::array_t to std::vector
    auto part_candidates_vec = array_to_vector(part_candidates);
    auto indptr_dst_vec = array_to_vector(indptr_dst);
    auto part2Node_vec = array_to_vector(part2Node);
    vector<uint32_t> num_parts_per_node_acc_vec;

    // call split_part
    split_part(part_candidates_vec, part2Node_vec, indptr_dst_vec, num_parts_per_node_acc_vec, part_size);

    // convert std::vector to py::array_t
    part_candidates = as_pyarray(std::move(part_candidates_vec));
    part2Node = as_pyarray(std::move(part2Node_vec));


    return {part_candidates, part2Node, as_pyarray(std::move(num_parts_per_node_acc_vec))};
}

// return {intersection_num_per_part};
// note: intersection_set_per_part (but some part is identical, so saving for all parts is redundant;) The probablity is minor, we just take all parts as cnadidates.
// static vector<py::array_t<uint32_t>>
static vector<py::list>
py_get_part_intersect(py::array_t<uint32_t> part_ptr_on_indices_src,
                      py::array_t<uint32_t> indices_src,
                      py::array_t<uint32_t> indptr_src,
                      py::array_t<uint32_t> indices_dst,

                      int part_size,
                      bool discard_remainder_part = false)
{
    // covert py::array_t to std::vector
    auto part_ptr_on_indices_src_vec = array_to_vector(part_ptr_on_indices_src);
    auto indices_src_vec = array_to_vector(indices_src);
    auto indptr_src_vec = array_to_vector(indptr_src);
    auto indices_dst_vec = array_to_vector(indices_dst);

    // call the C++ function
    auto intersection_per_part = get_part_intersect(part_ptr_on_indices_src_vec, indices_src_vec, indptr_src_vec, indices_dst_vec, part_size, discard_remainder_part);
    int num_parts = part_ptr_on_indices_src_vec.size() - 1;
    vector<uint32_t> intersection_num_per_part(num_parts);
#pragma omp parallel for
    for (int i = 0; i < num_parts; i++)
    {
        intersection_num_per_part[i] = intersection_per_part[i].size();
    }
    py::list intersection_num_per_part_list;
    intersection_num_per_part_list.append(as_pyarray(std::move(intersection_num_per_part)));

    // covert vector of std::vector to list of py::array_t
    py::list part_intersect_list;
    for (auto &intersection_set : intersection_per_part)
    {
        part_intersect_list.append(as_pyarray(std::move(intersection_set)));
    }
    return {part_intersect_list, intersection_num_per_part_list};
}

// return {index_ordered, index_picked};
static vector<py::array_t<uint32_t>>
py_greedy_max_reuse(py::array_t<uint32_t> intersection_num_per_part,
                    py::array_t<uint32_t> part_ptr_on_indices_src,
                    py::array_t<uint32_t> indices_src)
{
    // covert py::array_t to std::vector
    auto intersection_num_per_part_vec = array_to_vector(intersection_num_per_part);
    auto part_ptr_on_indices_src_vec = array_to_vector(part_ptr_on_indices_src);
    auto indices_src_vec = array_to_vector(indices_src);

    // call the C++ function
    auto return_vecs = greedy_max_reuse(intersection_num_per_part_vec, part_ptr_on_indices_src_vec, indices_src_vec);

    // covert std::vector to py::array_t
    return {as_pyarray(std::move(return_vecs[0])), as_pyarray(std::move(return_vecs[1]))};
}

////// current attention ///////

// a inline function, input is

// when considering picked parts, you need to

// return new_part_ptr_on_indices_src, new_indices_src
static vector<py::array_t<uint32_t>>
py_reorganize_indices(py::array_t<uint32_t> part_ptr_on_indices_src,
                      py::array_t<uint32_t> indices_src,

                      py::list intersection_per_part,
                      py::array_t<uint32_t> index_picked,
                      py::array_t<uint32_t> indptr_dst,
                      py::array_t<uint32_t> num_parts_per_node_acc,
                      int parts_size)
{
    // convert py::array_t to std::vector
    auto part_ptr_on_indices_src_vec = array_to_vector(part_ptr_on_indices_src);
    auto indices_src_vec = array_to_vector(indices_src);

    vector<vector<uint32_t>> intersection_per_part_vec;
    for (auto &intersection_set : intersection_per_part)
    {
        intersection_per_part_vec.push_back(array_to_vector(py::cast<py::array_t<uint32_t>>(intersection_set)));
    }
    auto index_picked_vec = array_to_vector(index_picked);
    auto indptr_dst_vec = array_to_vector(indptr_dst);
    auto num_parts_per_node_acc_vec = array_to_vector(num_parts_per_node_acc);

    vector<uint32_t> column_index_cache_vec;
    vector<uint32_t> part_pointers_cache_vec;
    vector<uint32_t> part_cache_flagindex_vec;

    // call the C++ function
    reorganize_indices(part_ptr_on_indices_src_vec, indices_src_vec,
                       intersection_per_part_vec, index_picked_vec,
                       indptr_dst_vec,
                       column_index_cache_vec,
                       part_pointers_cache_vec,
                       part_cache_flagindex_vec,
                       num_parts_per_node_acc_vec,
                       parts_size);

    // convert std::vector to py::array_t

    return {as_pyarray(std::move(indices_src_vec)), as_pyarray(std::move(part_ptr_on_indices_src_vec)), 
    as_pyarray(std::move(part_pointers_cache_vec)), as_pyarray(std::move(column_index_cache_vec)), 
    as_pyarray(std::move(part_cache_flagindex_vec))};
}

// py_reorganize_indices_uniquely()
static vector<py::array_t<uint32_t>>
py_reorganize_indices_uniquely(py::array_t<uint32_t> part_ptr_on_indices_src,
                      py::array_t<uint32_t> indices_src,

                      py::list intersection_per_part,
                      py::array_t<uint32_t> index_picked,
                      py::array_t<uint32_t> indptr_dst,
                      py::array_t<uint32_t> num_parts_per_node_acc,
                      int parts_size)
{
    // convert py::array_t to std::vector
    auto part_ptr_on_indices_src_vec = array_to_vector(part_ptr_on_indices_src);
    auto indices_src_vec = array_to_vector(indices_src);

    vector<vector<uint32_t>> intersection_per_part_vec;
    for (auto &intersection_set : intersection_per_part)
    {
        intersection_per_part_vec.push_back(array_to_vector(py::cast<py::array_t<uint32_t>>(intersection_set)));
    }
    auto index_picked_vec = array_to_vector(index_picked);
    auto indptr_dst_vec = array_to_vector(indptr_dst);
    auto num_parts_per_node_acc_vec = array_to_vector(num_parts_per_node_acc);

    vector<uint32_t> column_index_cache_vec;
    vector<uint32_t> part_pointers_cache_vec;
    vector<uint32_t> part2Node_ptr_vec;
    vector<uint32_t> part2Node_col_vec;

    // call the C++ function
    reorganize_indices_uniquely(part_ptr_on_indices_src_vec, indices_src_vec,
                       intersection_per_part_vec, index_picked_vec,
                       indptr_dst_vec,
                       column_index_cache_vec,
                       part_pointers_cache_vec,
                       part2Node_ptr_vec,
                       part2Node_col_vec,
                       num_parts_per_node_acc_vec,
                       parts_size);

    // convert std::vector to py::array_t

    return {as_pyarray(std::move(column_index_cache_vec)), as_pyarray(std::move(part_pointers_cache_vec)),
    as_pyarray(std::move(part2Node_ptr_vec)), as_pyarray(std::move(part2Node_col_vec))};
}


PYBIND11_MODULE(reuse_search_ext, m)
{
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("py_inverse_csc_or_csr", &py_inverse_csc, "function to inverse csc or csr");
    m.def("py_split_part", &py_split_part, "function to split part");
    m.def("py_get_part_intersect", &py_get_part_intersect, "function to get neighbor intersection(list of np.array) and its size(np.array of size) of every part");
    m.def("py_greedy_max_reuse", &py_greedy_max_reuse, "function to get the ordered index of candidates parts and the order of every part being picked(start from 1, 0 is not picked)");
    m.def("py_reorganize_indices", &py_reorganize_indices, "function to reorganize the indices of every nodes according to the picked parts");
    m.def("py_reorganize_indices_uniquely", &py_reorganize_indices_uniquely, "function to (part-wise uniquely) construct the src_nodes(what nodes in part) and dst_node(s)(for resue part), the reuse part is at the beginning");

}