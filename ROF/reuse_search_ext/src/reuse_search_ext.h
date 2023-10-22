#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numeric>
#include <omp.h>
#include <set>
#include <string>
#include <vector>
#include <queue>

using namespace std;


void inverse_csc(std::vector<uint32_t> &indptr_dst,
				 std::vector<uint32_t> &indices_src,
				 std::vector<uint32_t> &indptr_src,
				 std::vector<uint32_t> &indices_dst)
{
	/*
	Example:
	// row_ptr  = {0, 1, 3, 6, 8, 9}
	// col_idx  = {1, 0, 2, 0, 1, 3, 2, 4, 3}
	(the former tow args)

	// new_row_ptr = {0, 2, 4, 6, 8, 9}
	// new_col_idx = {1, 2, 0, 2, 1, 3, 2, 4, 3}
	(the latter tow args)
	*/

	int num_nodes = indptr_dst.size() - 1;
	int loop_num_nodes = num_nodes;
	int num_edges = indices_src.size();
	// vector of vector
	vector<vector<uint32_t>> indices_dst_vec(num_nodes);
// openmp
#pragma omp parallel for
	for (int i = 0; i < loop_num_nodes; i++)
	{
#pragma omp parallel for
		for (int j = indptr_dst[i]; j < static_cast<int>(indptr_dst[i + 1]); j++)
		{
#pragma omp critical
			indices_dst_vec[indices_src[j]].push_back(i);
		}
	}
	cout << "inverse in vector of vector ends..." << endl;

// sort all vector in indices_dst_vec with omp, in increasing order
#pragma omp parallel for
	for (int i = 0; i < loop_num_nodes; i++)
	{
		sort(indices_dst_vec[i].begin(), indices_dst_vec[i].end());
	}
	cout << "sort all vector in indices_dst_vec ends..." << endl;

	// accumulate the size of vector in indices_dst_vec to get indptr_src
	indptr_src.resize(num_nodes + 1);
	indptr_src[0] = 0;
	for (int i = 0; i < num_nodes; i++)
	{
		indptr_src[i + 1] = indptr_src[i] + indices_dst_vec[i].size();
	}
	cout << "accumulate for indptr_src ends..." << endl;

	// concatenate the vector in indices_dst_vec to get indices_dst
	indices_dst.resize(num_edges);
#pragma omp parallel for
	for (int i = 0; i < loop_num_nodes; i++)
	{
		memcpy(indices_dst.data() + indptr_src[i], indices_dst_vec[i].data(),
			   indices_dst_vec[i].size() * sizeof(uint32_t));
	}
	cout << "concatenate for indices_dst ends..." << endl;
}


// generate part candiates in natural way, split indptr_dst into part_candidates, without change of indices
void split_part(vector<uint32_t> &part_candidates, vector<uint32_t> &part2Node, vector<uint32_t> &indptr_dst, vector<uint32_t> &num_parts_per_node_acc,
				int part_size)
{
	// print size of all inputs
	cout << "] size of part_candidates: " << part_candidates.size() << endl;
	cout << "] size of indptr_dst: " << indptr_dst.size() << endl;
	cout << "] part_size: " << part_size << endl;
    // print ten elements of indptr_dst
	cout << "ten elements of indptr_dst: ";
	for (int i = 0; i < 10; i++)
	{
		cout << indptr_dst[i] << " ";
	}

	int num_parts = 0;
	int num_nodes = indptr_dst.size() - 1;
	num_parts_per_node_acc.resize(num_nodes);
#pragma omp parallel for reduction(+ : num_parts)
	for (int i = 0; i < num_nodes; i++)
	{
		int span = indptr_dst[i + 1] - indptr_dst[i];
		int pn = (span + part_size - 1) / part_size;
		num_parts_per_node_acc[i] = pn;
		num_parts += pn;
	}
	// accumulate num_parts_per_node_acc
	for (int i = 1; i < num_nodes; i++)
	{
		num_parts_per_node_acc[i] += num_parts_per_node_acc[i - 1];
	}

	// print last element of num_parts_per_node_acc
	cout << "last element of num_parts_per_node_acc : "
		 << num_parts_per_node_acc[num_nodes - 1] << endl;

    // resize part_candidates to num_parts + 1, and set first element to 0
    // part_candidates is indptr of all parts
	part_candidates.resize(num_parts + 1);
	part_candidates[0] = 0;

	part2Node.resize(num_parts);

    // when using indptr, it is csr, and the node denoted in indptr is ordered
#pragma omp parallel for
	for (int i = 0; i < num_nodes; i++)
	{
		int span = indptr_dst[i + 1] - indptr_dst[i];
		int num_parts = (span + part_size - 1) / part_size;
		// int old_size = part2Node.size();
		// part2Node.resize(old_size + num_parts);
		// std::fill(part2Node.begin() + old_size, part2Node.end(), i);

		int base_parts_num = num_parts_per_node_acc[i] - num_parts;
		int base_indice = indptr_dst[i];

		std::fill(part2Node.begin() + base_parts_num, part2Node.begin() + base_parts_num + num_parts, i);
#pragma omp parallel for
		for (int j = 0; j < num_parts; j++)
		{
			if (j == num_parts - 1)
				part_candidates[base_parts_num + j + 1] = indptr_dst[i + 1];
			else
				part_candidates[base_parts_num + j + 1] =
					base_indice + (j + 1) * part_size;
		}
	}
}


// get_part_intersection
vector<vector<uint32_t>> get_part_intersect(vector<uint32_t> &part_ptr_on_indices_src,
									vector<uint32_t> &indices_src,
									
									vector<uint32_t> &indptr_src,
									vector<uint32_t> &indices_dst,

									int part_size,
									bool discard_remainder_part = false)
{
	cout << "discard_remainder_part: " << discard_remainder_part << endl;
	int num_parts = part_ptr_on_indices_src.size() - 1;
	cout << "candidate part num: " << num_parts << endl;
	// vector<uint32_t> intersection_num_per_part(num_parts);
	vector<vector<uint32_t>> intersection_per_part(num_parts);

	int loop_num_parts = num_parts;
	cout << "loop_num_parts: " << loop_num_parts << endl;

#pragma omp parallel for
	for (int i = 0; i < loop_num_parts; i++)
	{
		// find indices_dst intersection of all nodes in part i
		// int intersection_num = 0;
		int part_start = part_ptr_on_indices_src[i];
		int part_end = part_ptr_on_indices_src[i + 1];
		if (discard_remainder_part && part_end - part_start != part_size)
		{
			intersection_per_part[i] = vector<uint32_t>();
			continue;
		}

		uint32_t cur_node = indices_src[part_start];
		
		auto cur_node_set_bg = indices_dst.begin() + indptr_src[cur_node];
		auto cur_node_set_ed = indices_dst.begin() + indptr_src[cur_node + 1];

		// // print the size of cur_node_set_bg, cur_node_set_ed
		// cout << "i: " << i << " , cur_node_set_bg: " << indptr_src[cur_node]
		//      << " , cur_node_set_ed: " << indptr_src[cur_node + 1];
		// cout << "   " << "cur_node_set_ed-cur_node_set_bg: "
		//      << indptr_src[cur_node + 1] - indptr_src[cur_node] << endl;

		set<uint32_t> cur_intersection(cur_node_set_bg, cur_node_set_ed);

#pragma omp parallel for
		for (int j = part_start+1; j < part_end; j++)
		{
			cur_node = indices_src[j];

			// find intersection of cur_node
			auto cur_node_set_bg = indices_dst.begin() + indptr_src[cur_node];
			auto cur_node_set_ed = indices_dst.begin() + indptr_src[cur_node + 1];
			set<uint32_t> cur_node_set(cur_node_set_bg, cur_node_set_ed);

			// if (cur_node_set.size() > 1 || cur_intersection.size() > 1)
			//   break;
			// #pragma omp critical
			// set_intersection(cur_node_set, cur_intersection);
			// cur_intersection = intersection(cur_node_set, cur_intersection);

			set<uint32_t> intersection;
			set_intersection(cur_node_set.begin(), cur_node_set.end(),
							 cur_intersection.begin(), cur_intersection.end(),
							 inserter(intersection, intersection.begin()));
			cur_intersection = intersection;

			// if (cur_intersection.size() == 0)
			//   break;
		}
		// intersection_num_per_part[i] = cur_intersection.size();
		// only save the intersection size, but I also want the content of the intersection
		intersection_per_part[i] = vector<uint32_t>(cur_intersection.begin(), cur_intersection.end());
	}
	cout << "intersection over" << endl;

	return intersection_per_part;
}


// greedy max reuse
vector<vector<uint32_t>> greedy_max_reuse(vector<uint32_t> &intersection_num_per_part,
					 vector<uint32_t> &part_ptr_on_indices_src,
					 vector<uint32_t> &indices_src,
					 uint32_t pick_threshold=2)
{
	// print the size of all parameters
	cout << "intersection_num_per_part.size(): "
		 << intersection_num_per_part.size() << endl;
	cout << "part_ptr_on_indices_src.size(): " << part_ptr_on_indices_src.size()
		 << endl;

	// get the index_ordered of element from largest to smallest in
	// intersection_num_per_part
	vector<uint32_t> index_ordered(intersection_num_per_part.size());
	// initialize empty vector index_picked
	vector<uint32_t> index_picked; // 
	// int part_reuse_amount = 0;

	// iota(index_ordered.begin(), index_ordered.end(), 0); // equivalent to the
	// following openmp implementation

	int start_value = 0;
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(index_ordered.size()); i++)
	{
		index_ordered[i] = start_value + i;
	}

	sort(index_ordered.begin(), index_ordered.end(),
		 [&intersection_num_per_part](size_t i1, size_t i2)
		 {
			 return intersection_num_per_part[i1] > intersection_num_per_part[i2];
		 });
	cout << "sort over" << endl;
	// a set contains all the nodes that unconflicted
	set<uint32_t> unconflicted_nodes;

	int part_idx = 0;
	for (int i = 0; i < static_cast<int>(index_ordered.size()); i++)
	{
		if (intersection_num_per_part[index_ordered[i]] < pick_threshold)
			break;
		part_idx = index_ordered[i];
		int part_start = part_ptr_on_indices_src[part_idx];
		int part_end = part_ptr_on_indices_src[part_idx + 1];
		// put the nodes from part_start to part_end into a set
		set<uint32_t> cur_node_set(indices_src.begin() + part_start,
								   indices_src.begin() + part_end);

		// check if the nodes in cur_node_set is conflicted with the nodes in unconflicted_nodes
		bool is_conflicted = false;
		for (auto it = cur_node_set.begin(); it != cur_node_set.end(); it++)
		{
			if (unconflicted_nodes.find(*it) != unconflicted_nodes.end())
			{
				is_conflicted = true;
				break;
			}
		}

		// if not conflicted, add the nodes in cur_node_set to unconflicted_nodes
		if (!is_conflicted)
		{
			index_picked.push_back(part_idx);
			unconflicted_nodes.insert(cur_node_set.begin(), cur_node_set.end());
			// part_reuse_amount += intersection_num_per_part[part_idx] - 1;
		}
	}

	// return part_reuse_amount;
	return {index_ordered, index_picked};
}

// greedy max reuse

/*

Output: `picked_C` = {}(`empty`), `updated_part_intersect_list` = {[],...}(`part_intersect_list`)
Fuction: get_conflict_parts_of()
1. Init a max heap `H` of candidates, ordered by the candidate's `intersection_list` size in current `updated_part_intersect_list`
2. 
while (c=H.get_max_part() with >2 length):
	H.remove(c)
	for cc in get_conflict_parts_of(c):
		updated_part_intersect_list[cc] = updated_part_intersect_list[cc] - updated_part_intersect_list[c]
		H.update_value(cc)
*/
vector<vector<uint32_t>> dynamic_greedy_max_reuse(vector<uint32_t> &intersection_num_per_part,
					 vector<uint32_t> &part_ptr_on_indices_src,
					 vector<uint32_t> &indices_src,
					 vector<vector<uint32_t>> &intersection_per_part,
					 uint32_t pick_threshold=2)
{

	// Define a struct for the elements in the queue
	struct Element {
	uint32_t intersection_size;
	uint32_t intersection_idx;
	};

	// Define a custom comparator for the priority queue
	struct Compare {
	bool operator()(const Element& a, const Element& b) const {
		return a.intersection_size > b.intersection_size;
	}
	};

	std::priority_queue<Element, std::vector<Element>, Compare> min_heap;

	// print size of intersection_per_part
	cout << "intersection_per_part.size(): " << intersection_per_part.size() << endl;
	for (uint32_t idx = 0; idx < intersection_per_part.size(); idx++)
	{
		min_heap.push(Element{intersection_num_per_part[idx], idx});
	}

	// get the index_ordered of element from largest to smallest in
	// intersection_num_per_part
	vector<uint32_t> index_ordered(intersection_num_per_part.size());
	// initialize empty vector index_picked
	vector<uint32_t> index_picked; 
	// int part_reuse_amount = 0;

	int start_value = 0;
#pragma omp parallel for
	for (int i = 0; i < static_cast<int>(index_ordered.size()); i++)
	{
		index_ordered[i] = start_value + i;
	}


	sort(index_ordered.begin(), index_ordered.end(),
		 [&intersection_num_per_part](size_t i1, size_t i2)
		 {
			 return intersection_num_per_part[i1] > intersection_num_per_part[i2];
		 });
	cout << "sort over" << endl;
	// a set contains all the nodes that unconflicted
	set<uint32_t> unconflicted_nodes;

	int part_idx = 0;
	for (int i = 0; i < static_cast<int>(index_ordered.size()); i++)
	{
		if (intersection_num_per_part[index_ordered[i]] < pick_threshold)
			break;
		part_idx = index_ordered[i];
		int part_start = part_ptr_on_indices_src[part_idx];
		int part_end = part_ptr_on_indices_src[part_idx + 1];
		// put the nodes from part_start to part_end into a set
		set<uint32_t> cur_node_set(indices_src.begin() + part_start,
								   indices_src.begin() + part_end);

		// check if the nodes in cur_node_set is conflicted with the nodes in unconflicted_nodes
		bool is_conflicted = false;
		for (auto it = cur_node_set.begin(); it != cur_node_set.end(); it++)
		{
			if (unconflicted_nodes.find(*it) != unconflicted_nodes.end())
			{
				is_conflicted = true;
				break;
			}
		}

		// if not conflicted, add the nodes in cur_node_set to unconflicted_nodes
		if (!is_conflicted)
		{
			index_picked.push_back(part_idx);
			unconflicted_nodes.insert(cur_node_set.begin(), cur_node_set.end());
			// part_reuse_amount += intersection_num_per_part[part_idx] - 1;
		}
	}

	// return part_reuse_amount;
	return {index_ordered, index_picked};
}


void reorganize_indices(
	vector<uint32_t> &part_ptr_on_indices_src,
	vector<uint32_t> &indices_src,

	vector<vector<uint32_t>> &intersection_per_part,
	vector<uint32_t> &index_picked,
	vector<uint32_t> &indptr_dst,

	vector<uint32_t> &column_index_cache,
	vector<uint32_t> &part_pointers_cache,
	vector<uint32_t> &part_cache_flagindex,

	vector<uint32_t> &num_parts_per_node_acc,
	int part_size)
{
	// copy indptr_dst to cur_reorg_pos
	vector<uint32_t> cur_reorg_pos(indptr_dst.begin(), indptr_dst.end());
	vector<uint32_t> indices_src_copy(indices_src.begin(), indices_src.end());
	
	part_pointers_cache.push_back(0);
	part_cache_flagindex.resize(part_ptr_on_indices_src.size()-1);
	num_parts_per_node_acc.back() = 0;

	// part_cache_flagindex, 
	// part_pointers_cache, column_index_cache 
	

	// loop index_picked, to reorganize picked parts one by one.
	for (int i = 0; i < static_cast<int>(index_picked.size()); i++)
	{
		int part_idx = index_picked[i];
		int part_start = part_ptr_on_indices_src[part_idx];
		int part_end = part_ptr_on_indices_src[part_idx + 1];
		if (part_end - part_start != part_size)
		{
			cout << "part size is not equal to part_size" << endl;
		}

		// know whose neighbor contain current part
		auto intersection_of_part = intersection_per_part[part_idx];

		for (int j = 0; j < static_cast<int>(intersection_of_part.size()); j++)
		{
			int dst_node_idx = intersection_of_part[j];

			// please change the reuse number of non-PartSize parts to 0; ✅
			// and not to pick 0 reuse part in greedy max. ✅

			// how to create new part_ptr 
			// sol-1. if every picked part has fixed size(PartSize), then we can directly use the raw part_ptr!! 
			// sol-2. otherwise, we need to record every step reorganization, that's not good. 
			int reorg_start = cur_reorg_pos[dst_node_idx];
			int neighbor_end = indptr_dst[dst_node_idx + 1];
			
			// int old_reorg_start = reorg_start;
			for (int k = part_start; k < part_end; k++)
			{
				// find indices_src[k] in dst_node_idx's neighbor, and swap it to the front of dst_node_idx's neighbor
				auto pos = find(indices_src.begin() + reorg_start, indices_src.begin() + neighbor_end, indices_src_copy[k]);

				if (pos != indices_src.begin() + neighbor_end)
				{
					swap(*pos, indices_src[reorg_start]);
					reorg_start++;
				}
				else
				{
					cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << endl;
					cout << "cur_reorg_pos[dst_node_idx]: " << cur_reorg_pos[dst_node_idx] << endl;
					cout << "indptr_dst[dst_node_idx]: " << indptr_dst[dst_node_idx] << endl;
					cout << "indptr_dst[dst_node_idx + 1]: " << indptr_dst[dst_node_idx + 1] << endl;
					cout << "impossible error: 'pos == indices_src.begin() + neighbor_end' " << endl;
					cout << "-----------------------------------" << endl;
				}
			}


			//  represent edge for cache
			if (reorg_start - static_cast<int>(cur_reorg_pos[dst_node_idx]) != part_size)
			{
				cout << "impossible error: 'reorg_start - cur_reorg_pos[dst_node_idx] != part_size' " << endl;
			}
			cur_reorg_pos[dst_node_idx] = reorg_start;

			// i+1
			int num_nodes = num_parts_per_node_acc.size();
			part_cache_flagindex[num_parts_per_node_acc[(dst_node_idx-1+num_nodes)%num_nodes]] = i + 1;
			num_parts_per_node_acc[(dst_node_idx-1+num_nodes)%num_nodes] += 1;

		}

		// append indice_src_copy[part_start:part_end] to column_index_cache
		column_index_cache.insert(column_index_cache.end(), indices_src_copy.begin() + part_start, indices_src_copy.begin() + part_end);
		// according ptr; But, if only accept part with PartSize, then we don not need it, just 3*id
		part_pointers_cache.push_back(column_index_cache.size());
	}
}

void reorganize_indices_uniquely(
	vector<uint32_t> &part_ptr_on_indices_src,
	vector<uint32_t> &indices_src,

	vector<vector<uint32_t>> &intersection_per_part,
	vector<uint32_t> &index_picked,
	vector<uint32_t> &indptr_dst, // denote the src range of every dst node

	vector<uint32_t> &column_index_cache, //vector<uint32_t> &uniqpart_column_index,
	vector<uint32_t> &part_pointers_cache, //vector<uint32_t> &uniqpart_pointers,
	// vector<uint32_t> &part_cache_flagindex,
	vector<uint32_t> &part2Node_ptr,
	vector<uint32_t> &part2Node_col,

	vector<uint32_t> &num_parts_per_node_acc,
	int part_size)
{
	// copy indptr_dst to cur_reorg_pos
	vector<uint32_t> cur_reorg_pos(indptr_dst.begin(), indptr_dst.end());
	vector<uint32_t> indices_src_copy(indices_src.begin(), indices_src.end());
	
	part_pointers_cache.push_back(0);
	// part_cache_flagindex.resize(part_ptr_on_indices_src.size()-1);
	num_parts_per_node_acc.back() = 0;

	// part_cache_flagindex, 
	// part_pointers_cache, column_index_cache 
	part2Node_ptr.push_back(0);

	// loop index_picked, to reorganize picked parts one by one.
	for (int i = 0; i < static_cast<int>(index_picked.size()); i++)
	{
		int part_idx = index_picked[i];
		int part_start = part_ptr_on_indices_src[part_idx];
		int part_end = part_ptr_on_indices_src[part_idx + 1];
		if (part_end - part_start != part_size)
		{
			cout << "part size is not equal to part_size" << endl;
		}

		// know whose neighbor contain current part
		auto intersection_of_part = intersection_per_part[part_idx];
		int size_of_intersection = static_cast<int>(intersection_of_part.size());
		part2Node_ptr.push_back(part2Node_ptr.back() + size_of_intersection);
		part2Node_col.insert(part2Node_col.end(), intersection_of_part.begin(), intersection_of_part.end());

		for (int j = 0; j < size_of_intersection; j++)
		{
			int dst_node_idx = intersection_of_part[j];

			// please change the reuse number of non-PartSize parts to 0; ✅
			// and not to pick 0 reuse part in greedy max. ✅

			// how to create new part_ptr 
			// sol-1. if every picked part has fixed size(PartSize), then we can directly use the raw part_ptr!! 
			// sol-2. otherwise, we need to record every step reorganization, that's not good. 
			int reorg_start = cur_reorg_pos[dst_node_idx];
			int neighbor_end = indptr_dst[dst_node_idx + 1];
			
			// int old_reorg_start = reorg_start;
			for (int k = part_start; k < part_end; k++)
			{
				// find indices_src[k] in dst_node_idx's neighbor, and swap it to the front of dst_node_idx's neighbor
				auto pos = find(indices_src.begin() + reorg_start, indices_src.begin() + neighbor_end, indices_src_copy[k]);

				if (pos != indices_src.begin() + neighbor_end)
				{
					swap(*pos, indices_src[reorg_start]);
					reorg_start++;
				}
				else
				{
					cout << "cur_reorg_pos[dst_node_idx]: " << cur_reorg_pos[dst_node_idx] << endl;
					cout << "indptr_dst[dst_node_idx]: " << indptr_dst[dst_node_idx] << endl;
					cout << "indptr_dst[dst_node_idx + 1]: " << indptr_dst[dst_node_idx + 1] << endl;
					cout << "impossible error: 'pos == indices_src.begin() + neighbor_end' " << endl;
				}
			}



			if (reorg_start - static_cast<int>(cur_reorg_pos[dst_node_idx]) != part_size)
			{
				cout << "impossible error: 'reorg_start - cur_reorg_pos[dst_node_idx] != part_size' " << endl;
			}
			cur_reorg_pos[dst_node_idx] = reorg_start;

			// i+1
			int num_nodes = num_parts_per_node_acc.size();
			// part_cache_flagindex[num_parts_per_node_acc[(dst_node_idx-1+num_nodes)%num_nodes]] = i + 1; // a certain part use a certain cache
			num_parts_per_node_acc[(dst_node_idx-1+num_nodes)%num_nodes] += 1;

		}

		// append indice_src_copy[part_start:part_end] to column_index_cache
		column_index_cache.insert(column_index_cache.end(), indices_src_copy.begin() + part_start, indices_src_copy.begin() + part_end);
		// according ptr; But, if only accept part with PartSize, then we don not need it, just 3*id
		part_pointers_cache.push_back(column_index_cache.size());
	}

	for (int node = 0; node < static_cast<int>(cur_reorg_pos.size())-1; node++)
	{
			// 		int reorg_start = cur_reorg_pos[dst_node_idx];
			// int neighbor_end = indptr_dst[dst_node_idx + 1];
		int cur_node_parts_start = cur_reorg_pos[node];
		int cur_node_parts_end = indptr_dst[node+1];
		if (cur_node_parts_end == cur_node_parts_start)
		{
			continue;
		}
		int base_column_index_cache = column_index_cache.size();
		column_index_cache.insert(column_index_cache.end(), indices_src.begin() + cur_node_parts_start, indices_src.begin() + cur_node_parts_end);
		int remain_parts_num = (cur_node_parts_end - cur_node_parts_start + part_size - 1) / part_size;
		// int remain_nodes_num = (cur_node_parts_end - cur_node_parts_start) - (remain_parts_num - 1) * part_size;

		for (int n_rma_p = 1; n_rma_p < remain_parts_num; n_rma_p++)
		{
			part_pointers_cache.push_back(base_column_index_cache + n_rma_p * part_size);
			part2Node_col.push_back(node);
			part2Node_ptr.push_back(part2Node_col.size());
		}
		part_pointers_cache.push_back(base_column_index_cache + (cur_node_parts_end - cur_node_parts_start));
		part2Node_col.push_back(node);
		part2Node_ptr.push_back(part2Node_col.size());
		// check part2Node_ptr and part_pointers_cache has the same size
		if (part2Node_ptr.size() != part_pointers_cache.size())
		{
			cout << "part2Node_ptr.size() != part_pointers_cache.size()" << endl;
		}
	}



}