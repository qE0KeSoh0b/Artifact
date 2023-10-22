SERVER = 
REPO = 

import argparse
from termcolor import colored
import os
import time
def t_click(msg, t0):
    t1 = time.time()
    print(msg, t1-t0, f"({t1})" )
    return t1, t1-t0
t_stamp = time.time()
argparser = argparse.ArgumentParser()
# argparser.add_argument('--kernel_type', '-kt', type=str, default='ROF_v1')
# argparser.add_argument('--n_hidden', '-nh', type=int, default=16)
argparser.add_argument('--dim_input', '-di', type=int, default=1433)
argparser.add_argument('--dim_output', '-do', type=int, default=7)
argparser.add_argument('--datasetname', '-d', type=str, default='cora')
argparser.add_argument('--part_size', '-ps', type=int, default=2)
argparser.add_argument('--reorder_flag', '-rf', type=int, default=0)
# warmup, runs
argparser.add_argument('--warmup', '-w', type=int, default=10)
argparser.add_argument('--runs', '-r', type=int, default=100)
args = argparser.parse_args()
'''
env CUDA_VISIBLE_DEVICES=1 python small_g_search.py -d cora -ps 2 -rf 1
'''

log_file_name = "small_g_search"
# get short name of args
for arg in vars(args):
    log_file_name += f"-{arg}-{getattr(args, arg)}"
log_file_name += ".log"

t_delta_list = []
t_stamp_list = []
t_name_list = []

import sys
import socket

hostname = socket.gethostname()
print("hostname:", hostname)
if hostname == SERVER:
    sys.path.append(f"{REPO}/ROF/reuse_search_ext/src/")
    sys.path.append(f"{REPO}/ROF/reuse_conv/")
    sys.path.append(f"{REPO}/ROF/") 
    sys.path.append(f"{REPO}") # for utils

    data_root = f"{REPO}/datasets"
    path = f"{REPO}/ROF_evaluation/reuse/{log_file_name}"

else:
    assert False, "please add your import path in this server"
    # get current directory of this file
    cur_dir = sys.path[0]
    sys.path.append(cur_dir + "/reuse_search_ext/src/")
    sys.path.append(cur_dir + "/reuse_conv/")
    sys.path.append("..")

print("log_file_name for reused ratio:", log_file_name)

from utils.plot_utils import plot_training_time

import numpy as np
import torch
from scipy.sparse import coo_matrix
import torch.nn.functional as F

from data_prepare import load_graph
import reuse_search_ext
# print module
print(reuse_search_ext)

from ROF_models import ROF_GCN,ROF_inputInfo
from reuse_conv import ROF_GCNConv
from torch_geometric.nn import GCNConv,SAGEConv
t_stamp,t_delta = t_click("-------------------- import over: ",  t_stamp)
t_name_list.append("import over")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)

# must first import torch
import ROF


#################### test load_graph ####################
##                                                     ##
datasetname = args.datasetname
# datasetname = "cora"
# datasetname = "citeseer"
# datasetname = "pubmed"
features, edge_index, labels, train_mask, val_mask, test_mask = load_graph(
    data_root= data_root,
    rabbit_reorder_flag=args.reorder_flag,
    graph_name=datasetname,
    device='cpu',
    other_require=None)
#########################################################

# after optional reorder, convert to coo, and then convert to csr
num_nodes = features.shape[0]
num_edges = edge_index.shape[1]
val = [1] * num_edges
scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
scipy_csr = scipy_coo.tocsr()

# 1. in edge_index, upper is src, lower is dst; `coo` is the same
# 2. convert `coo` to `csr`, then indices is dst; i.e., indices mean out edge of node 0,1,2...
# 3. convert `csr` to array, then the row is src, col is dst.
indptr_src = scipy_csr.indptr
indices_dst = scipy_csr.indices

# because it is symmetric, otherwise we need inverse
indptr_dst = indptr_src
indices_src = indices_dst
# ...
print("*indices_src.shape:", indices_src.shape)

t_stamp,t_delta = t_click("-------------------- load graph and convert csr over: ",  t_stamp)
t_name_list.append("load graph and convert csr over")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)

# search resure structure in csr format
a = np.empty(0, dtype=np.int32)
b = np.empty(0, dtype=np.int32)
part_size = args.part_size
part_ptr_on_indices_src, part2Node, num_parts_per_node_acc = reuse_search_ext.py_split_part(a, b, indptr_dst,
                                                     part_size)
# import pdb;pdb.set_trace()
t_stamp,t_delta = t_click("-------------------- reuse_search_ext.py_split_part over: ",  t_stamp)
t_name_list.append("reuse_search_ext.py_split_part over")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)


##########

# discard_remainder_part = False
# part_intersect_list, intersection_num_per_part = reuse_search_ext.py_get_part_intersect(
#     part_ptr_on_indices_src, indices_src, indptr_src, indices_dst,
#     part_size, discard_remainder_part)
# t_stamp,t_delta = t_click("-------------------- reuse_search_ext.py_get_part_intersect over(discard False): ",  t_stamp)
# t_name_list.append("reuse_search_ext.py_get_part_intersect over(discard False)")
# t_delta_list.append(t_delta)
# t_stamp_list.append(t_stamp)

discard_remainder_part = True
# discard_remainder_part = False
part_intersect_list, intersection_num_per_part = reuse_search_ext.py_get_part_intersect(
    part_ptr_on_indices_src, indices_src, indptr_src, indices_dst,
    part_size, discard_remainder_part)
t_stamp,t_delta = t_click("-------------------- reuse_search_ext.py_get_part_intersect over(discard True): ",  t_stamp)
t_name_list.append("reuse_search_ext.py_get_part_intersect over(discard True)")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)

check_num = []
for item in part_intersect_list:
    check_num.append(item.shape[0])
print("check_num vs intersection_num_per_part:", np.abs(np.array(check_num)-intersection_num_per_part[0]).sum())
print("intersection_num_per_part[0].sum() = ", intersection_num_per_part[0].sum())
print("len(intersection_num_per_part[0]) [intersection_num_per_part[0]>0].sum() = ", len(intersection_num_per_part[0]), intersection_num_per_part[0].sum())
print("(intersection_num_per_part[0]>2).sum() = ",(intersection_num_per_part[0]>2).sum())
print("intersection_num_per_part[0][intersection_num_per_part[0]>2].sum() = ",intersection_num_per_part[0][intersection_num_per_part[0]>2].sum())
# import pdb;pdb.set_trace()

##########
# HINT: here compute the reuse ratio
index_ordered, index_picked = reuse_search_ext.py_greedy_max_reuse(
    intersection_num_per_part, part_ptr_on_indices_src, indices_src)
print("index_ordered.shape:", index_ordered.shape)
print("index_picked.shape:", index_picked.shape, index_picked.dtype)

reused_part_num = intersection_num_per_part[0][index_picked].sum()
reused_part_reduction = (intersection_num_per_part[0][index_picked]-1).sum()
reused_edge_num = intersection_num_per_part[0][index_picked].sum()*part_size
reused_edge_reduction = (intersection_num_per_part[0][index_picked]-1).sum()*part_size
raw_part_num = intersection_num_per_part[0].shape[0]
raw_edge_num = indices_src.shape[0]
print("raw_part_num:", raw_part_num)
print("raw_edge_num:", raw_edge_num)
print("  reused_part_num (not -1 for every part), ratio:", reused_part_num, reused_part_num/raw_part_num)
print("  reused_part_reduction (-1 for every part), ratio:", reused_part_reduction, reused_part_reduction/raw_part_num)

print("    reused_edge_num (not -1 for every part), ratio:", reused_edge_num, reused_edge_num/raw_edge_num)
print("    reused_edge_reduction (-1 for every part), ratio:", reused_edge_reduction, reused_edge_reduction/raw_edge_num)

# import pdb;pdb.set_trace()
t_stamp,t_delta = t_click("-------------------- reuse_search_ext.py_greedy_max_reuse over: ",  t_stamp)
t_name_list.append("reuse_search_ext.py_greedy_max_reuse over")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)
##########

new_indices_src, _, \
    part_pointers_cache, column_index_cache, \
        part_cache_flagindex = reuse_search_ext.py_reorganize_indices(
    part_ptr_on_indices_src, indices_src,
    part_intersect_list, index_picked, indptr_dst, num_parts_per_node_acc,
    part_size)
# import pdb;pdb.set_trace()
t_stamp,t_delta = t_click("-------------------- reuse_search_ext.py_reorganize_indices over: ",  t_stamp)
t_name_list.append("reuse_search_ext.py_reorganize_indices over")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)

##########


uniqpart_column_index, uniqpart_pointers, part2Node_ptr, part2Node_col =  reuse_search_ext.py_reorganize_indices_uniquely(
    part_ptr_on_indices_src, indices_src,
    part_intersect_list, index_picked, indptr_dst, num_parts_per_node_acc,
    part_size)
# import pdb;pdb.set_trace()
t_stamp,t_delta = t_click("-------------------- reuse_search_ext.py_reorganize_indices_uniquely over: ",  t_stamp)
t_name_list.append("reuse_search_ext.py_reorganize_indices_uniquely over")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)
##########


features = features.cuda()
labels = labels.cuda()
edge_index = edge_index.cuda()
t_stamp,t_delta = t_click("-------------------- raw graph to gpu over: ",  t_stamp)
t_name_list.append("raw graph to gpu over")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)

degree_in = np.diff(indptr_dst)
# degree_out = np.diff(indptr_src)
warpsPerBlock = 4
# print("type and dtype of indptr_dst: ", type(indptr_dst), indptr_dst.dtype);
indptr_dst = torch.from_numpy(indptr_dst).cuda()
# t_stamp = t_click("-------------------- to gpu phase1 over: ",  t_stamp)
# print("type and dtype of new_indices_src: ", type(new_indices_src), new_indices_src.dtype);
new_indices_src = torch.from_numpy(new_indices_src.astype(np.int32)).cuda()
# print("type and dtype of degree_in: ", type(degree_in), degree_in.dtype);
degree_in = torch.from_numpy(degree_in.astype(np.float32)).cuda()
# print("type and dtype of part_ptr_on_indices_src: ", type(part_ptr_on_indices_src), part_ptr_on_indices_src.dtype);
part_ptr_on_indices_src = torch.from_numpy(part_ptr_on_indices_src.astype(np.int32)).cuda()
# print("type and dtype of part2Node: ", type(part2Node), part2Node.dtype);
part2Node = torch.from_numpy(part2Node.astype(np.int32)).cuda()
print("type and dtype of part_size: ", type(part_size))
# print("type and dtype of dimWorker: ", type(dimWorker), dimWorker.dtype)
print("type and dtype of warpsPerBlock: ", type(warpsPerBlock))
# print("type and dtype of part_result_cache: ", type(part_result_cache), part_result_cache.dtype)
# print("type and dtype of part_cache_flagindex: ", type(part_cache_flagindex), part_cache_flagindex.dtype);
part_cache_flagindex = torch.from_numpy(part_cache_flagindex.astype(np.int32)).cuda()
# print("type and dtype of part_pointers_cache: ", type(part_pointers_cache), part_pointers_cache.dtype);
part_pointers_cache = torch.from_numpy(part_pointers_cache.astype(np.int32)).cuda()
# print("type and dtype of column_index_cache: ", type(column_index_cache), column_index_cache.dtype);
column_index_cache = torch.from_numpy(column_index_cache.astype(np.int32)).cuda()
t_stamp,t_delta = t_click("-------------------- convert to tensor Phase1 over: ",  t_stamp)
t_name_list.append("convert to tensor Phase1 over")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)

# BUG:
# uniqpart_column_index, uniqpart_pointers, part2Node_ptr, part2Node_col
# print("type and dtype of uniqpart_column_index: ", type(uniqpart_column_index), uniqpart_column_index.dtype);
uniqpart_column_index = torch.from_numpy(uniqpart_column_index.astype(np.int32)).cuda()
# print("type and dtype of uniqpart_pointers: ", type(uniqpart_pointers), uniqpart_pointers.dtype);
uniqpart_pointers = torch.from_numpy(uniqpart_pointers.astype(np.int32)).cuda()
# print("type and dtype of part2Node_ptr: ", type(part2Node_ptr), part2Node_ptr.dtype);
part2Node_ptr = torch.from_numpy(part2Node_ptr.astype(np.int32)).cuda()
# print("type and dtype of part2Node_col: ", type(part2Node_col), part2Node_col.dtype);
part2Node_col = torch.from_numpy(part2Node_col.astype(np.int32)).cuda()
t_stamp,t_delta = t_click("-------------------- convert to tensor Phase2 over: ",  t_stamp)
t_name_list.append("convert to tensor Phase2 over")
t_delta_list.append(t_delta)
t_stamp_list.append(t_stamp)

inputInfo = ROF_inputInfo(indptr_dst, new_indices_src,
                          degree_in,
                          part_ptr_on_indices_src, part2Node, part_size,
                        #   dimWorker,
                          warpsPerBlock,
                        #   part_result_cache,
                          part_cache_flagindex,
                          part_pointers_cache, column_index_cache,
                          # BUG:
                            uniqpart_column_index, uniqpart_pointers, part2Node_ptr, part2Node_col
                          )
# import pdb;pdb.set_trace()
cpu_deivce = torch.device('cpu')

# if not os.path.exists("test.pt"):
#     inputInfo.to(cpu_deivce)
#     # print inputInfo in red color
#     print(colored(inputInfo, 'red'))
#     torch.save(inputInfo, "test.pt")
# else:
#     inputInfo = torch.load("test.pt")
#     inputInfo.to(cpu_deivce)
#     # print inputInfo in blue color
#     print(colored(inputInfo, 'blue'))

cuda0 = torch.device('cuda')
inputInfo.to(cuda0)

#########################################################
#########################################################

# arg_list = ["kernel_type", "dim_input", "dim_output", "datasetname", "part_size", "reorder_flag"]
arg_list = ["dim_input", "dim_output", "datasetname", "part_size", "reorder_flag"]
short_arg_list = ["di", "do", "d", "ps", "rf"]
kernel_args_list = ["warpsPerBlock"] # =4
print("ROF_inputInfo:", inputInfo)

dim_input = args.dim_input
dim_output = args.dim_output

output_txt = ""
output_txt += "{\n"
# add inputInfo.warpsPerBlock
output_txt += f"\twarpPerBlock:{inputInfo.warpPerBlock},\n"
for arg in vars(args):
    output_txt += f"\t{arg}:{getattr(args, arg)},\n"
output_txt += "}\n"


x = torch.randn(features.shape[0], dim_input).cuda()
weight = torch.randn(dim_input, dim_output).cuda()
# 
if dim_output > 32:
    dimWorker = 32
else:
    dimWorker = dim_output
inputInfo.dimWorker = dimWorker
# 
reuse_size=inputInfo.part_pointers_cache.shape[0] - 1 

# print inputInfo.part_pointers_cache.shape[0] in red color
print(colored(inputInfo.part_pointers_cache.shape[0], 'red'))

inputInfo.part_result_cache = torch.Tensor(reuse_size, dim_output+1).cuda()
# 
# "GNNA", "ROF_v1", "ROF_v2"
column_index, degrees, part_pointers, part2Node, partSize, dimWorker, warpPerBlock, part_result_cache, part_cache_flagindex, part_pointers_cache, column_index_cache =  inputInfo.column_index, inputInfo.degrees, inputInfo.part_pointers, inputInfo.part2Node, inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock, inputInfo.part_result_cache, inputInfo.part_cache_flagindex, inputInfo.part_pointers_cache, inputInfo.column_index_cache
uniqpart_column_index, uniqpart_pointers, part2Node_ptr, part2Node_col = inputInfo.uniqpart_column_index, inputInfo.uniqpart_pointers, inputInfo.part2Node_ptr, inputInfo.part2Node_col


kernel_types = ["GNNA", "ROF_v1", "ROF_v2"]
# kernel_types = ["GNNA", "ROF_v2", "ROF_v1"]
# kernel_types = ["ROF_v1", "ROF_v2", "GNNA"]
# kernel_types = ["ROF_v2", "ROF_v1", "GNNA"]
# kernel_types = ["ROF_v1", "GNNA", "ROF_v2"]
# kernel_types = ["ROF_v2", "GNNA", "ROF_v1"]


print("--------------------")
output_txt += "--------------------\n"
# # # # # # 
for kernel_type in kernel_types:
    if kernel_type == "GNNA":
        # Create events for recording the start and end of the kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        for i in range(10):
            a = ROF.GNNA_forward(x, weight, column_index,
                                degrees, part_pointers, part2Node,
                                partSize, dimWorker, warpPerBlock)[0]

        # Record the start event
        start_event.record()

        # Call the CUDA kernel 100 times
        elapsed_times = []
        for i in range(100):
            a = ROF.GNNA_forward(x, weight, column_index,
                                degrees, part_pointers, part2Node,
                                partSize, dimWorker, warpPerBlock)[0]

            # Record the end event
            end_event.record()

            # Wait for the events to complete
            torch.cuda.synchronize()

            # Compute the elapsed time
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)

            # Record the start event for the next iteration
            start_event.record()

# # # # # # 
    elif kernel_type == "ROF_v1":
        # Create events for recording the start and end of the kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        for i in range(10):
            a = ROF.reuse_forward_v1(x, weight, column_index,
                                            degrees, part_pointers, part2Node,
                                            partSize, dimWorker, warpPerBlock,
                                            part_result_cache, part_cache_flagindex,
                                            part_pointers_cache, column_index_cache
                                            )[0]

        # Record the start event
        start_event.record()

        # Call the CUDA kernel 100 times
        elapsed_times = []
        for i in range(100):
            a = ROF.reuse_forward_v1(x, weight, column_index,
                                            degrees, part_pointers, part2Node,
                                            partSize, dimWorker, warpPerBlock,
                                            part_result_cache, part_cache_flagindex,
                                            part_pointers_cache, column_index_cache
                                            )[0]

            # Record the end event
            end_event.record()

            # Wait for the events to complete
            torch.cuda.synchronize()

            # Compute the elapsed time
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)

            # Record the start event for the next iteration
            start_event.record()

# # # # # # 
    elif kernel_type == "ROF_v2":
        # Create events for recording the start and end of the kernel
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        for i in range(args.warmup):
            a = ROF.reuse_forward_v2(x, weight, uniqpart_column_index, 
                                        degrees, uniqpart_pointers, part2Node_ptr, part2Node_col,
                                        partSize, dimWorker, warpPerBlock
                                        )[0]

        # Record the start event
        start_event.record()

        # Call the CUDA kernel 100 times
        elapsed_times = []
        for i in range(args.runs):
            a = ROF.reuse_forward_v2(x, weight, uniqpart_column_index, 
                                        degrees, uniqpart_pointers, part2Node_ptr, part2Node_col,
                                        partSize, dimWorker, warpPerBlock
                                        )[0]

            # Record the end event
            end_event.record()

            # Wait for the events to complete
            torch.cuda.synchronize()

            # Compute the elapsed time
            elapsed_time_ms = start_event.elapsed_time(end_event)
            elapsed_times.append(elapsed_time_ms)

            # Record the start event for the next iteration
            start_event.record()

    # Compute the average and standard deviation of the last 100 runs
    avg_time = np.mean(elapsed_times)
    std_time = np.std(elapsed_times)

    print("***")
    print(f"Kernel type: {kernel_type}",end=";     ")
    print(f"Avg: {avg_time:.4g} ms",end="; ")
    print(f"std: {std_time:.4g} ms")
    print("***")

    output_txt += "***\n"
    if kernel_type == "GNNA":
        output_txt += f"Kernel type: {kernel_type};       "
    else:
        output_txt += f"Kernel type: {kernel_type};     "
    output_txt += f"Avg: {avg_time:.4g} ms; "
    output_txt += f"std: {std_time:.4g} ms\n"
    output_txt += "***\n"

print("--------------------")
output_txt += "--------------------\n"

result_path = args.datasetname + "_result.txt"
import os
if not os.path.exists(result_path):
    # create file as empty
    with open(result_path, "w") as f:
        f.write("")
        f.close()
with open(result_path, "a") as f:
    f.write(output_txt)
    f.write("\n")
    f.close()





exit()
#########################################################
#########################################################
arg_list = ["dim_input", "dim_output", "datasetname", "part_size", "reorder_flag"]
short_arg_list = ["di", "do", "d", "ps", "rf"]

'''
conda activate cuda111 ; add-cu110 ; add-gcc11;
env CUDA_VISIBLE_DEVICES=3 python small-main.py -d cora -ps 2 -rf 1 -di 1433 -do 16 -w 100 -r 1000
env CUDA_VISIBLE_DEVICES=3 python small-main.py -d cora -ps 2 -rf 1 -di 16 -do 16 -w 100 -r 1000
env CUDA_VISIBLE_DEVICES=3 python small-main.py -d cora -ps 2 -rf 1 -di 16 -do 7 -w 100 -r 1000

---
env CUDA_VISIBLE_DEVICES=3 python small-main.py -d cora -ps 3 -rf 1 -di 1433 -do 16 -w 100 -r 1000
env CUDA_VISIBLE_DEVICES=3 python small-main.py -d cora -ps 3 -rf 0 -di 1433 -do 16 -w 100 -r 1000
env CUDA_VISIBLE_DEVICES=3 python small-main.py -d cora -ps 4 -rf 1 -di 1433 -do 16 -w 100 -r 1000
env CUDA_VISIBLE_DEVICES=3 python small-main.py -d cora -ps 4 -rf 0 -di 1433 -do 16 -w 100 -r 1000


'''

n_layers = 1
activation = F.relu
dropout = 0.5
n_hidden = argparser.parse_args().n_hidden
weight_decay = 5e-4
lr = 1e-2
# kernel_type = "GNNA"
# kernel_type = "ROF_v1"
# kernel_type = "ROF_v2"
kernel_type = argparser.parse_args().kernel_type
n_classes = int(labels.max()) + 1
modelname = "rof_gcn"
rof_gcn = ROF_GCN(in_feats=features.shape[1],
                  n_hidden=n_hidden,
                  n_classes=n_classes,
                  n_layers=n_layers,
                  activation=activation,
                  dropout=dropout,
                  add_self_loops=False,
                  reuse_size=part_pointers_cache.shape[0] - 1,
                  kernel_type=kernel_type).cuda()
loss_fcn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(rof_gcn.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)


print(t_name_list)
print(t_stamp_list)
print(t_delta_list)

'''
conda activate cuda111 ; add-cu110 ; add-gcc11;
cd ROF_evaluation/4-2-reuse-ratio/;
env CUDA_VISIBLE_DEVICES=1 python small_g_search.py
'''