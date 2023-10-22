REPO = 
SERVER = 

import argparse
import os
import time
def t_click(msg, t0):
    t1 = time.time()
    print(msg, t1-t0, f"({t1})" )
    return t1, t1-t0
t_stamp = time.time()
argparser = argparse.ArgumentParser()
# argparser.add_argument('--save_reorder', '-sr', type=int, default=0)
# argparser.add_argument('--load_reorder', '-lr', type=int, default=1)
argparser.add_argument('--kernel_type', '-kt', type=str, default='ROF_v2')
argparser.add_argument('--n_hidden', '-nh', type=int, default=128)
# argparser.add_argument('--dim_input', '-di', type=int, default=1433)
# argparser.add_argument('--dim_output', '-do', type=int, default=7)
argparser.add_argument('--datasetname', '-d', type=str, default='ogbn-arxiv')
argparser.add_argument('--part_size', '-ps', type=int, default=5)
argparser.add_argument('--reorder_flag', '-rf', type=int, default=0)
argparser.add_argument('--log_to_file', '-lf', type=int, default=0)
argparser.add_argument('--k_hops', '-kh', type=int, default=-1)
# warmup, runs
argparser.add_argument('--warmup', '-w', type=int, default=0)
argparser.add_argument('--runs', '-r', type=int, default=100)
args = argparser.parse_args()
'''
conda activate cuda111 ; add-cu111 ; add-gcc11; cd ROF_evaluation/4-2-reuse-ratio/;
env CUDA_VISIBLE_DEVICES=1 python small_g_search.py -d cora -ps 2 -rf 1
env CUDA_VISIBLE_DEVICES=1 python middle_g_search.py -d ogbn-arxiv -ps 2 -rf 0

env CUDA_VISIBLE_DEVICES=2 python middle-rof.py -kt ROF_v2 -nh 128 -d ogbn-arxiv -ps 5 -rf 0 

env CUDA_VISIBLE_DEVICES=2 python middle-rof.py -kt ROF_v2 -nh 128 -d ogbn-products -ps 5 -rf 0
'''

log_file_name = "curvefile"
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
    if args.log_to_file>0:
        sys.stdout = open(path, 'w')
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

from data_prepare import load_graph, induced_k_hop_subgraph
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
# datasetname = "ogbn-arxiv"
# datasetname = "ogbn-products"

if datasetname == "ogbn-products":
    rofinput_arg_list = ["part_size", "reorder_flag", "k_hops"]
    rofinput_id = "rofinput"
    for arg in rofinput_arg_list:
        rofinput_id += f"-{arg}-{getattr(args, arg)}"
    rofinput_id += ".pt"

    hop_load_graph_arg_list = ["reorder_flag", "k_hops"]
    hop_load_graph_id = "hop_load_graph"
    for arg in hop_load_graph_arg_list:
        hop_load_graph_id += f"-{arg}-{getattr(args, arg)}"
    hop_load_graph_id += ".pt"

    load_graph_arg_list = ["reorder_flag"]
    load_graph_id = "load_graph"
    for arg in load_graph_arg_list:
        load_graph_id += f"-{arg}-{getattr(args, arg)}"
    load_graph_id += ".pt"


if not os.path.exists(rofinput_id):

    if not os.path.exists(hop_load_graph_id):

        if not os.path.exists(load_graph_id):

            features, edge_index, labels, train_idx, valid_idx, test_idx = load_graph(
                data_root= data_root,
                rabbit_reorder_flag=args.reorder_flag,
                graph_name=datasetname,
                device='cpu',
                other_require=None)
            
            saved_load_graph = {
                "features": features,
                "edge_index": edge_index,
                "labels": labels,
                "train_idx": train_idx,
                "valid_idx": valid_idx,
                "test_idx": test_idx
            }
            torch.save(saved_load_graph, load_graph_id)
        else:
            saved_load_graph = torch.load(load_graph_id)
            features, edge_index, labels, train_idx, valid_idx, test_idx = \
                saved_load_graph["features"], saved_load_graph["edge_index"], saved_load_graph["labels"], \
                    saved_load_graph["train_idx"], saved_load_graph["valid_idx"], saved_load_graph["test_idx"]

        num_nodes = features.shape[0]
        full_num_edges = edge_index.shape[1]

        # print the ratio of train/valid/test
        print("dtype of train_idx:", train_idx.dtype)
        print("train/valid/test ratio:", train_idx.shape[0]/num_nodes, valid_idx.shape[0]/num_nodes, test_idx.shape[0]/num_nodes)
        print("args.k_hops:", args.k_hops)
        if args.k_hops > 0:
            edge_index = induced_k_hop_subgraph(edge_index, train_idx, args.k_hops)
    
        saved_hop_load_graph = {
            "features": features,
            "edge_index": edge_index,
            "labels": labels,
            "train_idx": train_idx,
            "valid_idx": valid_idx,
            "test_idx": test_idx,
            "num_nodes": num_nodes,
            "full_num_edges": full_num_edges
        }
        torch.save(saved_hop_load_graph, hop_load_graph_id)
    else:
        saved_hop_load_graph = torch.load(hop_load_graph_id)
        features, edge_index, labels, train_idx, valid_idx, test_idx, num_nodes, full_num_edges = \
            saved_hop_load_graph["features"], saved_hop_load_graph["edge_index"], saved_hop_load_graph["labels"], \
                saved_hop_load_graph["train_idx"], saved_hop_load_graph["valid_idx"], saved_hop_load_graph["test_idx"], \
                    saved_hop_load_graph["num_nodes"], saved_hop_load_graph["full_num_edges"]


    num_edges = edge_index.shape[1]
    print("num_edges/full_num_edges:", num_edges, full_num_edges, num_edges/full_num_edges)

    #########################################################

    # after optional reorder, convert to coo, and then convert to csr
    val = [1] * num_edges
    scipy_coo1 = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
    scipy_csr1 = scipy_coo1.tocsr()

    scipy_coo2 = coo_matrix((val, [edge_index[1],edge_index[0]]), shape=(num_nodes, num_nodes))
    scipy_csr2 = scipy_coo2.tocsr()

    # 1. in edge_index, upper is src, lower is dst; `coo` is the same
    # 2. convert `coo` to `csr`, then indices is dst; i.e., indices mean out edge of node 0,1,2...
    # 3. convert `csr` to array, then the row is src, col is dst.
    indptr_src = scipy_csr1.indptr
    indices_dst = scipy_csr1.indices

    # because it is symmetric, otherwise we need inverse
    indptr_dst = scipy_csr2.indptr
    indices_src = scipy_csr2.indices
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


    t_stamp,t_delta = t_click("-------------------- reuse_search_ext.py_greedy_max_reuse over: ",  t_stamp)
    t_name_list.append("reuse_search_ext.py_greedy_max_reuse over")
    t_delta_list.append(t_delta)
    t_stamp_list.append(t_stamp)
    ##########
    # import pdb;pdb.set_trace()

    # column_index_cache, part_pointers_cache, part_cache_flagindex = 
    def check_py_reorganize_indices(part_ptr_on_indices_src, indices_src,
                        intersection_per_part, index_picked, indptr_dst,
                        #    column_index_cache, part_pointers_cache, part_cache_flagindex,
                        num_parts_per_node_acc, part_size):
        # import pdb;pdb.set_trace()
        cur_reorg_pos = indptr_dst.copy()
        indices_src_copy = indices_src.copy()
        
        column_index_cache = []
        part_pointers_cache = [0]
        part_cache_flagindex = [0] * (len(part_ptr_on_indices_src) - 1)
        num_parts_per_node_acc[-1] = 0
        
        for i in range(index_picked.shape[0]):
            part_idx = index_picked[i]
            part_start = part_ptr_on_indices_src[part_idx]
            part_end = part_ptr_on_indices_src[part_idx + 1]
            
            if part_end - part_start != part_size:
                print("part size is not equal to part_size")
            
            intersection_of_part = intersection_per_part[part_idx]
            
            for j in range(len(intersection_of_part)):
                dst_node_idx = intersection_of_part[j]
                
                reorg_start = cur_reorg_pos[dst_node_idx]
                neighbor_end = indptr_dst[dst_node_idx + 1]
                
                for k in range(part_start, part_end):
                    pos = np.where(indices_src_copy[reorg_start:neighbor_end]==indices_src_copy[k])
                    
                    if pos != -1:
                        indices_src_copy[pos + reorg_start], indices_src_copy[reorg_start] = indices_src_copy[reorg_start], indices_src_copy[pos + reorg_start]
                        reorg_start += 1
                    else:
                        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                        print("cur_reorg_pos[dst_node_idx]:", cur_reorg_pos[dst_node_idx])
                        print("indptr_dst[dst_node_idx]:", indptr_dst[dst_node_idx])
                        print("indptr_dst[dst_node_idx + 1]:", indptr_dst[dst_node_idx + 1])
                        print("impossible error: 'pos == -1'")
                        print("-----------------------------------")
                
                if reorg_start - cur_reorg_pos[dst_node_idx] != part_size:
                    print("impossible error: 'reorg_start - cur_reorg_pos[dst_node_idx] != part_size'")
                
                cur_reorg_pos[dst_node_idx] = reorg_start
                
                num_nodes = len(num_parts_per_node_acc)
                part_cache_flagindex[(dst_node_idx - 1 + num_nodes) % num_nodes] = i + 1
                num_parts_per_node_acc[(dst_node_idx - 1 + num_nodes) % num_nodes] += 1
            
            column_index_cache.extend(indices_src_copy[part_start:part_end])
            part_pointers_cache.append(len(column_index_cache))


    # new_indices_src, _, \
    #     part_pointers_cache, column_index_cache, \
    #         part_cache_flagindex = check_py_reorganize_indices(
    #     part_ptr_on_indices_src, indices_src,
    #     part_intersect_list, index_picked, indptr_dst, num_parts_per_node_acc,
    #     part_size)


    # import pdb;pdb.set_trace()


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

    # save inputInfo to rofinput_id for further fast runnning
    cpu_deivce = torch.device('cpu')
    inputInfo.to(cpu_deivce)
    torch.save(inputInfo, rofinput_id)

    cuda0 = torch.device('cuda')
    inputInfo.to(cuda0)

else:
    print("load inputInfo from rofinput_id:", rofinput_id)
    inputInfo = torch.load(rofinput_id)
    cuda0 = torch.device('cuda')
    inputInfo.to(cuda0)

# # ogbn-arxiv 128 40
# # # ogbn-products 100 47


n_layers = 3
activation = F.relu
dropout = 0.5
n_hidden = args.n_hidden
weight_decay = 5e-4
lr = 1e-2
# kernel_type = "GNNA"
# kernel_type = "ROF_v1"
# kernel_type = "ROF_v2"
kernel_type = args.kernel_type
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

loss_list_epoch = []
elapsed_time_list_epoch = []

t0 = time.time()
for epoch in range(args.warmup + args.runs):
    t1 = time.time()
    rof_gcn.train()
    optimizer.zero_grad()
    out = rof_gcn(features, edge_index, inputInfo)
    loss = loss_fcn(out[train_idx], labels[train_idx])
    loss.backward()
    optimizer.step()

    if epoch >= args.warmup:
        loss_list_epoch.append(loss.item())
        elapsed_time_list_epoch.append(time.time() - t1)
        print(f"epoch: {epoch}, loss: {loss.item()}")

# save loss_list_epoch and elapsed_time_list_epoch
loss_list_epoch = np.array(loss_list_epoch)
elapsed_time_list_epoch = np.array(elapsed_time_list_epoch)
curve_dict = {
    "loss": loss_list_epoch,
    "elapsed_time": elapsed_time_list_epoch
}
curve_dict_path = f"curve_dict-{log_file_name}.pt"

'''
conda activate cuda111 ; add-cu111 ; add-gcc11;
# no cu111 in lccpu28, change to cu110
conda activate cuda111 ; add-cu110 ; add-gcc11;

env CUDA_VISIBLE_DEVICES=1 python small_g_search.py
'''