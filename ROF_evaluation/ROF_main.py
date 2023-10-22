import argparse
import time
def t_click(msg, t0):
    t1 = time.time()
    print(msg, t1-t0, f"({t1})" )
    return t1
t_stamp = time.time()
argparser = argparse.ArgumentParser()
argparser.add_argument('--kernel_type', '-kt', type=str, default='ROF_v1')
argparser.add_argument('--n_hidden', '-nh', type=int, default=16)

import sys
# get current directory of this file
cur_dir = sys.path[0]
sys.path.append(cur_dir + "/reuse_search_ext/src/")
sys.path.append(cur_dir + "/reuse_conv/")
sys.path.append("..")

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
t_stamp = t_click("-------------------- import over: ",  t_stamp)
#################### test load_graph ####################
##                                                     ##
datasetname = "cora"
# datasetname = "citeseer"
# datasetname = "pubmed"
features, edge_index, labels, train_mask, val_mask, test_mask = load_graph(
    data_root='../data',
#     rabbit_reorder_flag=True,
    rabbit_reorder_flag=False,
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

t_stamp = t_click("-------------------- load graph and convert csr over: ",  t_stamp)

# search resure structure in csr format
a = np.empty(0, dtype=np.int32)
b = np.empty(0, dtype=np.int32)
part_size = 3
part_ptr_on_indices_src, part2Node, num_parts_per_node_acc = reuse_search_ext.py_split_part(a, b, indptr_dst,
                                                     part_size)
# import pdb;pdb.set_trace()
t_stamp = t_click("-------------------- reuse_search_ext.py_split_part over: ",  t_stamp)


##########

discard_remainder_part = False
part_intersect_list, intersection_num_per_part = reuse_search_ext.py_get_part_intersect(
    part_ptr_on_indices_src, indices_src, indptr_src, indices_dst,
    part_size, discard_remainder_part)
t_stamp = t_click("-------------------- reuse_search_ext.py_get_part_intersect over: ",  t_stamp)

discard_remainder_part = True
# discard_remainder_part = False
part_intersect_list, intersection_num_per_part = reuse_search_ext.py_get_part_intersect(
    part_ptr_on_indices_src, indices_src, indptr_src, indices_dst,
    part_size, discard_remainder_part)
t_stamp = t_click("-------------------- reuse_search_ext.py_get_part_intersect over: ",  t_stamp)

check_num = []
for item in part_intersect_list:
    check_num.append(item.shape[0])
print("check_num vs intersection_num_per_part:", np.abs(np.array(check_num)-intersection_num_per_part[0]).sum())
print("intersection_num_per_part[0].sum() = ", intersection_num_per_part[0].sum())
print("(intersection_num_per_part[0]>2).sum() = ",(intersection_num_per_part[0]>2).sum(), intersection_num_per_part[0][intersection_num_per_part[0]>2].sum())
# import pdb;pdb.set_trace()

##########

index_ordered, index_picked = reuse_search_ext.py_greedy_max_reuse(
    intersection_num_per_part, part_ptr_on_indices_src, indices_src)
print("index_ordered.shape:", index_ordered.shape)
print("index_picked.shape:", index_picked.shape)
# import pdb;pdb.set_trace()
t_stamp = t_click("-------------------- reuse_search_ext.py_greedy_max_reuse over: ",  t_stamp)
##########

new_indices_src, _, \
    part_pointers_cache, column_index_cache, \
        part_cache_flagindex = reuse_search_ext.py_reorganize_indices(
    part_ptr_on_indices_src, indices_src,
    part_intersect_list, index_picked, indptr_dst, num_parts_per_node_acc,
    part_size)
# import pdb;pdb.set_trace()
t_stamp = t_click("-------------------- reuse_search_ext.py_reorganize_indices over: ",  t_stamp)
##########


uniqpart_column_index, uniqpart_pointers, part2Node_ptr, part2Node_col =  reuse_search_ext.py_reorganize_indices_uniquely(
    part_ptr_on_indices_src, indices_src,
    part_intersect_list, index_picked, indptr_dst, num_parts_per_node_acc,
    part_size)
# import pdb;pdb.set_trace()
t_stamp = t_click("-------------------- reuse_search_ext.py_reorganize_indices_uniquely over: ",  t_stamp)
##########


features = features.cuda()
labels = labels.cuda()
edge_index = edge_index.cuda()
t_stamp = t_click("-------------------- raw graph to gpu over: ",  t_stamp)

degree_in = np.diff(indptr_dst)
# degree_out = np.diff(indptr_src)
warpsPerBlock = 4
print("type and dtype of indptr_dst: ", type(indptr_dst), indptr_dst.dtype); indptr_dst = torch.from_numpy(indptr_dst).cuda()
# t_stamp = t_click("-------------------- to gpu phase1 over: ",  t_stamp)
print("type and dtype of new_indices_src: ", type(new_indices_src), new_indices_src.dtype); new_indices_src = torch.from_numpy(new_indices_src.astype(np.int32)).cuda()
print("type and dtype of degree_in: ", type(degree_in), degree_in.dtype); degree_in = torch.from_numpy(degree_in.astype(np.float32)).cuda()
print("type and dtype of part_ptr_on_indices_src: ", type(part_ptr_on_indices_src), part_ptr_on_indices_src.dtype); part_ptr_on_indices_src = torch.from_numpy(part_ptr_on_indices_src.astype(np.int32)).cuda()
print("type and dtype of part2Node: ", type(part2Node), part2Node.dtype); part2Node = torch.from_numpy(part2Node.astype(np.int32)).cuda()
print("type and dtype of part_size: ", type(part_size))
# print("type and dtype of dimWorker: ", type(dimWorker), dimWorker.dtype)
print("type and dtype of warpsPerBlock: ", type(warpsPerBlock))
# print("type and dtype of part_result_cache: ", type(part_result_cache), part_result_cache.dtype)
print("type and dtype of part_cache_flagindex: ", type(part_cache_flagindex), part_cache_flagindex.dtype); part_cache_flagindex = torch.from_numpy(part_cache_flagindex.astype(np.int32)).cuda()
print("type and dtype of part_pointers_cache: ", type(part_pointers_cache), part_pointers_cache.dtype); part_pointers_cache = torch.from_numpy(part_pointers_cache.astype(np.int32)).cuda()
print("type and dtype of column_index_cache: ", type(column_index_cache), column_index_cache.dtype); column_index_cache = torch.from_numpy(column_index_cache.astype(np.int32)).cuda()
t_stamp = t_click("-------------------- convert to tensor Phase1 over: ",  t_stamp)

# BUG:
# uniqpart_column_index, uniqpart_pointers, part2Node_ptr, part2Node_col
print("type and dtype of uniqpart_column_index: ", type(uniqpart_column_index), uniqpart_column_index.dtype); uniqpart_column_index = torch.from_numpy(uniqpart_column_index.astype(np.int32)).cuda()
print("type and dtype of uniqpart_pointers: ", type(uniqpart_pointers), uniqpart_pointers.dtype); uniqpart_pointers = torch.from_numpy(uniqpart_pointers.astype(np.int32)).cuda()
print("type and dtype of part2Node_ptr: ", type(part2Node_ptr), part2Node_ptr.dtype); part2Node_ptr = torch.from_numpy(part2Node_ptr.astype(np.int32)).cuda()
print("type and dtype of part2Node_col: ", type(part2Node_col), part2Node_col.dtype); part2Node_col = torch.from_numpy(part2Node_col.astype(np.int32)).cuda()
t_stamp = t_click("-------------------- convert to tensor Phase2 over: ",  t_stamp)

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

cuda0 = torch.device('cuda')
inputInfo.to(cuda0)

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



def pyg_evaluate(model, features, inputInfo, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, inputInfo)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

t_stamp = t_click("-------------------- prepare train over: ",  t_stamp)

dur = []
relative_seconds = []
accs = []
losses = []
end2endt0 = time.time()
epochs = 1000
for epoch in range(epochs):
    rof_gcn.train()
    t0 = time.time()
    optimizer.zero_grad()
    logits = rof_gcn(features, inputInfo)
    loss = loss_fcn(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    dur.append(time.time() - t0)

    acc = pyg_evaluate(rof_gcn, features, inputInfo, labels, val_mask)
    # print(
    #     "Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | ".
    #     format(epoch, dur[-1], loss.item(), acc))

    relative_seconds.append(time.time() - end2endt0)
    accs.append(acc)
    losses.append(loss.item())

t_stamp = t_click("-------------------- train over: ",  t_stamp)

end2endt = time.time() - end2endt0
test_acc = pyg_evaluate(rof_gcn, features, inputInfo, labels, test_mask)
print("Test accuracy {:.4g}".format(test_acc))
plot_dict = {
    "kernel_type": kernel_type,
    "device": cuda0,
    "dataset": datasetname,
    "model": modelname,
    "hidden": n_hidden,
    "layers": n_layers + 1,
    "activation": str(activation),
    "dropout": dropout,
    "lr": lr,
    "weight_decay": weight_decay,
    "epochs": epochs,
}
savefig = True
plot_training_time(dur,
                    warmup_epochs=3,
                    t=end2endt,
                    info_dict=plot_dict,
                    savefig=savefig,
                    test_acc=test_acc,
                    hint=kernel_type+str(n_hidden))

# save relative_seconds and accs as numpy array
np.savez(
    # f"../results/{datasetname}_{modelname}_{n_hidden}_{n_layers}_{str(activation)}_{dropout}_{lr}_{weight_decay}_{epochs}.npz",
    "../results/rof.npz",
    relative_seconds=relative_seconds,
    losses=losses,
    accs=accs,
    test_acc=test_acc)


# if __name__ == '__main__':
#     argparser = argparse.ArgumentParser()
#     argparser.add_argument('--data_root', '-dr', type=str, default='../data')
#     argparser.add_argument('--rabbit_reorder_flag',
#                            '-rrf',
#                            type=bool,
#                            default=True)  # True or False
#     argparser.add_argument('--graph_name', '-gn', type=str, default='cora')
#     argparser.add_argument('--device', '-d', type=str, default='cpu')

#     args = argparser.parse_args()
#     # print(args)

# conda activate cuda111 ; add-cu111 ; add-gcc11; cd ROF
# env CUDA_VISIBLE_DEVICES=1 python ROF_main.py