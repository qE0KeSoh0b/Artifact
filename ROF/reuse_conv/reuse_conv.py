import torch
import ROF
import math


# std::vector<torch::Tensor> GNNA_forward(
#     torch::Tensor input,
#     torch::Tensor weight,
#     torch::Tensor column_index,
#     torch::Tensor degrees,
#     torch::Tensor part_pointers,
#     torch::Tensor part2Node,
#     int partSize,
#     int dimWorker,
#     int warpPerBlock)

# std::vector<torch::Tensor> GNNA_backward(
#     torch::Tensor d_output,
#     torch::Tensor X,
#     torch::Tensor weight,
#     torch::Tensor column_index,
#     torch::Tensor degrees,
#     torch::Tensor part_pointers,
#     torch::Tensor part2Node,
#     int partSize,
#     int dimWorker,
#     int warpPerBlock)

# HALF:
class ROF_GCNConvFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, inputInfo, add_self_loops=False, kernel_type="ROF_v1"):
        # "GNNA", "ROF_v1", "ROF_v2"
        column_index, degrees, part_pointers, part2Node, partSize, dimWorker, warpPerBlock, part_result_cache, part_cache_flagindex, part_pointers_cache, column_index_cache =  inputInfo.column_index, inputInfo.degrees, inputInfo.part_pointers, inputInfo.part2Node, inputInfo.partSize, inputInfo.dimWorker, inputInfo.warpPerBlock, inputInfo.part_result_cache, inputInfo.part_cache_flagindex, inputInfo.part_pointers_cache, inputInfo.column_index_cache
        # row_pointers = inputInfo.row_pointers

        uniqpart_column_index, uniqpart_pointers, part2Node_ptr, part2Node_col = inputInfo.uniqpart_column_index, inputInfo.uniqpart_pointers, inputInfo.part2Node_ptr, inputInfo.part2Node_col


        ctx.save_for_backward(x, weight, column_index, degrees, part_pointers, part2Node,
                            #   partSize, dimWorker, warpPerBlock,
                                part_result_cache, part_cache_flagindex, part_pointers_cache, column_index_cache,
                                uniqpart_column_index, uniqpart_pointers, part2Node_ptr, part2Node_col)

        ctx.partSize = partSize
        ctx.dimWorker = dimWorker
        ctx.warpPerBlock = warpPerBlock
        ctx.kernel_type = kernel_type

        if not add_self_loops:
            # TODO: add self loops option
            if kernel_type == "GNNA":
                return ROF.GNNA_forward(x, weight, column_index,
                                        degrees, part_pointers, part2Node,
                                        partSize, dimWorker, warpPerBlock
                                        )[0]
            elif kernel_type == "ROF_v1":
                return ROF.reuse_forward_v1(x, weight, column_index,
                                      degrees, part_pointers, part2Node,
                                      partSize, dimWorker, warpPerBlock,
                                      part_result_cache, part_cache_flagindex,
                                      part_pointers_cache, column_index_cache
                                      )[0]
            elif kernel_type == "ROF_v2":
                return ROF.reuse_forward_v2(x, weight, uniqpart_column_index, 
                                        degrees, uniqpart_pointers, part2Node_ptr, part2Node_col,
                                        partSize, dimWorker, warpPerBlock
                                        )[0]
            else:
                raise NotImplementedError
            # if here you give the list as output, then the graidient will be cut off, can not propagate.
        else:
            raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, column_index, degrees, part_pointers, part2Node, part_result_cache, part_cache_flagindex, part_pointers_cache, column_index_cache = ctx.saved_tensors[:-4]
        partSize, dimWorker, warpPerBlock = ctx.partSize, ctx.dimWorker, ctx.warpPerBlock
        kernel_type = ctx.kernel_type
        uniqpart_column_index, uniqpart_pointers, part2Node_ptr, part2Node_col = ctx.saved_tensors[-4:]
        if kernel_type == "GNNA":
            d_input, d_weight = ROF.GNNA_backward(grad_output, x, weight, column_index, degrees, part_pointers, part2Node, partSize, dimWorker, warpPerBlock)
        elif kernel_type == "ROF_v1":
            d_input, d_weight = ROF.reuse_backward_v1(grad_output, x, weight, column_index, degrees, part_pointers, part2Node, partSize, dimWorker, warpPerBlock, part_result_cache, part_cache_flagindex, part_pointers_cache, column_index_cache)
        elif kernel_type == "ROF_v2":
            d_input, d_weight = ROF.reuse_backward_v2(grad_output, x, weight, 
                                                      uniqpart_column_index, degrees, uniqpart_pointers, part2Node_ptr, part2Node_col,
                                                        partSize, dimWorker, warpPerBlock)
            
        else:
            raise NotImplementedError
        return d_input, d_weight, None, None, None


# OKAY:
class ROF_GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(ROF_GCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kwargs = kwargs
        self.add_self_loops = kwargs.get('add_self_loops', True)
        self.kernel_type = kwargs.get('kernel_type', 'ROF_v1')
        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reuse_size = kwargs.get('reuse_size', 0)
        if self.reuse_size <= 0:
            Warning("reuse_size should be larger than 0, otherwise CUDA error: invalid configuration argument")
        self.part_result_cache = torch.nn.Parameter(torch.Tensor(self.reuse_size, out_channels+1))

        self.dimWorker = kwargs.get('dimWorker', 0)
        if self.dimWorker == 0:
            if out_channels > 32:
                self.dimWorker = 32
            else:
                self.dimWorker = out_channels

        self.reset_parameters()

    def reset_parameters(self):
        weight_initializer = self.kwargs.get('weight_initializer', 'glorot')
        if weight_initializer == 'glorot':
            torch.nn.init.xavier_uniform_(self.weight)
        elif weight_initializer == 'uniform':
            bound = 1.0 / math.sqrt(self.in_channels)
            torch.nn.init.uniform_(self.weight.data, -bound, bound)
        elif weight_initializer == 'he':
            # same as pyg
            fan = self.in_channels
            a = math.sqrt(5)
            bound = math.sqrt(6 / ((1 + a**2) * fan))
            self.weight.data.uniform_(-bound, bound)
        else:
            raise NotImplementedError
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
        if self.reuse_size > 0:
            torch.nn.init.zeros_(self.part_result_cache)

    def forward(self, x, inputInfo):
        inputInfo.part_result_cache = self.part_result_cache
        inputInfo.dimWorker = self.dimWorker
        if self.bias is None:
            return ROF_GCNConvFunc.apply(x, self.weight, inputInfo, self.add_self_loops, self.kernel_type)
        else:
            return ROF_GCNConvFunc.apply(x, self.weight, inputInfo, self.add_self_loops, self.kernel_type)+self.bias
