#include <torch/extension.h>
#include <vector>

////////////////////////////////////////////
//
// OKAY: CUDA backend kernel, Declarations
//
////////////////////////////////////////////

// HINT: SAG cuda backend kernel, Declaration
torch::Tensor SAG_cuda(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock);

// HINT: `GNNA forward` cuda backend kernel, Declaration
std::vector<torch::Tensor> spmm_forward_cuda_GNNA(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock
);

// HINT: `GNNA backward` cuda backend kernel, Declaration
std::vector<torch::Tensor> spmm_backward_cuda_GNNA(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor weight,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock
);

// HINT: `ROF v1 forward` cuda backend kernel, Declaration
std::vector<torch::Tensor> reuse_spmm_forward_cuda_v1(
    torch::Tensor input,
    torch::Tensor weight,
    // torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock,

    torch::Tensor part_result_cache,
    torch::Tensor part_cache_flagindex,
    torch::Tensor part_pointers_cache,
    torch::Tensor column_index_cache);

// HINT: `ROF v1 backward` cuda backend kernel, Declaration
std::vector<torch::Tensor> reuse_spmm_backward_cuda_v1(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor weight,
    // torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock,
    
    torch::Tensor cache_result,
    torch::Tensor part_cache_flag, // 
    torch::Tensor part_pointers_cache, // 
    torch::Tensor column_index_cache // same size as (picked_part x partsize)
);


// TODO: `ROF v2 forward` cuda backend kernel, Declaration
std::vector<torch::Tensor> reuse_spmm_forward_cuda_v2(
    torch::Tensor input,
    torch::Tensor weight,
    // torch::Tensor row_pointers, // no use
    torch::Tensor uniqpart_column_index,
    torch::Tensor degrees,
    torch::Tensor uniqpart_pointers,
    torch::Tensor part2Node_ptr,
    torch::Tensor part2Node_col,
    int partSize,
    int dimWorker,
    int warpPerBlock
);


// TODO: `ROF v2 backward` cuda backend kernel, Declaration
std::vector<torch::Tensor> reuse_spmm_backward_cuda_v2(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor weight,
    // torch::Tensor row_pointers, // no use
    torch::Tensor uniqpart_column_index,
    torch::Tensor degrees,
    torch::Tensor uniqpart_pointers,
    torch::Tensor part2Node_ptr,
    torch::Tensor part2Node_col,
    int partSize,
    int dimWorker,
    int warpPerBlock
);


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

////////////////////////////////////////////
//
// OKAY: python interfaces
//
////////////////////////////////////////////

// HINT: python interface, for SAG (only CUDA version now), Double wrap, Check tensor, CUDA/CPU dispatch
torch::Tensor SAG(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock)
{
    CHECK_INPUT(input);
    CHECK_INPUT(row_pointers);
    CHECK_INPUT(column_index);
    CHECK_INPUT(degrees);
    CHECK_INPUT(part_pointers);
    CHECK_INPUT(part2Node);

    return SAG_cuda(input, row_pointers, column_index,
                    degrees, part_pointers, part2Node,
                    partSize, dimWorker, warpPerBlock);
}



// HINT: python interface, for GNNA forward (only CUDA version now), Double wrap, check tensor, CUDA/CPU dispatch
std::vector<torch::Tensor> GNNA_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    CHECK_INPUT(column_index);
    CHECK_INPUT(degrees);
    CHECK_INPUT(part_pointers);
    CHECK_INPUT(part2Node);

    return spmm_forward_cuda_GNNA(input, weight, column_index,
                                  degrees, part_pointers, part2Node,
                                  partSize, dimWorker, warpPerBlock);
}

// HINT: python interface, for GNNA backward (only CUDA version now), Double wrap, check tensor, CUDA/CPU dispatch
std::vector<torch::Tensor> GNNA_backward(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor weight,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock)
{
    CHECK_INPUT(d_output);
    CHECK_INPUT(X);
    CHECK_INPUT(weight);
    CHECK_INPUT(column_index);
    CHECK_INPUT(degrees);
    CHECK_INPUT(part_pointers);
    CHECK_INPUT(part2Node);

    return spmm_backward_cuda_GNNA(d_output, X, weight, column_index,
                                   degrees, part_pointers, part2Node,
                                   partSize, dimWorker, warpPerBlock);
}

// HINT: python interface, for ROF v1 forward (only CUDA version now), Double wrap, check tensor, CUDA/CPU dispatch
std::vector<torch::Tensor> reuse_spmm_forward_v1(
    torch::Tensor input,
    torch::Tensor weight,
    // torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock,

    torch::Tensor part_result_cache,
    torch::Tensor part_cache_flagindex,
    torch::Tensor part_pointers_cache,
    torch::Tensor column_index_cache)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    // CHECK_INPUT(row_pointers);
    CHECK_INPUT(column_index);
    CHECK_INPUT(degrees);
    CHECK_INPUT(part_pointers);
    CHECK_INPUT(part2Node);
    CHECK_INPUT(part_result_cache);
    CHECK_INPUT(part_cache_flagindex);
    CHECK_INPUT(part_pointers_cache);
    CHECK_INPUT(column_index_cache);


    // TODO: CPU version
    return reuse_spmm_forward_cuda_v1(input, weight, column_index,
                                      degrees, part_pointers, part2Node,
                                      partSize, dimWorker, warpPerBlock,
                                      part_result_cache, part_cache_flagindex,
                                      part_pointers_cache, column_index_cache);
}

// HINT: python interface, for ROF v1 backward (only CUDA version now), Double wrap, check tensor, CUDA/CPU dispatch
std::vector<torch::Tensor> reuse_spmm_backward_v1(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    // torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize,
    int dimWorker,
    int warpPerBlock,

    torch::Tensor part_result_cache,
    torch::Tensor part_cache_flagindex,
    torch::Tensor part_pointers_cache,
    torch::Tensor column_index_cache)
{
    CHECK_INPUT(d_output);
    CHECK_INPUT(X);
    CHECK_INPUT(W);
    // CHECK_INPUT(row_pointers);
    CHECK_INPUT(column_index);
    CHECK_INPUT(degrees);
    CHECK_INPUT(part_pointers);
    CHECK_INPUT(part2Node);
    CHECK_INPUT(part_result_cache);
    CHECK_INPUT(part_cache_flagindex);
    CHECK_INPUT(part_pointers_cache);
    CHECK_INPUT(column_index_cache);

    // TODO: CPU version

    return reuse_spmm_backward_cuda_v1(d_output, X, W, column_index,
                                       degrees, part_pointers, part2Node,
                                       partSize, dimWorker, warpPerBlock,
                                       part_result_cache, part_cache_flagindex,
                                       part_pointers_cache, column_index_cache);
}

// TODO: python interface, for ROF v2 forward (only CUDA version now), Double wrap, check tensor, CUDA/CPU dispatch
std::vector<torch::Tensor> reuse_spmm_forward_v2(
    torch::Tensor input,
    torch::Tensor weight,
    // torch::Tensor row_pointers, // no use
    torch::Tensor uniqpart_column_index,
    torch::Tensor degrees,
    torch::Tensor uniqpart_pointers,
    torch::Tensor part2Node_ptr,
    torch::Tensor part2Node_col,
    int partSize,
    int dimWorker,
    int warpPerBlock)
{
    CHECK_INPUT(input);
    CHECK_INPUT(weight);
    // CHECK_INPUT(row_pointers);
    CHECK_INPUT(uniqpart_column_index);
    CHECK_INPUT(degrees);
    CHECK_INPUT(uniqpart_pointers);
    CHECK_INPUT(part2Node_ptr);
    CHECK_INPUT(part2Node_col);

    // TODO: CPU version
    return reuse_spmm_forward_cuda_v2(input, weight, uniqpart_column_index,
                                      degrees, uniqpart_pointers, part2Node_ptr,
                                      part2Node_col, partSize, dimWorker, warpPerBlock);
}

// TODO: python interface, for ROF v2 backward (only CUDA version now), Double wrap, check tensor, CUDA/CPU dispatch
std::vector<torch::Tensor> reuse_spmm_backward_v2(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor W,
    // torch::Tensor row_pointers, // no use
    torch::Tensor uniqpart_column_index,
    torch::Tensor degrees,
    torch::Tensor uniqpart_pointers,
    torch::Tensor part2Node_ptr,
    torch::Tensor part2Node_col,
    int partSize,
    int dimWorker,
    int warpPerBlock)
{
    CHECK_INPUT(d_output);
    CHECK_INPUT(X);
    CHECK_INPUT(W);
    // CHECK_INPUT(row_pointers);
    CHECK_INPUT(uniqpart_column_index);
    CHECK_INPUT(degrees);
    CHECK_INPUT(uniqpart_pointers);
    CHECK_INPUT(part2Node_ptr);
    CHECK_INPUT(part2Node_col);

    // TODO: CPU version

    return reuse_spmm_backward_cuda_v2(d_output, X, W, uniqpart_column_index,
                                       degrees, uniqpart_pointers, part2Node_ptr,
                                       part2Node_col, partSize, dimWorker, warpPerBlock);
}

////////////////////////////////////////////
//
// OKAY: export python interface
//
////////////////////////////////////////////
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("SAG", &SAG, "GNNAdvisor base Scatter-and-Gather Kernel (CUDA)");

    m.def("GNNA_forward", &GNNA_forward, "GNNAdvisor base forward (CUDA)");
    m.def("GNNA_backward", &GNNA_backward, "GNNAdvisor base backward (CUDA)");

    m.def("reuse_forward_v1", &reuse_spmm_forward_v1, "ROF gcn forward with reuse _v1(CUDA)");
    m.def("reuse_backward_v1", &reuse_spmm_backward_v1, "ROF gcn backward with reuse _v1(CUDA)");

    m.def("reuse_forward_v2", &reuse_spmm_forward_v2, "ROF gcn forward with reuse _v2(CUDA)");
    m.def("reuse_backward_v2", &reuse_spmm_backward_v2, "ROF gcn backward with reuse _v2(CUDA)");
}