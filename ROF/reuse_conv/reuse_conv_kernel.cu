#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define WARP_SIZE 32
#define input_type float
#define cache_flag_type int
// #define sqrtf(x) x
#define ready 1

__global__ void warmup(){}

__device__ inline 
void atomicAdd_F(float* address, float value)
{
  float old = value;  
  while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}


////////////////////////////////////////////
//
// OKAY: Basic Scatter-And-Gather kernel.
//
////////////////////////////////////////////
// HINT: Declaration of kernel
template <typename scalar_t>
__global__ void SAG_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
);
// HINT: Wrap kernel using torch::Tensor and dispatch
torch::Tensor SAG_cuda(
    torch::Tensor input,
    torch::Tensor row_pointers,
    torch::Tensor column_index,
    torch::Tensor degrees,
    torch::Tensor part_pointers,
    torch::Tensor part2Node,
    int partSize, 
    int dimWorker, 
    int warpPerBlock
){
    auto output = torch::zeros_like(input);

    const int num_nodes = output.size(0);
    const int dim = output.size(1);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (num_parts * WARP_SIZE + block  - 1) / block; 
    int shared_memory = partSize*warpPerBlock*sizeof(int)+warpPerBlock*dim*sizeof(float);

    // printf("grid: %d, block: %d, shared_memory: %d\n", grid, block, shared_memory);
    // printf("dim: %d, num_nodes: %d, num_parts: %d\n", dim, num_nodes, num_parts);
    // printf("dimWorker: %d\n", dimWorker);
	// #define PROFILE 200

	#ifdef PROFILE
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i=0; i<PROFILE; i++) {
        warmup<<<1,1>>>();
    }
	cudaEventRecord(start, 0);
    
    for (int i=0; i<PROFILE; i++) 
	#endif 
    AT_DISPATCH_FLOATING_TYPES(input.type(), "Scatter_and_Gather", ([&] {
                                SAG_cuda_kernel<scalar_t><<<grid, block, shared_memory>>>(
                                    output.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    input.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
                                    row_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    column_index.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    degrees.packed_accessor32<float,1,torch::RestrictPtrTraits>(),
                                    part_pointers.packed_accessor32<int,1,torch::RestrictPtrTraits>(), 
                                    part2Node.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
                                    num_nodes, 
                                    dim,
                                    num_parts,
                                    partSize,
                                    dimWorker,
                                    warpPerBlock
                                );
                            }));
    
                            
    #ifdef PROFILE
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gflop = 2*column_index.size(0)/1e6*dim;
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("TC-GNN -- Time (ms): %.3f, GFLOPs: %.3f\n", milliseconds/PROFILE, gflop/(milliseconds/PROFILE));
    printf("\n================================\n");
    #endif

    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }
    
    return output;
}
// HINT: Defniton of kernel
template <typename scalar_t>
__global__ void SAG_cuda_kernel(
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> row_pointers, 
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> part2Node,
    const int num_nodes, 
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
) {

    int tid =  blockIdx.x * blockDim.x + threadIdx.x;         // global thread-id
    int warpId = tid / WARP_SIZE;                             // global warp-id
    int block_warpId = threadIdx.x / WARP_SIZE;               // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;                     // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                      // part information.
    int *partial_ids = part_meta;                                           // caching ids
    float *partial_results = (float*)&part_meta[partSize*warpPerBlock];     // caching partial results.

    if (warpId < num_parts){

        int srcId = part2Node[warpId];              // aggregated source node
        int partBeg = part_pointers[warpId];        // partitioning pointer start
        int partEnd = part_pointers[warpId + 1];    // part pointer end

        // Cache the part neighbors.
        const int pindex_base = block_warpId * partSize;
        // #pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += dimWorker){
            // printf("1--pindex_base: %d, laneid: %d\n", pindex_base, laneid);
            partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
            // if(partial_ids[pindex_base + laneid]  >= num_nodes || partial_ids[pindex_base + laneid]  < 0) printf("---- partial_ids: %d\n", partial_ids[pindex_base + laneid] );
        }

         __syncwarp();

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            // if (laneid == 0) printf("2--pindex_base: %d, nIdx: %d\n", pindex_base, nIdx);
            int nid = partial_ids[pindex_base + nIdx];
            // if(nid >= num_nodes || nid < 0) printf("Error nid: %d\n", nid);

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
                #pragma unroll
                for (int d = laneid; d < dim; d += dimWorker){
                    partial_results[presult_base + d] = 0.0f;
                }
            
            if (laneid < dimWorker)
            #pragma unroll
            for (int d = laneid; d < dim; d += dimWorker){
                partial_results[presult_base + d] += input[nid][d];
            }
        }

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
        #pragma unroll
        for (int d = laneid; d < dim; d += dimWorker){
            atomicAdd_F((float*)&output[srcId][d], partial_results[presult_base + d]);
        }
    }
}


////////////////////////////////////////////
//
// OKAY: `Foward & Backward Pass (GCN) for GNNA` with reuse node update --> neighbor aggregation
//
////////////////////////////////////////////
// HINT: Defintion of `aggregation kernel` （of GNNA)
template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel_GNNA(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // `global` thread-id ; 
    int warpId = tid / WARP_SIZE;                    // `global` warp-id ; [0, num_parts-1] (in maybe >num_parts, its based on block)
    int block_warpId = threadIdx.x / WARP_SIZE;      // block warp-id ; [0, warpPerBlock-1]
    int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid ; [0,31]

    extern __shared__ int part_meta[];                                     // part information.
    int *partial_ids = part_meta;                                          // caching ids
    float *partial_results = (float *)&part_meta[partSize * warpPerBlock]; // caching partial results.

    if (warpId < num_parts) // warpId is global warpId, here discard the tail cases, real work is just within num_parts
    {

        int srcId = part2Node[warpId];   // aggregated source node
        float src_norm = degrees[srcId]; // norm of the source node

        int partBeg = part_pointers[warpId];     // partitioning pointer start
        int partEnd = part_pointers[warpId + 1]; // part pointer end

        // Cache the part neighbors by all threads from a warp.
        const int pindex_base = block_warpId * partSize;
#pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE)
        {
            // if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
            partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
        }

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];
            // float degree_norm = __fmul_rn(src_norm, degrees[nid]);
            float degree_norm_inv = __frsqrt_rn(degrees[nid]);

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
#pragma unroll
                    for (int d = laneid; d < dim; d += dimWorker)
                    {
                        partial_results[presult_base + d] = 0.0f;
                    }

            if (laneid < dimWorker)
#pragma unroll
                for (int d = laneid; d < dim; d += dimWorker)
                {
                    // if(nid >= num_nodes || nid < 0) printf("aggregation: %d\n", nid);
                    partial_results[presult_base + d] += __fmul_rn(degree_norm_inv, input[nid][d]);
                    // partial_results[presult_base + d] += input[nid][d];
                    // atomicAdd((float *)&output[srcId][d], input[nid][d]);
                }
        }

        src_norm = __frsqrt_rn(src_norm);
        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
#pragma unroll
            for (int d = laneid; d < dim; d += dimWorker)
            {
                atomicAdd_F((float *)&output[srcId][d], partial_results[presult_base + d] * src_norm);
                // atomicAdd_F((float *)&output[srcId][d], partial_results[presult_base + d]);
            }
    }
}


// HINT: Wrap `aggregation kernel` for `GNNA forward` , torch::Tensor as paras
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
)
{
    auto tmp = torch::mm(input, weight);
    auto output = torch::zeros({input.size(0), weight.size(1)}, input.device());

    const int dim = output.size(1);

    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (num_parts * WARP_SIZE + block - 1) / block;
    int shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);

    spmm_forward_cuda_kernel_GNNA<input_type><<<grid, block, shared_memory>>>(
        output.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        tmp.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        column_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        part_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        part2Node.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        dim,
        num_parts,
        partSize,
        dimWorker,
        warpPerBlock
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return {output};

}


// HINT: Wrap `aggregation kernel` for `GNNA backward` , torch::Tensor as paras
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
)
{
    auto d_input_prime = torch::zeros_like(d_output);
    const int dim = d_input_prime.size(1);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (num_parts * WARP_SIZE + block - 1) / block;
    int shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);

    spmm_forward_cuda_kernel_GNNA<input_type><<<grid, block, shared_memory>>>(
        d_input_prime.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        d_output.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        column_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        part_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        part2Node.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        dim,
        num_parts,
        partSize,
        dimWorker,
        warpPerBlock
    );

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    auto d_input = torch::mm(d_input_prime, weight.transpose(0,1));
    auto d_weight = torch::mm(X.transpose(0,1), d_input_prime);

    return {d_input, d_weight};
}



////////////////////////////////////////////
//
// OKAY: `Foward & Backward Pass (GCN) for ROF v1` with reuse node update --> neighbor aggregation
//
////////////////////////////////////////////
// HINT: Defintion of `caching kernel` （of ROF v1)
template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel_caching(
    // torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    // torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index_cache,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers_cache,
    // torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,
    // const int num_nodes,
    const int dim,
    // const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock,
    // torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> cache_table,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cache_result,
    const int cache_num // should be part_pointers_cache.size() - 1
)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
    int warpId = tid / WARP_SIZE;                    // global warp-id
    int block_warpId = threadIdx.x / WARP_SIZE;      // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                     // part information.
    int *partial_ids = part_meta;                                          // caching ids
    float *partial_results = (float *)&part_meta[partSize * warpPerBlock]; // caching partial results.


    if (warpId < cache_num) // warpId is global warpId, here discard the tail cases, real work is just within num_parts
    {

        // int srcId = part2Node[warpId];           // aggregated source node
        int partBeg = part_pointers_cache[warpId];     // partitioning pointer start
        int partEnd = part_pointers_cache[warpId + 1]; // part pointer end
        // float src_norm = degrees[srcId];         // norm of the source node
        // Cache the part neighbors by all threads from a warp.
        const int pindex_base = block_warpId * partSize;
#pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE)
        {
            // if ((pindex_base + nidx - partBeg)>=0 && (pindex_base + nidx - partBeg)<partSize * warpPerBlock && nidx<column_index_cache.size() && nidx>=0)
            partial_ids[pindex_base + nidx - partBeg] = column_index_cache[nidx];
            // column_index_cache[nidx] = 0;
            // column_index_cache[0] = 0;
            // partial_ids[pindex_base + nidx - partBeg] = 0;
            // partial_ids[1] = 0;
        }

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];
            // float degree_norm = __fmul_rn(src_norm, degrees[nid]);
            float degree_norm_inv = __frsqrt_rn(degrees[nid]);

            // Initialize shared memory for partial results
            if (nIdx == 0)
                if (laneid < dimWorker)
#pragma unroll
                    for (int d = laneid; d < dim; d += dimWorker)
                    {
                        partial_results[presult_base + d] = 0.0f;
                    }

            if (laneid < dimWorker)
#pragma unroll
                for (int d = laneid; d < dim; d += dimWorker)
                {
                    // if(nid >= num_nodes || nid < 0) printf("aggregation: %d\n", nid);
                    partial_results[presult_base + d] += __fmul_rn(degree_norm_inv, input[nid][d]);
                    // partial_results[presult_base + d] += input[nid][d];
                    // atomicAdd((float *)&output[srcId][d], input[nid][d]);
                }
        }
        // I need to cache the result in partial_results for latter reuse, reside in shared memory? or global memory?

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
#pragma unroll
            for (int d = laneid; d < dim; d += dimWorker)
            {
                // no need add, no need atomic, no one race for the same cache_result[warpId][d]
                cache_result[warpId][d] = partial_results[presult_base + d];
            }

        cache_result[warpId][dim] = ready;
    }
}
// HINT: Defintion of `reusing kernel` (of ROF v1)
template <typename scalar_t>
__global__ void spmm_forward_cuda_kernel_reusing(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    // torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> row_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,
    // const int num_nodes,
    const int dim,
    const int num_parts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock,
    // torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> cache_table,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cache_result,
    torch::PackedTensorAccessor32<cache_flag_type, 1, torch::RestrictPtrTraits> part_cache_flag
    // save the position(index from 1) of the cache of the part, otherwise 0
    // const int cache_num // should be part_pointers_cache.size() - 1
)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
    int warpId = tid / WARP_SIZE;                    // global warp-id
    int block_warpId = threadIdx.x / WARP_SIZE;      // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                     // part information.
    int *partial_ids = part_meta;                                          // caching ids
    float *partial_results = (float *)&part_meta[partSize * warpPerBlock]; // caching partial results.

    if (warpId < num_parts) // warpId is global warpId, here discard the tail cases, real work is just within num_parts
    {
        int reuse_flag = part_cache_flag[warpId];
        bool cache_ready = false;
        if (reuse_flag > 0)
        {
            // if the result is ready, reuse it
            cache_ready = cache_result[reuse_flag - 1][dim];
        }

        int srcId = part2Node[warpId];   // aggregated source node
        float src_norm = degrees[srcId]; // norm of the source node
        // if not ready, compute it
        if (!cache_ready)
        {

            int partBeg = part_pointers[warpId];     // partitioning pointer start
            int partEnd = part_pointers[warpId + 1]; // part pointer end

            // Cache the part neighbors by all threads from a warp.
            const int pindex_base = block_warpId * partSize;
#pragma unroll
            for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE)
            {
                // if(column_index[nidx] >= num_nodes || column_index[nidx] < 0) printf("column_index: %d\n", column_index[nidx]);
                partial_ids[pindex_base + nidx - partBeg] = column_index[nidx];
            }

            // Neighbor aggregation within each part
            const int presult_base = block_warpId * dim;
            for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
            {
                int nid = partial_ids[pindex_base + nIdx];
                // float degree_norm = __fmul_rn(src_norm, degrees[nid]);
                float degree_norm_inv = __frsqrt_rn(degrees[nid]);

                // Initialize shared memory for partial results
                if (nIdx == 0)
                    if (laneid < dimWorker)
#pragma unroll
                        for (int d = laneid; d < dim; d += dimWorker)
                        {
                            partial_results[presult_base + d] = 0.0f;
                        }

                if (laneid < dimWorker)
#pragma unroll
                    for (int d = laneid; d < dim; d += dimWorker)
                    {
                        // if(nid >= num_nodes || nid < 0) printf("aggregation: %d\n", nid);
                        partial_results[presult_base + d] += __fmul_rn(degree_norm_inv, input[nid][d]);
                        // partial_results[presult_base + d] += input[nid][d];
                        // atomicAdd((float *)&output[srcId][d], input[nid][d]);
                    }
            }

            src_norm = __frsqrt_rn(src_norm);
            // output the result to global memory from the shared memory
            if (laneid < dimWorker)
#pragma unroll
                for (int d = laneid; d < dim; d += dimWorker)
                {
                    atomicAdd_F((float *)&output[srcId][d], partial_results[presult_base + d] * src_norm);
                    // atomicAdd_F((float *)&output[srcId][d], partial_results[presult_base + d]);
                }
        }
        else
        { 
        //OPTION: resusing cache
            // if the result is ready, reuse it
            src_norm = __frsqrt_rn(src_norm);
            if (laneid < dimWorker)
#pragma unroll
                for (int d = laneid; d < dim; d += dimWorker)
                {// OPTION: use constant value as cache directly
                    atomicAdd_F((float *)&output[srcId][d], cache_result[reuse_flag - 1][d] * src_norm);
                    // atomicAdd_F((float *)&output[srcId][d], cache_result[reuse_flag - 1][d]);
                }  
        }
    }
}
// HINT: Wrap `caching&reusing kernels` for `ROF v1 forward` , torch::Tensor as paras
std::vector<torch::Tensor> reuse_spmm_forward_cuda_v1(
    torch::Tensor input,
    torch::Tensor weight,
    // torch::Tensor row_pointers, // no use
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
)
{
    auto tmp = torch::mm(input, weight);
    // return {tmp};
    // auto output = torch::zeros_like(tmp);
    auto output = torch::zeros({input.size(0), weight.size(1)}, input.device());
/*
    std::cout << "tmp.device(): " << tmp.device() << std::endl;
    std::cout << "output.device(): " << output.device() << std::endl;
    std::cout << "input.device(): " << input.device() << std::endl;
    std::cout << "weight.device(): " << weight.device() << std::endl;
    std::cout << "part_pointers.device(): " << part_pointers.device() << std::endl;
    std::cout << "part2Node.device(): " << part2Node.device() << std::endl;
    std::cout << "part_cache_flag.device(): " << part_cache_flag.device() << std::endl;
*/
    // return {output};
    const int dim = output.size(1);
    const int num_nodes = output.size(0);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (num_parts * WARP_SIZE + block - 1) / block;
    int shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);
/*
    std::cout << "warpPerBlock: " << warpPerBlock << std::endl;
    std::cout << "WARP_SIZE: " << WARP_SIZE << std::endl;
    std::cout << "block: " << block << std::endl;
    std::cout << "num_parts: " << num_parts << std::endl;
    std::cout << "grid: " << grid << std::endl;
    std::cout << "shared_memory: " << shared_memory << std::endl;
*/
    const int cache_num = cache_result.size(0);
    const int cache_block = warpPerBlock * WARP_SIZE;
    const int cache_grid = (cache_num * WARP_SIZE + cache_block - 1) / cache_block;
    int cache_shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);
/*
    std::cout << "cache_num: " << cache_num << std::endl;
    std::cout << "cache_block: " << cache_block << std::endl;
    std::cout << "cache_grid: " << cache_grid << std::endl;
    std::cout << "cache_shared_memory: " << cache_shared_memory << std::endl;
*/
    // printf("num_parts: %d, ", num_parts);
    // std::std::cout << "part_cache_flag.shape: " << part_cache_flag.sizes() << std::std::endl;


    //OPTION: caching kernel
    spmm_forward_cuda_kernel_caching<input_type><<<cache_grid, cache_block, cache_shared_memory>>>(
        // output.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        tmp.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        // row_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        column_index_cache.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        part_pointers_cache.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        // part2Node.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        // num_nodes,
        dim,
        // num_parts,
        partSize,
        dimWorker,
        warpPerBlock,
        cache_result.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        cache_num);
/*
    cudaError_t error1 = cudaGetLastError();
    if (error1 != cudaSuccess)
    {
        printf("CUDA error1: %s\n", cudaGetErrorString(error1));
        exit(-1);
    }
*/
    cudaDeviceSynchronize();


    spmm_forward_cuda_kernel_reusing<input_type><<<grid, block, shared_memory>>>(
        output.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        tmp.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        // row_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        column_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        part_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        part2Node.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        // num_nodes,
        dim,
        num_parts,
        partSize,
        dimWorker,
        warpPerBlock,
        cache_result.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        part_cache_flag.packed_accessor32<cache_flag_type, 1, torch::RestrictPtrTraits>());

    cudaError_t error2 = cudaGetLastError();
    if (error2 != cudaSuccess)
    {
        printf("CUDA error2: %s\n", cudaGetErrorString(error2));
        exit(-1);
    }


    return {output};
    // return {cache_result};
    // return {tmp};
}
// HINT: Wrap `caching&reusing kernels` for `ROF v1 backward` , torch::Tensor as paras
std::vector<torch::Tensor> reuse_spmm_backward_cuda_v1(
    torch::Tensor d_output,
    torch::Tensor X,
    torch::Tensor weight,
    // torch::Tensor row_pointers, // no use
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
){


    // ****** propagate ******
    auto d_input_prime = torch::zeros_like(d_output);
    const int dim = d_input_prime.size(1);
    // const int num_nodes = d_input_prime.size(0);
    const int num_parts = part2Node.size(0);

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (num_parts * WARP_SIZE + block - 1) / block;
    int shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);
/*
    std::cout << "warpPerBlock: " << warpPerBlock << std::endl;
    std::cout << "WARP_SIZE: " << WARP_SIZE << std::endl;
    std::cout << "block: " << block << std::endl;
    std::cout << "num_parts: " << num_parts << std::endl;
    std::cout << "grid: " << grid << std::endl;
    std::cout << "shared_memory: " << shared_memory << std::endl;
*/
    const int cache_num = cache_result.size(0);
    const int cache_block = warpPerBlock * WARP_SIZE;
    const int cache_grid = (cache_num * WARP_SIZE + cache_block - 1) / cache_block;
    int cache_shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);
/*
    std::cout << "cache_num: " << cache_num << std::endl;
    std::cout << "cache_block: " << cache_block << std::endl;
    std::cout << "cache_grid: " << cache_grid << std::endl;
    std::cout << "cache_shared_memory: " << cache_shared_memory << std::endl;
*/
    // printf("num_parts: %d, ", num_parts);
    // std::std::cout << "part_cache_flag.shape: " << part_cache_flag.sizes() << std::std::endl;


    //OPTION: caching kernel
    spmm_forward_cuda_kernel_caching<input_type><<<cache_grid, cache_block, cache_shared_memory>>>(
        // output.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        d_output.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        // row_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        column_index_cache.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        part_pointers_cache.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        // part2Node.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        // num_nodes,
        dim,
        // num_parts,
        partSize,
        dimWorker,
        warpPerBlock,
        cache_result.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        cache_num);

    cudaDeviceSynchronize();

    spmm_forward_cuda_kernel_reusing<input_type><<<grid, block, shared_memory>>>(
        d_input_prime.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        d_output.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        // row_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        column_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        part_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        part2Node.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        // num_nodes,
        dim,
        num_parts,
        partSize,
        dimWorker,
        warpPerBlock,
        cache_result.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        part_cache_flag.packed_accessor32<cache_flag_type, 1, torch::RestrictPtrTraits>());

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    // ****** propagate ******

    auto d_input = torch::mm(d_input_prime, weight.transpose(0,1));
    auto d_weight = torch::mm(X.transpose(0,1), d_input_prime);

    return {d_input, d_weight};
}






////////////////////////////////////////////
//
// TODO: `Foward & Backward Pass (GCN) for ROF v2` with reuse node update --> neighbor aggregation
//
////////////////////////////////////////////
// HINT: Defintion of `fused kernel` （of ROF v2)
template <typename scalar_t>
// __global__ void spmm_forward_cuda_kernel_fuse(
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
//     torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> column_index,
//     torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part_pointers,

//     torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node,

//     const int dim,
//     const int num_parts,
//     const int partSize,
//     const int dimWorker,
//     const int warpPerBlock
// )
__global__ void spmm_forward_cuda_kernel_fuse(
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> input,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> uniqpart_column_index,
    torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> degrees,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> uniqpart_pointers,

    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node_ptr,
    torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> part2Node_col,

    const int dim,
    const int num_uniqparts,
    const int partSize,
    const int dimWorker,
    const int warpPerBlock
)
{

    int tid = blockIdx.x * blockDim.x + threadIdx.x; // global thread-id
    int warpId = tid / WARP_SIZE;                    // global warp-id
    int block_warpId = threadIdx.x / WARP_SIZE;      // block warp-id
    int laneid = threadIdx.x % WARP_SIZE;            // warp thread-id -- laneid

    extern __shared__ int part_meta[];                                     // part information.
    int *partial_ids = part_meta;                                          // caching ids
    float *partial_results = (float *)&part_meta[partSize * warpPerBlock]; // caching partial results.

    if (warpId < num_uniqparts) // warpId is global warpId, here discard the tail cases, real work is just within num_uniqparts
    {

        int part2Node_beg = part2Node_ptr[warpId];     // partitioning pointer start
        int part2Node_end = part2Node_ptr[warpId + 1]; // part pointer end
        int dst_num = part2Node_end - part2Node_beg;

        int partBeg = uniqpart_pointers[warpId];     // partitioning pointer start
        int partEnd = uniqpart_pointers[warpId + 1]; // part pointer end

        // Cache the part neighbors by all threads from a warp.
        const int pindex_base = block_warpId * partSize;
#pragma unroll
        for (int nidx = partBeg + laneid; nidx < partEnd; nidx += WARP_SIZE)
        {
            // if(uniqpart_column_index[nidx] >= num_nodes || uniqpart_column_index[nidx] < 0) printf("uniqpart_column_index: %d\n", uniqpart_column_index[nidx]);
            partial_ids[pindex_base + nidx - partBeg] = uniqpart_column_index[nidx];
        }

        // __syncwarp();

        // Neighbor aggregation within each part
        const int presult_base = block_warpId * dim;
        if (laneid < dimWorker)
        {
#pragma unroll
            for (int d = laneid; d < dim; d += dimWorker)
            {
                partial_results[presult_base + d] = 0.0f;
            }
        }
        for (int nIdx = 0; nIdx < partEnd - partBeg; nIdx++)
        {
            int nid = partial_ids[pindex_base + nIdx];
            // float degree_norm = __fmul_rn(src_norm, degrees[nid]);
            float degree_norm_inv = __frsqrt_rn(degrees[nid]);

            // IF NO 0-LEN PART, SHARE MEMORY CAN BE INITIALIZED INSIDE THE LOOP
            // Initialize shared memory for partial results
//             if (nIdx == 0)
//                 if (laneid < dimWorker)
// #pragma unroll
//                     for (int d = laneid; d < dim; d += dimWorker)
//                     {
//                         partial_results[presult_base + d] = 0.0f;
//                     }

            if (laneid < dimWorker)
#pragma unroll
                for (int d = laneid; d < dim; d += dimWorker)
                {
                    // if(nid >= num_nodes || nid < 0) printf("aggregation: %d\n", nid);
                    partial_results[presult_base + d] += __fmul_rn(degree_norm_inv, input[nid][d]);
                    // partial_results[presult_base + d] += input[nid][d];
                    // atomicAdd((float *)&output[srcId][d], input[nid][d]);
                }
        }

        // output the result to global memory from the shared memory
        if (laneid < dimWorker)
// #pragma unroll
            for (int i_dst = part2Node_beg; i_dst < part2Node_end; i_dst++)
            {
                int srcId = part2Node_col[i_dst];
                float src_norm = degrees[srcId];
                src_norm = __frsqrt_rn(src_norm);

#pragma unroll
                for (int d = laneid; d < dim; d += dimWorker)
                {
                    atomicAdd_F((float *)&output[srcId][d], partial_results[presult_base + d] * src_norm);
                    // atomicAdd_F((float *)&output[srcId][d], partial_results[presult_base + d]);
                }
            }
    }
}

// HINT: Wrap `fused kernels` for `ROF v2 forward` , torch::Tensor as paras
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
)
{
    auto tmp = torch::mm(input, weight);
    auto output = torch::zeros({input.size(0), weight.size(1)}, input.device());
    const int dim = output.size(1);
    const int num_uniqparts = uniqpart_pointers.size(0)-1;

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (num_uniqparts * WARP_SIZE + block - 1) / block;
    int shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);

    spmm_forward_cuda_kernel_fuse<input_type><<<grid, block, shared_memory>>>(
        output.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        tmp.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        uniqpart_column_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        uniqpart_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        part2Node_ptr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        part2Node_col.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        dim,
        num_uniqparts,
        partSize,
        dimWorker,
        warpPerBlock);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    return {output};

}

// HINT: Wrap `fused kernels` for `ROF v2 backward` , torch::Tensor as paras
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
)
{
    auto d_input_prime = torch::zeros_like(d_output);
    const int dim = d_input_prime.size(1);
    const int num_uniqparts = uniqpart_pointers.size(0)-1;

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (num_uniqparts * WARP_SIZE + block - 1) / block;
    int shared_memory = partSize * warpPerBlock * sizeof(int) + warpPerBlock * dim * sizeof(float);

    spmm_forward_cuda_kernel_fuse<input_type><<<grid, block, shared_memory>>>(
        d_input_prime.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        d_output.packed_accessor32<input_type, 2, torch::RestrictPtrTraits>(),
        uniqpart_column_index.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        degrees.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        uniqpart_pointers.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        part2Node_ptr.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        part2Node_col.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        dim,
        num_uniqparts,
        partSize,
        dimWorker,
        warpPerBlock);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        exit(-1);
    }

    auto d_input = torch::mm(d_input_prime, weight.transpose(0,1));
    auto d_weight = torch::mm(X.transpose(0,1), d_input_prime);

    return {d_input, d_weight};
}




////////////////////////////////////////////
//
// TODO
//
////////////////////////////////////////////