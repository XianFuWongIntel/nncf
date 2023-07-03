#ifndef _COMMON_XPU_FUNCS_DP_HPP_
#define _COMMON_XPU_FUNCS_DP_HPP_

// Have to define common CUDA __device__ funcs in headers because moving them
// to separate translation units will require relocatable device code compilation,
// which is rumoured to degrade performance.

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common_xpu_defs.dp.hpp"
#define DISABLE_FP16(TYPE_NAME) std::enable_if_t< \
                     std::is_same<float, TYPE_NAME>::value || \
                     std::is_same<double, TYPE_NAME>::value, bool> = true

// support only warp size = 32
template <typename scalar_t, DISABLE_FP16(scalar_t)>
void sum_warp(volatile scalar_t* sharr, const sycl::nd_item<3> &item_ct1) {
    int tidx = item_ct1.get_local_id(2) & 31;
    if (tidx < 16) {
        sharr[tidx] += sharr[tidx + 16];
        sharr[tidx] += sharr[tidx + 8];
        sharr[tidx] += sharr[tidx + 4];
        sharr[tidx] += sharr[tidx + 2];
        sharr[tidx] += sharr[tidx + 1];
    }
}


template <typename scalar_t, DISABLE_FP16(scalar_t)>
inline void gather_warp_execution_results(scalar_t* sharr, const uint16_t tidx) {
    sharr[tidx] = tidx * CUDA_WARP_SIZE < CUDA_MAX_NUM_THREADS_PER_BLOCK ? sharr[tidx * CUDA_WARP_SIZE] : static_cast<scalar_t>(0.0);
}


// Reduces the contents of a shared memory array of CUDA_MAX_NUM_THREADS_PER_BLOCK using
// warp-powered reduction. The final sum will be stored in the 0-th element of the shared memory array.
template <typename scalar_t, DISABLE_FP16(scalar_t)>
void reduce_in_block_using_warp_sums(scalar_t* __restrict__ sh_mem,
        uint16_t tidx,
        const sycl::nd_item<3> &item_ct1) {
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // Will reduce the summation to CUDA_MAX_WARPS_PER_BLOCK elements that are
    // spaced CUDA_WARP_SIZE elements apart in the shared memory
    sum_warp(sh_mem + (tidx & ~(CUDA_WARP_SIZE - 1)), item_ct1);

    /*
    DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    if (tidx < CUDA_MAX_WARPS_PER_BLOCK) {
        // Do warp reduction again - because currently CUDA_MAX_WARPS_PER_BLOCK == CUDA_WARP_SIZE, this
        // will lead to the 0-th element of the shared memory containing the entire per-block sum
        gather_warp_execution_results(sh_mem, tidx);
        sum_warp(sh_mem, item_ct1);
    }
}


bool last_block(int32_t* counter, uint32_t total_blocks_count,
                const sycl::nd_item<3> &item_ct1) {
    /*
    DPCT1078:2: Consider replacing memory_order::acq_rel with
    memory_order::seq_cst for correctness if strong memory order restrictions
    are needed.
    */
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);

    int last = 0;
    if (item_ct1.get_local_id(2) == 0) {
        last =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                counter, 1);
    }

    /*
    DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    return (item_ct1.barrier(),
            sycl::any_of_group(item_ct1.get_group(),
                               last == total_blocks_count - 1));
}


template <typename scalar_t, typename scalar_accum_t = scalar_t, DISABLE_FP16(scalar_accum_t)>
void reduce_with_shared_memory(
        scalar_accum_t* __restrict__ sh_arr,
        scalar_accum_t current_thread_sum,
        const uint16_t tidx,
        const uint32_t bidx,
        scalar_accum_t* __restrict__ dev_tmp,
        int32_t* __restrict__ dev_last_block_counter,
        scalar_t* __restrict__ grad,
        uint32_t total_number_of_blocks,
        const sycl::nd_item<3> &item_ct1) {

    // Put each thread sum element into shared memory (CUDA_MAX_NUM_THREADS_PER_BLOCK elements in total)
    sh_arr[tidx] = current_thread_sum;

    // Do warp reduction on the entire shared memory of a single block
    reduce_in_block_using_warp_sums(sh_arr, tidx, item_ct1);

    // Store the per-block sum for each block in the pre-allocated array (which has dimensions equal to grid dimensions)
    if (tidx == 0) {
        dev_tmp[bidx] = sh_arr[0];
    }

    // Synchronize blocks and make the last block of the grid do the reduction across the per-block sums
    // to obtain final sums
    if (last_block(dev_last_block_counter, total_number_of_blocks, item_ct1)) {

        // WARNING: seems like this will only work for total number of blocks to reduce across that is < CUDA_MAX_NUM_THREADS_PER_BLOCK
        sh_arr[tidx] = tidx < total_number_of_blocks ? dev_tmp[tidx] : static_cast<scalar_accum_t>(0.0);
        reduce_in_block_using_warp_sums(sh_arr, tidx, item_ct1);

        if (tidx == 0) {
            grad[0] = sh_arr[0];
        }
    }
}


#endif // _COMMON_XPU_FUNCS_DP_HPP_
