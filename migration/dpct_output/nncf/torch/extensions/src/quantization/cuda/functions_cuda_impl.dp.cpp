#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "common_cuda_funcs.dp.hpp"
#include "common_cuda_defs.dp.hpp"

enum class ScaleType
{
    SINGLE_SCALE,
    PER_WEIGHT_CHANNEL,
    PER_ACTIVATION_CHANNEL
};


ScaleType get_scale_type(const at::Tensor& input, const at::Tensor& input_low, const at::Tensor& input_range)
{
    TORCH_CHECK(input_low.dim() == input_range.dim(), "input_low and input_range have different dimensionality");
    uint64_t scale_dim = input_range.dim();
    for (int i = 0; i < scale_dim; i++)
    {
        TORCH_CHECK(input_low.size(i) == input_range.size(i), "input_low and input_range have different dimension sizes");
    }

    uint64_t scale_count = input_range.numel();

    if (scale_dim > 0)
    {
        // For (NxCxHxW) input/output tensors, it is assumed that input_range is
        // either (1) for single-scale quantization, or (Nx1x1x1) for
        // per-channel scale weights quantization, or (1xCx1x1) for per-channel
        // activation quantization
        if (input_range.size(0) > 1)
        {
            TORCH_CHECK(input_range.size(0) == input.size(0), "Scale count and weights input channel count is different");
            TORCH_CHECK(input_range.size(0) == scale_count, "Scale shape is not flat");
            return ScaleType::PER_WEIGHT_CHANNEL;
        }
        else if (scale_dim >= 2 && input_range.size(1) > 1)
        {
            TORCH_CHECK(input_range.size(1) == input.size(1), "Scale count and activations channel count is different");
            TORCH_CHECK(input_range.size(1) == scale_count, "Scale shape is not flat");
            return  ScaleType::PER_ACTIVATION_CHANNEL;
        }
        // For (1x1x1x1) input/output tensors, it is assumed that input_range
        // should be PER_WEIGHT_CHANNEL
        if (scale_count == 1)
            return ScaleType::PER_WEIGHT_CHANNEL;
    }

    return ScaleType::SINGLE_SCALE;
}


namespace {

template <typename scalar_t>
void fakeQuantize(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels
        ) {
    scalar_t s = (levels - 1) / (*input_range);
    // zero_point is referred as ZP in docs
    scalar_t zero_point = sycl::round((-(*input_low) * s));
    /*
    DPCT1064:12: Migrated min call is used in a macro definition and is not
    valid for all macro uses. Adjust the code.
    */
    /*
    DPCT1064:13: Migrated round call is used in a macro definition and is not
    valid for all macro uses. Adjust the code.
    */
    (*output) =
        sycl::round((sycl::min(sycl::max((*input), (*input_low)),
                               (double)((*input_low) + (*input_range))) -
                     (*input_low)) *
                        s -
                    zero_point) /
        s;
}

template <typename scalar_t>
void q_cuda_forward_kernel(
        scalar_t* __restrict__ output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels,
        const uint64_t size,
        const uint64_t contiguous_elements_per_scale,
        const uint64_t scale_count,
        const sycl::nd_item<3> &item_ct1) {
    const uint32_t idx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                         item_ct1.get_local_id(2);

    if (idx < size) {
        // "Scales" are derived from input_low/input_range
        uint64_t scale_idx = static_cast<uint64_t>(idx / contiguous_elements_per_scale) % scale_count;
        fakeQuantize<scalar_t>((output + idx), (input + idx), input_low + scale_idx, input_range + scale_idx, levels);
    }
}

template <typename scalar_t>
void calcGrad(
        scalar_t* __restrict__ val_grad_input,
        scalar_t* __restrict__ val_grad_input_low,
        scalar_t* __restrict__ val_grad_input_range,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ output,
        const scalar_t range_low,
        const scalar_t range_high,
        const scalar_t reverted_range,
        const scalar_t val_low_grad) {
    *val_grad_input_range = 0;
    *val_grad_input_low = 0;
    *val_grad_input = 0;

    if ((*input) < range_low) {
        (*val_grad_input_range) = val_low_grad * (*grad_output);
        (*val_grad_input_low) = (*grad_output);
    } else if ((*input) > range_high) {
        (*val_grad_input_range) = (*grad_output);
        (*val_grad_input_low) = (*grad_output);
    } else {
        (*val_grad_input_range) = (*grad_output) * (((*output) - (*input)) * reverted_range);
        (*val_grad_input) = (*grad_output);
    }
}


template <typename scalar_t, typename scalar_accum_t>
void q_single_scale_cuda_backward_kernel(
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_input_low,
        scalar_t* __restrict__ grad_input_range,
        scalar_accum_t* __restrict__ dev_tmp_range,
        scalar_accum_t* __restrict__ dev_tmp_low,
        int32_t* __restrict__ dev_last_block_counter_range,
        int32_t* __restrict__ dev_last_block_counter_low,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels,
        const scalar_t level_low,
        const scalar_t level_high,
        const size_t size,
        const sycl::nd_item<3> &item_ct1,
        scalar_accum_t *sh_grad_range,
        scalar_accum_t *sh_grad_low) {
    const uint16_t tidx = item_ct1.get_local_id(2);
    const uint32_t bidx = item_ct1.get_group(2);
    const uint64_t gtidx = bidx * CUDA_MAX_NUM_THREADS_PER_BLOCK + tidx;
    const uint64_t grid_size =
        CUDA_MAX_NUM_THREADS_PER_BLOCK * item_ct1.get_group_range(2);

    scalar_accum_t sum_range = 0, sum_low = 0;
    scalar_t output, val_grad_input_range, val_grad_input_low;
    scalar_t alpha = level_low / level_high;
    scalar_t range_low = (*input_low);
    scalar_t range_high = (*input_low) + (*input_range);
    scalar_t reverted_range = 1 / (*input_range);
    for (size_t i = gtidx; i < size; i += grid_size) {
        fakeQuantize<scalar_t>(&output, (input + i), input_low, input_range, levels);
        calcGrad<scalar_t>((grad_input + i), &val_grad_input_low, &val_grad_input_range, (grad_output + i),
                 (input + i), &output, range_low, range_high, reverted_range, alpha);
        sum_range += val_grad_input_range;
        sum_low += val_grad_input_low;
    }

    reduce_with_shared_memory<scalar_t, scalar_accum_t>(
        sh_grad_range, sum_range, tidx, bidx, dev_tmp_range,
        dev_last_block_counter_range, grad_input_range,
        item_ct1.get_group_range(2), item_ct1);
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(
        sh_grad_low, sum_low, tidx, bidx, dev_tmp_low,
        dev_last_block_counter_low, grad_input_low, item_ct1.get_group_range(2),
        item_ct1);
}



template <typename scalar_t, typename scalar_accum_t>
void q_scale_per_weight_channel_cuda_backward_kernel(
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_input_low,
        scalar_t* __restrict__ grad_input_range,
        scalar_accum_t* __restrict__ dev_tmp_range,
        scalar_accum_t* __restrict__ dev_tmp_low,
        int32_t* __restrict__ dev_last_block_counter_range,
        int32_t* __restrict__ dev_last_block_counter_low,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels,
        const scalar_t level_low,
        const scalar_t level_high,
        const size_t elements_per_scale,
        const sycl::nd_item<3> &item_ct1,
        scalar_accum_t *sh_grad_range,
        scalar_accum_t *sh_grad_low) {
    const uint16_t tidx = item_ct1.get_local_id(2);
    const uint32_t scale_idx = item_ct1.get_group(2);
    const uint32_t per_scale_block_idx = item_ct1.get_group(1);

    const uint64_t per_scale_tidx = per_scale_block_idx * CUDA_MAX_NUM_THREADS_PER_BLOCK + tidx;
    const uint32_t total_blocks_per_scale = item_ct1.get_group_range(1);
    const uint64_t total_threads_per_scale = total_blocks_per_scale * CUDA_MAX_NUM_THREADS_PER_BLOCK;

    // Applying scale data offsets
    input_low += scale_idx;
    input_range += scale_idx;
    dev_tmp_low += scale_idx * total_blocks_per_scale;
    dev_tmp_range += scale_idx * total_blocks_per_scale;
    dev_last_block_counter_low += scale_idx;
    dev_last_block_counter_range += scale_idx;
    grad_input_low += scale_idx;
    grad_input_range += scale_idx;

    const size_t offset_for_scaled_quantized_elements = scale_idx * elements_per_scale;
    input += offset_for_scaled_quantized_elements;
    grad_input += offset_for_scaled_quantized_elements;
    grad_output += offset_for_scaled_quantized_elements;

    scalar_accum_t per_thread_grad_sum_range = 0, per_thread_grad_sum_low = 0;
    scalar_t output, val_grad_input_range, val_grad_input_low;
    scalar_t alpha = level_low / level_high;
    scalar_t range_low = (*input_low);
    scalar_t range_high = (*input_low) + (*input_range);
    scalar_t reverted_range = 1 / (*input_range);
    for (size_t i = per_scale_tidx; i < elements_per_scale; i += total_threads_per_scale) {
        fakeQuantize<scalar_t>(&output, (input + i), input_low, input_range, levels);
        calcGrad<scalar_t>((grad_input + i), &val_grad_input_low, &val_grad_input_range, (grad_output + i),
                 (input + i), &output, range_low, range_high, reverted_range, alpha);
        per_thread_grad_sum_range += val_grad_input_range;
        per_thread_grad_sum_low += val_grad_input_low;
    }

    reduce_with_shared_memory<scalar_t, scalar_accum_t>(
        sh_grad_range, per_thread_grad_sum_range, tidx, per_scale_block_idx,
        dev_tmp_range, dev_last_block_counter_range, grad_input_range,
        total_blocks_per_scale, item_ct1);
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(
        sh_grad_low, per_thread_grad_sum_low, tidx, per_scale_block_idx,
        dev_tmp_low, dev_last_block_counter_low, grad_input_low,
        total_blocks_per_scale, item_ct1);
}


template <typename scalar_t, typename scalar_accum_t>
void q_scale_per_activation_channel_cuda_backward_kernel(
        scalar_t* __restrict__ grad_input,
        scalar_t* __restrict__ grad_input_low,
        scalar_t* __restrict__ grad_input_range,
        scalar_accum_t* __restrict__ dev_tmp_range,
        scalar_accum_t* __restrict__ dev_tmp_low,
        int32_t* __restrict__ dev_last_block_counter_range,
        int32_t* __restrict__ dev_last_block_counter_low,
        const scalar_t* __restrict__ grad_output,
        const scalar_t* __restrict__ input,
        const scalar_t* __restrict__ input_low,
        const scalar_t* __restrict__ input_range,
        const scalar_t levels,
        const scalar_t level_low,
        const scalar_t level_high,
        const int64_t total_elements_per_scale,
        const int64_t contiguous_elements_per_scale,
        const int64_t scale_count,
        const int64_t leading_channel_offset,
        const sycl::nd_item<3> &item_ct1,
        scalar_accum_t *sh_grad_range,
        scalar_accum_t *sh_grad_low) {
    const uint16_t tidx = item_ct1.get_local_id(2);
    const uint32_t scale_idx = item_ct1.get_group(2);
    const uint32_t per_scale_block_idx = item_ct1.get_group(1);

    const uint64_t per_scale_tidx = per_scale_block_idx * CUDA_MAX_NUM_THREADS_PER_BLOCK + tidx;
    const uint32_t total_blocks_per_scale = item_ct1.get_group_range(1);
    const uint64_t total_threads_per_scale = total_blocks_per_scale * CUDA_MAX_NUM_THREADS_PER_BLOCK;

    // Applying scale data offsets
    input_low += scale_idx;
    input_range += scale_idx;
    dev_tmp_low += scale_idx * total_blocks_per_scale;
    dev_tmp_range += scale_idx * total_blocks_per_scale;
    dev_last_block_counter_low += scale_idx;
    dev_last_block_counter_range += scale_idx;
    grad_input_low += scale_idx;
    grad_input_range += scale_idx;

    scalar_accum_t per_thread_grad_sum_range = 0, per_thread_grad_sum_low = 0;
    scalar_t output, val_grad_input_range, val_grad_input_low;
    scalar_t alpha = level_low / level_high;
    scalar_t range_low = (*input_low);
    scalar_t range_high = (*input_low) + (*input_range);
    scalar_t reverted_range = 1 / (*input_range);


    // The blocks of values belonging to one and the same scale here are interleaved with a period
    // equal to contiguous_elements_per_scale. Will apply an offset to the beginning of the first
    // block of values belonging to the current scale of the thread block, and then, in the for loop, map
    // a contiguously changing loop iteration index into a value-block-skipping offset calculation pattern.

    const size_t initial_offset = scale_idx * contiguous_elements_per_scale;
    input += initial_offset;
    grad_input += initial_offset;
    grad_output += initial_offset;


    for (uint64_t i = per_scale_tidx; i < total_elements_per_scale; i += total_threads_per_scale) {
        size_t additional_offset = (i / contiguous_elements_per_scale) * leading_channel_offset + (i % contiguous_elements_per_scale);
        fakeQuantize<scalar_t>(&output, (input + additional_offset), input_low, input_range, levels);
        calcGrad<scalar_t>((grad_input + additional_offset), &val_grad_input_low, &val_grad_input_range, (grad_output + additional_offset),
                 (input + additional_offset), &output, range_low, range_high, reverted_range, alpha);
        per_thread_grad_sum_range += val_grad_input_range;
        per_thread_grad_sum_low += val_grad_input_low;
    }

    reduce_with_shared_memory<scalar_t, scalar_accum_t>(
        sh_grad_range, per_thread_grad_sum_range, tidx, per_scale_block_idx,
        dev_tmp_range, dev_last_block_counter_range, grad_input_range,
        total_blocks_per_scale, item_ct1);
    reduce_with_shared_memory<scalar_t, scalar_accum_t>(
        sh_grad_low, per_thread_grad_sum_low, tidx, per_scale_block_idx,
        dev_tmp_low, dev_last_block_counter_low, grad_input_low,
        total_blocks_per_scale, item_ct1);
}


}

at::Tensor q_cuda_forward(
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels) {
    at::DeviceGuard guard(input.device());
    const auto quantized_elements_count = input.numel();

    ScaleType scale_type = get_scale_type(input, input_low, input_range);

    uint64_t contiguous_elements_per_scale = 0;
    uint64_t scale_count = input_range.numel();
    switch (scale_type)
    {
        case ScaleType::PER_ACTIVATION_CHANNEL:
            // Scale count should be equal to 1-st input tensor dimension
            contiguous_elements_per_scale = quantized_elements_count / (input.size(0) * scale_count);
            break;
        case ScaleType::PER_WEIGHT_CHANNEL:
            // Scale count should be equal to 0-th input tensor dimension
            contiguous_elements_per_scale = quantized_elements_count / scale_count;
            break;
        default:
            contiguous_elements_per_scale = quantized_elements_count;
            break;
    }


    auto output = at::empty_like(input);

    /*
    DPCT1038:4: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    PROFILE(AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(),
                                                "q_cuda_forward", ([&] {
          /*
          DPCT1049:5: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
    ((sycl::queue *)(at::cuda::getCurrentCUDAStream()))
        ->submit([&](sycl::handler &cgh) {
          auto output_data_ptr_scalar_t_ct0 = output.data_ptr<scalar_t>();
          auto input_data_ptr_scalar_t_ct1 = input.data_ptr<scalar_t>();
          auto input_low_data_ptr_scalar_t_ct2 = input_low.data_ptr<scalar_t>();
          auto input_range_data_ptr_scalar_t_ct3 =
              input_range.data_ptr<scalar_t>();

          cgh.parallel_for(
              sycl::nd_range<3>(
                  sycl::range<3>(1, 1, GET_BLOCKS(quantized_elements_count)) *
                      sycl::range<3>(1, 1, CUDA_MAX_NUM_THREADS_PER_BLOCK),
                  sycl::range<3>(1, 1, CUDA_MAX_NUM_THREADS_PER_BLOCK)),
              [=](sycl::nd_item<3> item_ct1) {
                q_cuda_forward_kernel<scalar_t>(
                    output_data_ptr_scalar_t_ct0, input_data_ptr_scalar_t_ct1,
                    input_low_data_ptr_scalar_t_ct2,
                    input_range_data_ptr_scalar_t_ct3, levels,
                    quantized_elements_count, contiguous_elements_per_scale,
                    scale_count, item_ct1);
              });
        });
                                                }));)

    return output;
}


std::vector<at::Tensor> q_single_scale_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    at::DeviceGuard guard(input.device());
    const auto size = input.numel();
    auto grad_input = at::empty_like(grad_output);

    auto grad_input_range = at::empty({1}, grad_output.options());
    auto grad_input_low = at::empty({1}, grad_output.options());

    auto grid_size = std::min(GET_BLOCKS(size), CUDA_BLOCKS_PER_GRID_FOR_UNIFORM_ELTWISE);
    auto accum_options = get_accum_options(grad_output.options());

    auto dev_tmp_range = at::empty({grid_size}, accum_options);
    auto dev_tmp_low = at::empty({grid_size}, accum_options);
    auto dev_last_block_counter_range = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low = at::zeros({1},  at::device(grad_output.options().device()).dtype(at::kInt));

    /*
    DPCT1038:6: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    PROFILE(AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                input.scalar_type(), "q_single_scale_cuda_backward", ([&] {
      using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
      /*
      DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
      limit. To get the device limit, query info::device::max_work_group_size.
      Adjust the work-group size if needed.
      */
    ((sycl::queue *)(at::cuda::getCurrentCUDAStream()))
        ->submit([&](sycl::handler &cgh) {
          /*
          DPCT1101:14: 'CUDA_MAX_NUM_THREADS_PER_BLOCK' expression was replaced
          with a value. Modify the code to use the original expression, provided
          in comments, if it is correct.
          */
          sycl::local_accessor<scalar_accum_t, 1> sh_grad_range_acc_ct1(
              sycl::range<1>(1024 /*CUDA_MAX_NUM_THREADS_PER_BLOCK*/), cgh);
          /*
          DPCT1101:15: 'CUDA_MAX_NUM_THREADS_PER_BLOCK' expression was replaced
          with a value. Modify the code to use the original expression, provided
          in comments, if it is correct.
          */
          sycl::local_accessor<scalar_accum_t, 1> sh_grad_low_acc_ct1(
              sycl::range<1>(1024 /*CUDA_MAX_NUM_THREADS_PER_BLOCK*/), cgh);

          auto grad_input_data_ptr_scalar_t_ct0 =
              grad_input.data_ptr<scalar_t>();
          auto grad_input_low_data_ptr_scalar_t_ct1 =
              grad_input_low.data_ptr<scalar_t>();
          auto grad_input_range_data_ptr_scalar_t_ct2 =
              grad_input_range.data_ptr<scalar_t>();
          auto dev_tmp_range_data_ptr_scalar_accum_t_ct3 =
              dev_tmp_range.data_ptr<scalar_accum_t>();
          auto dev_tmp_low_data_ptr_scalar_accum_t_ct4 =
              dev_tmp_low.data_ptr<scalar_accum_t>();
          auto dev_last_block_counter_range_data_ptr_int32_t_ct5 =
              dev_last_block_counter_range.data_ptr<int32_t>();
          auto dev_last_block_counter_low_data_ptr_int32_t_ct6 =
              dev_last_block_counter_low.data_ptr<int32_t>();
          auto grad_output_data_ptr_scalar_t_ct7 =
              grad_output.data_ptr<scalar_t>();
          auto input_data_ptr_scalar_t_ct8 = input.data_ptr<scalar_t>();
          auto input_low_data_ptr_scalar_t_ct9 = input_low.data_ptr<scalar_t>();
          auto input_range_data_ptr_scalar_t_ct10 =
              input_range.data_ptr<scalar_t>();

          cgh.parallel_for(
              sycl::nd_range<3>(
                  sycl::range<3>(1, 1, grid_size) *
                      sycl::range<3>(1, 1, CUDA_MAX_NUM_THREADS_PER_BLOCK),
                  sycl::range<3>(1, 1, CUDA_MAX_NUM_THREADS_PER_BLOCK)),
              [=](sycl::nd_item<3> item_ct1) {
                q_single_scale_cuda_backward_kernel<scalar_t, scalar_accum_t>(
                    grad_input_data_ptr_scalar_t_ct0,
                    grad_input_low_data_ptr_scalar_t_ct1,
                    grad_input_range_data_ptr_scalar_t_ct2,
                    dev_tmp_range_data_ptr_scalar_accum_t_ct3,
                    dev_tmp_low_data_ptr_scalar_accum_t_ct4,
                    dev_last_block_counter_range_data_ptr_int32_t_ct5,
                    dev_last_block_counter_low_data_ptr_int32_t_ct6,
                    grad_output_data_ptr_scalar_t_ct7,
                    input_data_ptr_scalar_t_ct8,
                    input_low_data_ptr_scalar_t_ct9,
                    input_range_data_ptr_scalar_t_ct10, levels, level_low,
                    level_high, size, item_ct1,
                    sh_grad_range_acc_ct1.get_pointer(),
                    sh_grad_low_acc_ct1.get_pointer());
              });
        });
                }));)

    return {grad_input, grad_input_low, grad_input_range};
}



std::vector<at::Tensor> q_scale_per_weight_channel_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    at::DeviceGuard guard(input.device());
    const auto scale_count = input_range.size(0);
    const auto elements_per_scale = input.numel() / scale_count;

    auto grad_input = at::empty_like(grad_output);

    auto grad_input_low = at::empty(input_range.sizes(), grad_output.options());
    auto grad_input_range = at::empty(input_range.sizes(), grad_output.options());

    auto accum_options = get_accum_options(grad_output.options());
    sycl::range<3> grid_size = get_2d_grid_size_for_per_channel(scale_count);
    auto dev_tmp_range = at::zeros({grid_size[2], grid_size[1]}, accum_options);
    auto dev_tmp_low = at::zeros({grid_size[2], grid_size[1]}, accum_options);
    auto dev_last_block_counter_range =
        at::zeros({grid_size[2], 1},
                  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low =
        at::zeros({grid_size[2], 1},
                  at::device(grad_output.options().device()).dtype(at::kInt));

    /*
    DPCT1038:8: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    PROFILE(AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                input.scalar_type(), "q_single_scale_cuda_backward", ([&] {
              using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
              /*
              DPCT1049:9: The work-group size passed to the SYCL kernel may
              exceed the limit. To get the device limit, query
              info::device::max_work_group_size. Adjust the work-group size if
              needed.
              */
    ((sycl::queue *)(at::cuda::getCurrentCUDAStream()))
        ->submit([&](sycl::handler &cgh) {
          /*
          DPCT1101:16: 'CUDA_MAX_NUM_THREADS_PER_BLOCK' expression was replaced
          with a value. Modify the code to use the original expression, provided
          in comments, if it is correct.
          */
          sycl::local_accessor<scalar_accum_t, 1> sh_grad_range_acc_ct1(
              sycl::range<1>(1024 /*CUDA_MAX_NUM_THREADS_PER_BLOCK*/), cgh);
          /*
          DPCT1101:17: 'CUDA_MAX_NUM_THREADS_PER_BLOCK' expression was replaced
          with a value. Modify the code to use the original expression, provided
          in comments, if it is correct.
          */
          sycl::local_accessor<scalar_accum_t, 1> sh_grad_low_acc_ct1(
              sycl::range<1>(1024 /*CUDA_MAX_NUM_THREADS_PER_BLOCK*/), cgh);

          auto grad_input_data_ptr_scalar_t_ct0 =
              grad_input.data_ptr<scalar_t>();
          auto grad_input_low_data_ptr_scalar_t_ct1 =
              grad_input_low.data_ptr<scalar_t>();
          auto grad_input_range_data_ptr_scalar_t_ct2 =
              grad_input_range.data_ptr<scalar_t>();
          auto dev_tmp_range_data_ptr_scalar_accum_t_ct3 =
              dev_tmp_range.data_ptr<scalar_accum_t>();
          auto dev_tmp_low_data_ptr_scalar_accum_t_ct4 =
              dev_tmp_low.data_ptr<scalar_accum_t>();
          auto dev_last_block_counter_range_data_ptr_int32_t_ct5 =
              dev_last_block_counter_range.data_ptr<int32_t>();
          auto dev_last_block_counter_low_data_ptr_int32_t_ct6 =
              dev_last_block_counter_low.data_ptr<int32_t>();
          auto grad_output_data_ptr_scalar_t_ct7 =
              grad_output.data_ptr<scalar_t>();
          auto input_data_ptr_scalar_t_ct8 = input.data_ptr<scalar_t>();
          auto input_low_data_ptr_scalar_t_ct9 = input_low.data_ptr<scalar_t>();
          auto input_range_data_ptr_scalar_t_ct10 =
              input_range.data_ptr<scalar_t>();

          cgh.parallel_for(
              sycl::nd_range<3>(
                  grid_size *
                      sycl::range<3>(1, 1, CUDA_MAX_NUM_THREADS_PER_BLOCK),
                  sycl::range<3>(1, 1, CUDA_MAX_NUM_THREADS_PER_BLOCK)),
              [=](sycl::nd_item<3> item_ct1) {
                q_scale_per_weight_channel_cuda_backward_kernel<scalar_t,
                                                                scalar_accum_t>(
                    grad_input_data_ptr_scalar_t_ct0,
                    grad_input_low_data_ptr_scalar_t_ct1,
                    grad_input_range_data_ptr_scalar_t_ct2,
                    dev_tmp_range_data_ptr_scalar_accum_t_ct3,
                    dev_tmp_low_data_ptr_scalar_accum_t_ct4,
                    dev_last_block_counter_range_data_ptr_int32_t_ct5,
                    dev_last_block_counter_low_data_ptr_int32_t_ct6,
                    grad_output_data_ptr_scalar_t_ct7,
                    input_data_ptr_scalar_t_ct8,
                    input_low_data_ptr_scalar_t_ct9,
                    input_range_data_ptr_scalar_t_ct10, levels, level_low,
                    level_high, elements_per_scale, item_ct1,
                    sh_grad_range_acc_ct1.get_pointer(),
                    sh_grad_low_acc_ct1.get_pointer());
              });
        });
                }));)

    return {grad_input, grad_input_low, grad_input_range};
}


std::vector<at::Tensor> q_scale_per_activation_channel_cuda_backward(at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    at::DeviceGuard guard(input.device());
    const auto scale_count = input_range.size(1);
    const auto total_elements_per_scale = input.numel() / scale_count;
    const auto contiguous_elements_per_scale = input.numel() / (scale_count * input.size(0));
    const auto leading_channel_offset = input.numel() / input.size(0);

    auto grad_input = at::empty_like(grad_output);

    auto grad_input_low = at::empty(input_range.sizes(), grad_output.options());
    auto grad_input_range = at::empty(input_range.sizes(), grad_output.options());

    auto accum_options = get_accum_options(grad_output.options());
    sycl::range<3> grid_size = get_2d_grid_size_for_per_channel(scale_count);
    auto dev_tmp_range = at::zeros({grid_size[2], grid_size[1]}, accum_options);
    auto dev_tmp_low = at::zeros({grid_size[2], grid_size[1]}, accum_options);
    auto dev_last_block_counter_range =
        at::zeros({grid_size[2], 1},
                  at::device(grad_output.options().device()).dtype(at::kInt));
    auto dev_last_block_counter_low =
        at::zeros({grid_size[2], 1},
                  at::device(grad_output.options().device()).dtype(at::kInt));

    /*
    DPCT1038:10: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
    PROFILE(AT_DISPATCH_FLOATING_TYPES_AND_HALF(
                input.scalar_type(),
                "q_scale_per_activation_channel_cuda_backward", ([&] {
          using scalar_accum_t = ACCUM_TYPE_FOR(scalar_t);
          /*
          DPCT1049:11: The work-group size passed to the SYCL kernel may exceed
          the limit. To get the device limit, query
          info::device::max_work_group_size. Adjust the work-group size if
          needed.
          */
    ((sycl::queue *)(at::cuda::getCurrentCUDAStream()))
        ->submit([&](sycl::handler &cgh) {
          /*
          DPCT1101:18: 'CUDA_MAX_NUM_THREADS_PER_BLOCK' expression was replaced
          with a value. Modify the code to use the original expression, provided
          in comments, if it is correct.
          */
          sycl::local_accessor<scalar_accum_t, 1> sh_grad_range_acc_ct1(
              sycl::range<1>(1024 /*CUDA_MAX_NUM_THREADS_PER_BLOCK*/), cgh);
          /*
          DPCT1101:19: 'CUDA_MAX_NUM_THREADS_PER_BLOCK' expression was replaced
          with a value. Modify the code to use the original expression, provided
          in comments, if it is correct.
          */
          sycl::local_accessor<scalar_accum_t, 1> sh_grad_low_acc_ct1(
              sycl::range<1>(1024 /*CUDA_MAX_NUM_THREADS_PER_BLOCK*/), cgh);

          auto grad_input_data_ptr_scalar_t_ct0 =
              grad_input.data_ptr<scalar_t>();
          auto grad_input_low_data_ptr_scalar_t_ct1 =
              grad_input_low.data_ptr<scalar_t>();
          auto grad_input_range_data_ptr_scalar_t_ct2 =
              grad_input_range.data_ptr<scalar_t>();
          auto dev_tmp_range_data_ptr_scalar_accum_t_ct3 =
              dev_tmp_range.data_ptr<scalar_accum_t>();
          auto dev_tmp_low_data_ptr_scalar_accum_t_ct4 =
              dev_tmp_low.data_ptr<scalar_accum_t>();
          auto dev_last_block_counter_range_data_ptr_int32_t_ct5 =
              dev_last_block_counter_range.data_ptr<int32_t>();
          auto dev_last_block_counter_low_data_ptr_int32_t_ct6 =
              dev_last_block_counter_low.data_ptr<int32_t>();
          auto grad_output_data_ptr_scalar_t_ct7 =
              grad_output.data_ptr<scalar_t>();
          auto input_data_ptr_scalar_t_ct8 = input.data_ptr<scalar_t>();
          auto input_low_data_ptr_scalar_t_ct9 = input_low.data_ptr<scalar_t>();
          auto input_range_data_ptr_scalar_t_ct10 =
              input_range.data_ptr<scalar_t>();

          cgh.parallel_for(
              sycl::nd_range<3>(
                  grid_size *
                      sycl::range<3>(1, 1, CUDA_MAX_NUM_THREADS_PER_BLOCK),
                  sycl::range<3>(1, 1, CUDA_MAX_NUM_THREADS_PER_BLOCK)),
              [=](sycl::nd_item<3> item_ct1) {
                q_scale_per_activation_channel_cuda_backward_kernel<
                    scalar_t, scalar_accum_t>(
                    grad_input_data_ptr_scalar_t_ct0,
                    grad_input_low_data_ptr_scalar_t_ct1,
                    grad_input_range_data_ptr_scalar_t_ct2,
                    dev_tmp_range_data_ptr_scalar_accum_t_ct3,
                    dev_tmp_low_data_ptr_scalar_accum_t_ct4,
                    dev_last_block_counter_range_data_ptr_int32_t_ct5,
                    dev_last_block_counter_low_data_ptr_int32_t_ct6,
                    grad_output_data_ptr_scalar_t_ct7,
                    input_data_ptr_scalar_t_ct8,
                    input_low_data_ptr_scalar_t_ct9,
                    input_range_data_ptr_scalar_t_ct10, levels, level_low,
                    level_high, total_elements_per_scale,
                    contiguous_elements_per_scale, scale_count,
                    leading_channel_offset, item_ct1,
                    sh_grad_range_acc_ct1.get_pointer(),
                    sh_grad_low_acc_ct1.get_pointer());
              });
        });
                }));)

    return {grad_input, grad_input_low, grad_input_range};
}

std::vector<at::Tensor> q_cuda_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) {
    at::DeviceGuard guard(input.device());
    ScaleType scale_type = get_scale_type(input, input_low, input_range);

    switch (scale_type)
    {
        case ScaleType::PER_ACTIVATION_CHANNEL:
            return q_scale_per_activation_channel_cuda_backward(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
        case ScaleType::PER_WEIGHT_CHANNEL:
            return q_scale_per_weight_channel_cuda_backward(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
        case ScaleType::SINGLE_SCALE:
        default:
            return q_single_scale_cuda_backward(
                grad_output,
                input,
                input_low,
                input_range,
                levels,
                level_low,
                level_high);
    };
}
