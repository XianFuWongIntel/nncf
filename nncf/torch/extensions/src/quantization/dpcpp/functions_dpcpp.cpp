#include <vector>
#include <torch/extension.h>

#include "common_defs.h"
#include "quantization/functions_dpcpp_impl.h"

#define CHECK_INPUT(x) \
    CHECK_CPU(x);      \
    CHECK_CONTIGUOUS(x)

at::Tensor q_forward(
    at::Tensor input,
    at::Tensor input_low,
    at::Tensor input_range,
    int levels)
{
    CHECK_INPUT(input);
    CHECK_INPUT(input_low);
    CHECK_INPUT(input_range);

    return q_dpcpp_forward(input, input_low, input_range, levels);
}

std::vector<at::Tensor> q_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high) 
{
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    CHECK_INPUT(input_low);
    CHECK_INPUT(input_range);
    return q_dpcpp_backward(grad_output, input, input_low, input_range, levels, level_low, level_high);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("Quantize_forward", &q_forward, "Quantize forward (DPCPP)");
    m.def("Quantize_backward", &q_backward, "Quantize backward (DPCPP)");
}
