#ifndef _QUANTIZATION_FUNCTIONS_XPU_IMPL_H_
#define _QUANTIZATION_FUNCTIONS_XPU_IMPL_H_

at::Tensor q_xpu_forward(
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels);


std::vector<at::Tensor> q_xpu_backward(
        at::Tensor grad_output,
        at::Tensor input,
        at::Tensor input_low,
        at::Tensor input_range,
        int levels,
        int level_low,
        int level_high);

#endif // _QUANTIZATION_FUNCTIONS_XPU_IMPL_H_
