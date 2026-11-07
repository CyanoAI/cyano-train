#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void __launch_bounds__(256) swiglu_forward_kernel(
        const __half* __restrict__ gate,
        const __half* __restrict__ up,
        __half* __restrict__ output,
        int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const float gate_value = __half2float(gate[index]);
    const float up_value = __half2float(up[index]);
    const float silu = gate_value * sigmoid(gate_value);
    output[index] = __float2half(silu * up_value);
}

__global__ void __launch_bounds__(256) swiglu_backward_kernel(
        const __half* __restrict__ grad_output,
        const __half* __restrict__ gate,
        const __half* __restrict__ up,
        __half* __restrict__ grad_gate,
        __half* __restrict__ grad_up,
        int64_t elements) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) return;

    const float grad = __half2float(grad_output[index]);
    const float gate_value = __half2float(gate[index]);
    const float up_value = __half2float(up[index]);
    const float sigmoid_value = sigmoid(gate_value);
    const float silu = gate_value * sigmoid_value;
    const float silu_derivative = sigmoid_value *
        (1.0f + gate_value * (1.0f - sigmoid_value));

    grad_gate[index] = __float2half(grad * up_value * silu_derivative);
    grad_up[index] = __float2half(grad * silu);
}

extern "C" void launch_swiglu_forward(
        const void* gate,
        const void* up,
        void* output,
        int64_t elements,
        void* stream) {
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    hipLaunchKernelGGL(
        swiglu_forward_kernel,
        dim3(blocks), dim3(threads), 0,
        reinterpret_cast<hipStream_t>(stream),
        static_cast<const __half*>(gate),
        static_cast<const __half*>(up),
        static_cast<__half*>(output),
        elements);
}

extern "C" void launch_swiglu_backward(
        const void* grad_output,
        const void* gate,
        const void* up,
        void* grad_gate,
        void* grad_up,
        int64_t elements,
        void* stream) {
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    hipLaunchKernelGGL(
        swiglu_backward_kernel,
        dim3(blocks), dim3(threads), 0,
        reinterpret_cast<hipStream_t>(stream),
        static_cast<const __half*>(grad_output),
        static_cast<const __half*>(gate),
        static_cast<const __half*>(up),
        static_cast<__half*>(grad_gate),
        static_cast<__half*>(grad_up),
        elements);
}
