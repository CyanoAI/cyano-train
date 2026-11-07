#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__device__ __forceinline__ float block_sum(float value, float* shared) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down(value, offset);
    }
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        shared[threadIdx.x / warpSize] = value;
    }
    __syncthreads();

    float total = 0.0f;
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) {
        total = shared[threadIdx.x];
    }
    if (threadIdx.x < warpSize) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            total += __shfl_down(total, offset);
        }
    }
    return total;
}

__global__ void __launch_bounds__(256) rmsnorm_residual_forward_kernel(
        const __half* __restrict__ x,
        const __half* __restrict__ weight,
        __half* __restrict__ output,
        float* __restrict__ inv_rms,
        int rows,
        int hidden,
        float eps) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int64_t base = static_cast<int64_t>(row) * hidden;
    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = __half2float(x[base + col]);
        sum_sq += value * value;
    }

    __shared__ float partial[4];
    __shared__ float row_inv_rms;
    const float total = block_sum(sum_sq, partial);
    if (threadIdx.x == 0) {
        row_inv_rms = rsqrtf(total / static_cast<float>(hidden) + eps);
        inv_rms[row] = row_inv_rms;
    }
    __syncthreads();

    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = __half2float(x[base + col]);
        output[base + col] = __float2half(
            value * row_inv_rms * __half2float(weight[col]));
    }
}

__global__ void __launch_bounds__(256) rmsnorm_residual_backward_kernel(
        const __half* __restrict__ grad_output,
        const __half* __restrict__ x,
        const __half* __restrict__ weight,
        const float* __restrict__ inv_rms,
        __half* __restrict__ grad_x,
        float* __restrict__ grad_weight,
        int rows,
        int hidden) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int64_t base = static_cast<int64_t>(row) * hidden;
    const float inverse = inv_rms[row];
    float dot = 0.0f;
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = __half2float(x[base + col]);
        const float grad = __half2float(grad_output[base + col]);
        dot += grad * __half2float(weight[col]) * value;
    }

    __shared__ float partial[4];
    __shared__ float row_dot;
    const float total = block_sum(dot, partial);
    if (threadIdx.x == 0) row_dot = total;
    __syncthreads();

    const float correction = inverse * inverse * inverse * row_dot /
                             static_cast<float>(hidden);
    for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
        const float value = __half2float(x[base + col]);
        const float grad = __half2float(grad_output[base + col]);
        const float input_grad = grad * __half2float(weight[col]) * inverse -
                                 value * correction;
        const __half input_grad_half = __float2half(input_grad);
        grad_x[base + col] = input_grad_half;
        atomicAdd(grad_weight + col, grad * value * inverse);
    }
}

extern "C" void launch_rmsnorm_residual_forward(
        const void* x,
        const void* weight,
        void* output,
        void* inv_rms,
        int rows,
        int hidden,
        float eps,
        void* stream) {
    hipLaunchKernelGGL(
        rmsnorm_residual_forward_kernel,
        dim3(rows), dim3(256), 0, reinterpret_cast<hipStream_t>(stream),
        static_cast<const __half*>(x),
        static_cast<const __half*>(weight),
        static_cast<__half*>(output),
        static_cast<float*>(inv_rms),
        rows, hidden, eps);
}

extern "C" void launch_rmsnorm_residual_backward(
        const void* grad_output,
        const void* x,
        const void* weight,
        const void* inv_rms,
        void* grad_x,
        void* grad_weight,
        int rows,
        int hidden,
        void* stream) {
    hipLaunchKernelGGL(
        rmsnorm_residual_backward_kernel,
        dim3(rows), dim3(256), 0, reinterpret_cast<hipStream_t>(stream),
        static_cast<const __half*>(grad_output),
        static_cast<const __half*>(x),
        static_cast<const __half*>(weight),
        static_cast<const float*>(inv_rms),
        static_cast<__half*>(grad_x),
        static_cast<float*>(grad_weight),
        rows, hidden);
}
