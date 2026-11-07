#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void __launch_bounds__(256) rope_qk_forward_kernel(
        const __half* __restrict__ q,
        const __half* __restrict__ k,
        const float* __restrict__ cos_table,
        const float* __restrict__ sin_table,
        __half* __restrict__ q_out,
        __half* __restrict__ k_out,
        int batch,
        int sequence,
        int heads,
        int head_dim) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int total_pairs = batch * sequence * heads * half_dim;
    if (pair >= total_pairs) return;

    const int rotary_col = pair % half_dim;
    const int head = (pair / half_dim) % heads;
    const int position = (pair / (half_dim * heads)) % sequence;
    const int batch_index = pair / (half_dim * heads * sequence);
    const int64_t input_base =
        ((static_cast<int64_t>(batch_index) * sequence + position) * heads + head) * head_dim;
    const int64_t output_base =
        ((static_cast<int64_t>(batch_index) * heads + head) * sequence + position) * head_dim;
    const int64_t table_index = static_cast<int64_t>(position) * half_dim + rotary_col;

    const float cosine = cos_table[table_index];
    const float sine = sin_table[table_index];
    const float q0 = __half2float(q[input_base + rotary_col]);
    const float q1 = __half2float(q[input_base + rotary_col + half_dim]);
    const float k0 = __half2float(k[input_base + rotary_col]);
    const float k1 = __half2float(k[input_base + rotary_col + half_dim]);

    q_out[output_base + rotary_col] = __float2half(q0 * cosine - q1 * sine);
    q_out[output_base + rotary_col + half_dim] = __float2half(q1 * cosine + q0 * sine);
    k_out[output_base + rotary_col] = __float2half(k0 * cosine - k1 * sine);
    k_out[output_base + rotary_col + half_dim] = __float2half(k1 * cosine + k0 * sine);
}

__global__ void __launch_bounds__(256) rope_qk_backward_kernel(
        const __half* __restrict__ grad_q_out,
        const __half* __restrict__ grad_k_out,
        const float* __restrict__ cos_table,
        const float* __restrict__ sin_table,
        __half* __restrict__ grad_q,
        __half* __restrict__ grad_k,
        int batch,
        int sequence,
        int heads,
        int head_dim) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int half_dim = head_dim / 2;
    const int total_pairs = batch * sequence * heads * half_dim;
    if (pair >= total_pairs) return;

    const int rotary_col = pair % half_dim;
    const int head = (pair / half_dim) % heads;
    const int position = (pair / (half_dim * heads)) % sequence;
    const int batch_index = pair / (half_dim * heads * sequence);
    const int64_t input_base =
        ((static_cast<int64_t>(batch_index) * sequence + position) * heads + head) * head_dim;
    const int64_t output_base =
        ((static_cast<int64_t>(batch_index) * heads + head) * sequence + position) * head_dim;
    const int64_t table_index = static_cast<int64_t>(position) * half_dim + rotary_col;

    const float cosine = cos_table[table_index];
    const float sine = sin_table[table_index];
    const float gq0 = __half2float(grad_q_out[output_base + rotary_col]);
    const float gq1 = __half2float(grad_q_out[output_base + rotary_col + half_dim]);
    const float gk0 = __half2float(grad_k_out[output_base + rotary_col]);
    const float gk1 = __half2float(grad_k_out[output_base + rotary_col + half_dim]);

    grad_q[input_base + rotary_col] = __float2half(gq0 * cosine + gq1 * sine);
    grad_q[input_base + rotary_col + half_dim] = __float2half(gq1 * cosine - gq0 * sine);
    grad_k[input_base + rotary_col] = __float2half(gk0 * cosine + gk1 * sine);
    grad_k[input_base + rotary_col + half_dim] = __float2half(gk1 * cosine - gk0 * sine);
}

extern "C" void launch_rope_qk_forward(
        const void* q,
        const void* k,
        const void* cos_table,
        const void* sin_table,
        void* q_out,
        void* k_out,
        int batch,
        int sequence,
        int heads,
        int head_dim,
        void* stream) {
    const int total_pairs = batch * sequence * heads * (head_dim / 2);
    const int threads = 256;
    const int blocks = (total_pairs + threads - 1) / threads;
    hipLaunchKernelGGL(
        rope_qk_forward_kernel, dim3(blocks), dim3(threads), 0,
        reinterpret_cast<hipStream_t>(stream),
        static_cast<const __half*>(q), static_cast<const __half*>(k),
        static_cast<const float*>(cos_table), static_cast<const float*>(sin_table),
        static_cast<__half*>(q_out), static_cast<__half*>(k_out),
        batch, sequence, heads, head_dim);
}

extern "C" void launch_rope_qk_backward(
        const void* grad_q_out,
        const void* grad_k_out,
        const void* cos_table,
        const void* sin_table,
        void* grad_q,
        void* grad_k,
        int batch,
        int sequence,
        int heads,
        int head_dim,
        void* stream) {
    const int total_pairs = batch * sequence * heads * (head_dim / 2);
    const int threads = 256;
    const int blocks = (total_pairs + threads - 1) / threads;
    hipLaunchKernelGGL(
        rope_qk_backward_kernel, dim3(blocks), dim3(threads), 0,
        reinterpret_cast<hipStream_t>(stream),
        static_cast<const __half*>(grad_q_out), static_cast<const __half*>(grad_k_out),
        static_cast<const float*>(cos_table), static_cast<const float*>(sin_table),
        static_cast<__half*>(grad_q), static_cast<__half*>(grad_k),
        batch, sequence, heads, head_dim);
}
