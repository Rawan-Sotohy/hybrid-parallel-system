#include <cuda_runtime.h>

// GPU kernel
__global__ void transpose_kernel(double* in, double* out, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        out[col * n + row] = in[row * n + col];
    }
}

// CPU-callable wrapper 
void cuda_matrix_transpose(double* in, double* out, int n) {
    double *d_in, *d_out;
    size_t bytes = n * n * sizeof(double);

    // Allocate GPU memory
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);

    // Copy input from CPU RAM to GPU VRAM
    cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    // 16x16 = 256 threads per block, standard for 2D matrix work
    // blocks = how many blocks needed to cover the full matrix
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    transpose_kernel<<<blocks, threads>>>(d_in, d_out, n);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result from GPU VRAM back to CPU RAM
    cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_in);
    cudaFree(d_out);
}
