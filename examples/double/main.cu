#include <cuda_runtime.h>
#include <iostream>

#include "vika.cuh"

int main()
{
    const u32 n = 5;
    float h_data[n] = {1, 2, 3, 4, 5};
    float *d_data = NULL;
    cudaError_t err;
    err = cudaMalloc((void **)&d_data, n * sizeof(int));

    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    err = cudaMemcpy(d_data, h_data, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return 1;
    }
    u32 threads = 128;
    u32 blocks = (n + threads - 1) / threads;
    double_kernel<<<blocks, threads>>>(d_data, n);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return 1;
    }
    err = cudaMemcpy(h_data, d_data, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return 1;
    }
    cudaFree(d_data);
    for (u32 i = 0; i < n; i++)
    {
        std::cout << h_data[i] << ' ';
    }
    std::cout << std::endl;
}
