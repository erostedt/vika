#include <cuda_runtime.h>
#include <iostream>

#define VIKA_IMPLEMENTATION
#include "vika.cuh"

int main()
{
    using namespace vika;
    auto a = CudaOwningTensor2f::empty({2, 3}).unwrap();
    auto b = CudaOwningTensor2f::empty({3, 2}).unwrap();
    auto c = CudaOwningTensor2f::empty({2, 2}).unwrap();
    std::vector<f32> data_a = {1, 2, 3, 4, 5, 6};
    std::vector<f32> data_b = {7, 8, 9, 10, 11, 12};
    cudaError_t err;
    err = a.upload(data_a);
    if (err != cudaSuccess)
    {
        std::cerr << "Upload failed a: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = b.upload(data_b);
    if (err != cudaSuccess)
    {
        std::cerr << "Upload failed b: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    u32 M = 2;
    u32 N = 2;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(a.const_view(), b.const_view(), c.view());
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    const auto out = c.download().unwrap();

    for (u32 i = 0; i < out.size(); i++)
    {
        std::cout << out[i] << ' ';
    }
    std::cout << std::endl;
}
