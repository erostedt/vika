#include <cuda_runtime.h>
#include <iostream>

#include "vika.cuh"

int main()
{
    auto tensor = CudaOwningTensor<f32, 1>::create({5}).unwrap();
    std::vector<f32> data = {1, 2, 3, 4, 5};
    cudaError_t err = tensor.upload(data);
    if (err != cudaSuccess)
    {
        std::cerr << "Upload failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    u32 threads = 128;
    u32 blocks = (data.size() + threads - 1) / threads;
    double_kernel<<<blocks, threads>>>(tensor.data(), data.size());
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    std::vector<f32> out;
    err = tensor.download(out);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    for (u32 i = 0; i < out.size(); i++)
    {
        std::cout << out[i] << ' ';
    }
    std::cout << std::endl;
}
