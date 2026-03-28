#include <cuda_runtime.h>
#include <iostream>

#define VIKA_IMPLEMENTATION
#include "vika.cuh"

int main()
{
    using namespace vika;
    auto a = DeviceOwningTensor2f::from({1, 2, 3, 4, 5, 6}, {2, 3}).unwrap();
    auto b = DeviceOwningTensor2f::from({7, 8, 9, 10, 11, 12}, {3, 2}).unwrap();
    auto c = DeviceOwningTensor2f::empty({2, 2}).unwrap();
    auto d = DeviceOwningTensor1f::from({1, 2}).unwrap();
    auto layer = DenseLayer::with_weights(4, std::move(b), std::move(d));

    u32 M = 2;
    u32 N = 2;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(a.const_view(), b.const_view(), c.view());
    const auto err = cudaDeviceSynchronize();
    if (is_error(err))
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    const auto out = download(c).unwrap();

    for (u32 i = 0; i < out.size(); i++)
    {
        std::cout << out[i] << ' ';
    }
    std::cout << std::endl;
}
