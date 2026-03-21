#include "utest.h"

#include "vika.cuh"

UTEST(matmul, small)
{
    auto a = CudaOwningTensor2f::create({2, 3}).unwrap();
    auto b = CudaOwningTensor2f::create({3, 2}).unwrap();
    auto c = CudaOwningTensor2f::create({2, 2}).unwrap();
    std::vector<f32> data_a = {1, 2, 3, 4, 5, 6};
    std::vector<f32> data_b = {7, 8, 9, 10, 11, 12};
    ASSERT_EQ(a.upload(data_a), cudaSuccess);
    ASSERT_EQ(b.upload(data_b), cudaSuccess);

    u32 M = 2;
    u32 N = 2;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(a.const_view(), b.const_view(), c.view());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto out = c.download().unwrap();
    const std::vector<f32> expected = {58, 64, 139, 154};
    ASSERT_TRUE(out == expected);
}
