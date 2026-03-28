#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(matmul, small)
{
    using namespace vika;
    auto a = DeviceOwningTensor2f::from({1, 2, 3, 4, 5, 6}, {2, 3}).unwrap();
    auto b = DeviceOwningTensor2f::from({7, 8, 9, 10, 11, 12}, {3, 2}).unwrap();
    auto c = DeviceOwningTensor2f::empty({2, 2}).unwrap();

    u32 M = 2;
    u32 N = 2;
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block>>>(a.const_view(), b.const_view(), c.view());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto out = download(c).unwrap();
    const std::vector<f32> expected = {58, 64, 139, 154};
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}
