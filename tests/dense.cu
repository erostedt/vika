#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(dense, forward)
{
    using namespace vika;
    const auto inputs = DeviceOwningTensor2f::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}).unwrap();
    const auto weights = DeviceOwningTensor2f::from({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}).unwrap();
    const auto bias = DeviceOwningTensor1f::from({1.0f, 2.0f}).unwrap();
    auto outputs = DeviceOwningTensor2f::empty({2, 2}).unwrap();

    const u32 M = outputs.extent<0>();
    const u32 N = outputs.extent<1>();
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    dense_forward<<<grid, block>>>(inputs.const_view(), weights.const_view(), bias.const_view(), outputs.view());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto out = download(outputs).unwrap();
    const std::vector<f32> expected = {59.0f, 66.0f, 140.0f, 156.0f};

    EXPECT_EQ(out.rows(), 2u);
    EXPECT_EQ(out.cols(), 2u);
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}
