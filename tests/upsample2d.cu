#include "utest.h"

#include <numeric>
#include <vector>

#include "comparison.cuh"

#include "vika.cuh"

UTEST(upsample2d, forward)
{
    using namespace vika;
    // input [1, 2, 2, 1]:
    //   1 2
    //   3 4
    const auto inputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {1, 2, 2, 1}).unwrap();
    auto layer = Upsample2DLayer::with_extents(1, 2, 2, 1, 2).unwrap();
    const auto gpu_out = layer.forward({inputs.const_view()}).wait().unwrap();
    const auto out = download(gpu_out).unwrap();

    EXPECT_EQ(out.extent(0), 1u);
    EXPECT_EQ(out.extent(1), 4u);
    EXPECT_EQ(out.extent(2), 4u);
    EXPECT_EQ(out.extent(3), 1u);

    // Each input pixel replicated into its own 2x2 block, nearest-neighbor style:
    //   1 1 2 2
    //   1 1 2 2
    //   3 3 4 4
    //   3 3 4 4
    const std::vector<f32> expected = {1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4};
    for (usize h = 0; h < 4; ++h)
    {
        for (usize w = 0; w < 4; ++w)
        {
            EXPECT_NEAR(out(0, h, w, 0), expected[h * 4 + w], 1e-5f);
        }
    }
}

UTEST(upsample2d, forward_smaller_batch)
{
    using namespace vika;
    // Layer built with capacity 2, but the actual batch (1) is smaller.
    const auto inputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {1, 2, 2, 1}).unwrap();
    auto layer = Upsample2DLayer::with_extents(2, 2, 2, 1, 2).unwrap();
    const auto gpu_out = layer.forward({inputs.const_view()}).wait().unwrap();
    const auto out = download(gpu_out).unwrap();

    EXPECT_EQ(out.extent(0), 1u);
    EXPECT_EQ(out.extent(1), 4u);
    EXPECT_EQ(out.extent(2), 4u);
}

UTEST(upsample2d, forward_batch_exceeds_capacity)
{
    using namespace vika;
    const auto inputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 1, 2, 1}).unwrap();
    // Layer only has capacity for 1 sample, but the input has batch 2.
    auto layer = Upsample2DLayer::with_extents(1, 1, 2, 1, 2).unwrap();
    ASSERT_TRUE(layer.forward({inputs.const_view()}).is_error());
}

UTEST(upsample2d, backward)
{
    using namespace vika;
    // upstream [1, 4, 4, 1]:
    //    1  2  3  4
    //    5  6  7  8
    //    9 10 11 12
    //   13 14 15 16
    std::vector<f32> upstream_data(16);
    std::iota(upstream_data.begin(), upstream_data.end(), 1.0f);
    const auto upstream = DeviceOwningTensorf::from(upstream_data, {1, 4, 4, 1}).unwrap();

    auto layer = Upsample2DLayer::with_extents(1, 2, 2, 1, 2).unwrap();
    const auto d_inputs_job = layer.backward(upstream.const_view());
    ASSERT_TRUE(d_inputs_job.size() == 1);
    const auto d_inputs = download(d_inputs_job[0].wait().unwrap()).unwrap();

    // Reverse of forward: d_input(i, j) = sum of upstream's corresponding 2x2 block.
    //   (1+2+5+6)     (3+4+7+8)      = 14  22
    //   (9+10+13+14)  (11+12+15+16)  = 46  54
    EXPECT_NEAR(d_inputs(0, 0, 0, 0), 14.0f, 1e-5f);
    EXPECT_NEAR(d_inputs(0, 0, 1, 0), 22.0f, 1e-5f);
    EXPECT_NEAR(d_inputs(0, 1, 0, 0), 46.0f, 1e-5f);
    EXPECT_NEAR(d_inputs(0, 1, 1, 0), 54.0f, 1e-5f);
}

UTEST(upsample2d, with_extents_rejects_zero_scale)
{
    using namespace vika;
    ASSERT_TRUE(Upsample2DLayer::with_extents(1, 2, 2, 1, 0).is_error());
}
