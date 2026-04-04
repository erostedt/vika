#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"
#include <vector>

UTEST(conv, forward_valid_stride1_multi_channel)
{
    using namespace vika;
    constexpr usize batch = 2;
    constexpr usize height = 4;
    constexpr usize width = 4;
    constexpr usize channels = 2;

    auto cpu_inputs = HostTensor4f::zero({batch, height, width, channels});
    for (usize n = 0; n < batch; ++n)
    {
        for (usize h = 0; h < height; ++h)
        {
            for (usize w = 0; w < width; ++w)
            {
                for (usize c = 0; c < channels; ++c)
                {
                    cpu_inputs(n, h, w, c) = static_cast<f32>(n * 1000 + c * 100 + h * 10 + w);
                }
            }
        }
    }
    const auto inputs = upload(cpu_inputs).unwrap();

    constexpr usize kernel_height = 3;
    constexpr usize kernel_width = 3;
    constexpr usize out_channels = 2;
    std::vector<f32> weights(kernel_height * kernel_width * channels * out_channels);
    std::iota(std::begin(weights), std::end(weights), 0.0f);
    auto filters = DeviceOwningTensor4f::from(weights, {kernel_height, kernel_width, out_channels, channels}).unwrap();
    auto biases = DeviceOwningTensor1f::from({0, 0}).unwrap();
    auto layer = Conv2DLayer::with_weights(batch, height, width, std::move(filters), std::move(biases), 1, 0).unwrap();

    const auto gpu_out = layer.forward(inputs.const_view()).wait().unwrap();
    const auto out = download(gpu_out).unwrap();

    EXPECT_EQ(out.extent<0>(), batch);
    EXPECT_EQ(out.extent<1>(), 2u);
    EXPECT_EQ(out.extent<2>(), 2u);
    EXPECT_EQ(out.extent<3>(), out_channels);

    EXPECT_NEAR(out(0, 0, 0, 0), 21054.0f, 1e-4f);
    EXPECT_NEAR(out(0, 0, 1, 0), 21360.0f, 1e-4f);
    EXPECT_NEAR(out(0, 1, 0, 0), 24114.0f, 1e-4f);
    EXPECT_NEAR(out(0, 1, 1, 0), 24420.0f, 1e-4f);
    EXPECT_NEAR(out(0, 0, 0, 1), 22152.0f, 1e-4f);
    EXPECT_NEAR(out(0, 0, 1, 1), 22476.0f, 1e-4f);
    EXPECT_NEAR(out(0, 1, 0, 1), 25392.0f, 1e-4f);
    EXPECT_NEAR(out(0, 1, 1, 1), 25716.0f, 1e-4f);

    EXPECT_NEAR(out(1, 0, 0, 0), 327054.0f, 1e-4f);
    EXPECT_NEAR(out(1, 0, 1, 0), 327360.0f, 1e-4f);
    EXPECT_NEAR(out(1, 1, 0, 0), 330114.0f, 1e-4f);
    EXPECT_NEAR(out(1, 1, 1, 0), 330420.0f, 1e-4f);
    EXPECT_NEAR(out(1, 0, 0, 1), 346152.0f, 1e-4f);
    EXPECT_NEAR(out(1, 0, 1, 1), 346476.0f, 1e-4f);
    EXPECT_NEAR(out(1, 1, 0, 1), 349392.0f, 1e-4f);
    EXPECT_NEAR(out(1, 1, 1, 1), 349716.0f, 1e-4f);
}
