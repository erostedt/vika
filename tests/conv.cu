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

UTEST(conv, backward_valid_stride1)
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

    auto cpu_upstream = HostTensor4f::zero({batch, gpu_out.extents[1], gpu_out.extents[2], gpu_out.extents[3]});
    for (usize n = 0; n < batch; ++n)
    {
        for (usize out_h = 0; out_h < gpu_out.extents[1]; ++out_h)
        {
            for (usize out_w = 0; out_w < gpu_out.extents[2]; ++out_w)
            {
                for (usize out_c = 0; out_c < gpu_out.extents[3]; ++out_c)
                {
                    const auto idx =
                        ((n * gpu_out.extents[3] + out_c) * gpu_out.extents[1] + out_h) * gpu_out.extents[2] + out_w;
                    cpu_upstream(n, out_h, out_w, out_c) = static_cast<f32>(idx + 1) * 0.1f;
                }
            }
        }
    }

    const auto upstream = upload(cpu_upstream).unwrap();

    const auto d_inputs = download(layer.backward(upstream.const_view()).wait().unwrap()).unwrap();
    const auto [gpu_d_weights, gpu_d_biases] =
        layer.weight_gradients(inputs.const_view(), upstream.const_view()).wait().unwrap();
    const auto d_weights = download(gpu_d_weights).unwrap();
    const auto d_biases = download(gpu_d_biases).unwrap();

    EXPECT_EQ(d_inputs.extent<0>(), batch);
    EXPECT_EQ(d_inputs.extent<1>(), height);
    EXPECT_EQ(d_inputs.extent<2>(), width);
    EXPECT_EQ(d_inputs.extent<3>(), channels);

    EXPECT_NEAR(d_inputs(0, 0, 0, 0), 0.5f, 1e-4f);
    EXPECT_NEAR(d_inputs(0, 1, 1, 0), 25.8f, 1e-4f);
    EXPECT_NEAR(d_inputs(0, 3, 3, 0), 39.2f, 1e-4f);
    EXPECT_NEAR(d_inputs(0, 0, 0, 1), 1.7f, 1e-4f);
    EXPECT_NEAR(d_inputs(0, 2, 2, 1), 90.6f, 1e-4f);
    EXPECT_NEAR(d_inputs(1, 1, 1, 0), 80.2f, 1e-4f);
    EXPECT_NEAR(d_inputs(1, 3, 3, 0), 91.2f, 1e-4f);
    EXPECT_NEAR(d_inputs(1, 0, 0, 1), 5.7f, 1e-4f);
    EXPECT_NEAR(d_inputs(1, 2, 1, 1), 220.2f, 1e-4f);

    EXPECT_EQ(d_weights.extent<0>(), kernel_height);
    EXPECT_EQ(d_weights.extent<1>(), kernel_width);
    EXPECT_EQ(d_weights.extent<2>(), channels);
    EXPECT_EQ(d_weights.extent<3>(), out_channels);

    EXPECT_NEAR(d_weights(0, 0, 0, 0), 4232.8f, 1e-3f);
    EXPECT_NEAR(d_weights(1, 1, 0, 0), 4290.0f, 1e-3f);
    EXPECT_NEAR(d_weights(2, 2, 0, 0), 4347.2f, 1e-3f);
    EXPECT_NEAR(d_weights(0, 0, 1, 0), 4752.8f, 1e-3f);
    EXPECT_NEAR(d_weights(2, 2, 1, 0), 4867.2f, 1e-3f);
    EXPECT_NEAR(d_weights(0, 0, 0, 1), 5850.4f, 1e-3f);
    EXPECT_NEAR(d_weights(1, 1, 0, 1), 5942.8f, 1e-3f);
    EXPECT_NEAR(d_weights(2, 2, 0, 1), 6035.2f, 1e-3f);
    EXPECT_NEAR(d_weights(0, 0, 1, 1), 6690.4f, 1e-3f);
    EXPECT_NEAR(d_weights(2, 2, 1, 1), 6875.2f, 1e-3f);
}
