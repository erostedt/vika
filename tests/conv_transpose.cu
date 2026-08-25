#include "utest.h"

#include <vector>

#include "comparison.cuh"

#include "vika.cuh"

UTEST(conv_transpose, forward)
{
    using namespace vika;
    // input [1, 2, 2, 1]:
    //   1 2
    //   3 4
    const auto input = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {1, 2, 2, 1}).unwrap();
    // filters [kH, kW, C_out, C_in] = [2, 2, 1, 1]:
    //   1 2
    //   3 4
    auto filters = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2, 1, 1}).unwrap();
    auto biases = DeviceOwningTensorf::from({0.0f}).unwrap();

    auto layer = ConvTranspose2DLayer::with_weights(1, 2, 2, std::move(filters), std::move(biases), 1, 0).unwrap();
    const auto gpu_out = layer.forward({input.const_view()}).wait().unwrap();
    const auto out = download(gpu_out).unwrap();

    EXPECT_EQ(out.extent(0), 1u);
    EXPECT_EQ(out.extent(1), 3u);
    EXPECT_EQ(out.extent(2), 3u);
    EXPECT_EQ(out.extent(3), 1u);

    // Every input pixel scatters into a 2x2 block of the (larger) output via
    // oh = ih*stride - padding + kh, weighted by the filter and summed where blocks overlap -
    // hand-derived from ConvTranspose2D's definition, cross-checked two independent ways
    // (gathering per output cell, and scattering per input cell) before writing this in.
    const std::vector<f32> expected = {1, 4, 4, 6, 20, 16, 9, 24, 16};
    for (usize h = 0; h < 3; ++h)
    {
        for (usize w = 0; w < 3; ++w)
        {
            EXPECT_NEAR(out(0, h, w, 0), expected[h * 3 + w], 1e-4f);
        }
    }
}

UTEST(conv_transpose, backward)
{
    using namespace vika;
    // upstream [1, 3, 3, 1], all ones - every input position's 2x2 receptive field lies fully
    // inside the 3x3 output for this stride/padding/kernel combo, so each d_input should equal
    // the sum of every filter weight regardless of position.
    const std::vector<f32> ones(9, 1.0f);
    const auto upstream = DeviceOwningTensorf::from(ones, {1, 3, 3, 1}).unwrap();

    auto filters = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2, 1, 1}).unwrap();
    auto biases = DeviceOwningTensorf::from({0.0f}).unwrap();
    auto layer = ConvTranspose2DLayer::with_weights(1, 2, 2, std::move(filters), std::move(biases), 1, 0).unwrap();

    auto jobs = layer.backward(upstream.const_view());
    ASSERT_TRUE(jobs.size() == 1);
    const auto d_inputs = download(jobs[0].wait().unwrap()).unwrap();

    for (usize h = 0; h < 2; ++h)
    {
        for (usize w = 0; w < 2; ++w)
        {
            EXPECT_NEAR(d_inputs(0, h, w, 0), 10.0f, 1e-4f);
        }
    }
}

UTEST(conv_transpose, weight_gradients)
{
    using namespace vika;
    const auto input = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {1, 2, 2, 1}).unwrap();
    const std::vector<f32> ones(9, 1.0f);
    const auto upstream = DeviceOwningTensorf::from(ones, {1, 3, 3, 1}).unwrap();

    auto filters = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2, 1, 1}).unwrap();
    auto biases = DeviceOwningTensorf::from({0.0f}).unwrap();
    auto layer = ConvTranspose2DLayer::with_weights(1, 2, 2, std::move(filters), std::move(biases), 1, 0).unwrap();

    const auto [d_filters, d_biases] =
        layer.weight_gradients(input.const_view(), upstream.const_view()).wait().unwrap();
    const auto d_filters_cpu = download(d_filters).unwrap();
    const auto d_biases_cpu = download(d_biases).unwrap();

    // Every filter tap is fed by all 4 input positions equally (same full-overlap reasoning as
    // the backward test above), so each should equal the sum of every input value.
    const std::vector<f32> expected_d_filters = {10.0f, 10.0f, 10.0f, 10.0f};
    ASSERT_TRUE(are_close(d_filters_cpu, expected_d_filters, 1e-4f));
    ASSERT_TRUE(are_close(d_biases_cpu, std::vector<f32>{9.0f}, 1e-4f));
}

UTEST(conv_transpose, with_weights_rejects_bias_channel_mismatch)
{
    using namespace vika;
    auto filters = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2, 1, 1}).unwrap();
    // Wrong bias count (2 instead of 1 - filters' C_out is 1).
    auto biases = DeviceOwningTensorf::from({0.0f, 0.0f}).unwrap();
    ASSERT_TRUE(ConvTranspose2DLayer::with_weights(1, 2, 2, std::move(filters), std::move(biases), 1, 0).is_error());
}

UTEST(conv_transpose, forward_smaller_batch)
{
    using namespace vika;
    // Built for a batch of 2, handed 1 - the same view-slicing every other layer does, but this
    // is the newest layer and had no batch-related coverage of its own. Same single-sample input
    // and filters as forward() above, so the expected output is the same.
    const auto input = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {1, 2, 2, 1}).unwrap();
    auto filters = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2, 1, 1}).unwrap();
    auto biases = DeviceOwningTensorf::from({0.0f}).unwrap();

    auto layer = ConvTranspose2DLayer::with_weights(2, 2, 2, std::move(filters), std::move(biases), 1, 0).unwrap();
    const auto forwarded = layer.forward({input.const_view()}).wait();
    ASSERT_TRUE(forwarded.is_ok());
    const auto out = download(forwarded.unwrap()).unwrap();

    EXPECT_EQ(out.extent(0), 1u);
    EXPECT_EQ(out.extent(1), 3u);
    EXPECT_EQ(out.extent(2), 3u);
    ASSERT_TRUE(are_close(out, {1.0f, 4.0f, 4.0f, 6.0f, 20.0f, 16.0f, 9.0f, 24.0f, 16.0f}, 1e-4f));
}
