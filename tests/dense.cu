#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(dense, layer_forward)
{
    using namespace vika;
    const auto inputs = DeviceOwningTensor2f::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}).unwrap();
    auto weights = DeviceOwningTensor2f::from({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}).unwrap();
    auto bias = DeviceOwningTensor1f::from({1.0f, 2.0f}).unwrap();

    auto layer = DenseLayer::with_weights(2, std::move(weights), std::move(bias)).unwrap();
    const auto outputs = layer.forward(inputs.const_view()).wait().unwrap();
    const auto out = download(outputs).unwrap();
    const std::vector<f32> expected = {59.0f, 66.0f, 140.0f, 156.0f};

    EXPECT_EQ(out.rows(), 2u);
    EXPECT_EQ(out.cols(), 2u);
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}

UTEST(dense, layer_backward)
{
    using namespace vika;
    const auto d_outputs = DeviceOwningTensor2f::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    auto weights = DeviceOwningTensor2f::from({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, {3, 2}).unwrap();
    auto bias = DeviceOwningTensor1f::from({1.0f, 2.0f}).unwrap();
    auto layer = DenseLayer::with_weights(2, std::move(weights), std::move(bias)).unwrap();

    const auto d_inputs = layer.backward(d_outputs.const_view()).wait().unwrap();

    const auto out = download(d_inputs).unwrap();
    const std::vector<f32> expected = {17.0f, 23.0f, 29.0f, 39.0f, 53.0f, 67.0f};

    EXPECT_EQ(out.rows(), 2u);
    EXPECT_EQ(out.cols(), 3u);
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}

UTEST(dense, layer_weight_gradients)
{
    using namespace vika;
    const auto inputs = DeviceOwningTensor2f::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}).unwrap();
    const auto d_outputs = DeviceOwningTensor2f::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();

    auto weights = DeviceOwningTensor2f::from({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}).unwrap();
    auto bias = DeviceOwningTensor1f::from({1.0f, 2.0f}).unwrap();

    auto layer = DenseLayer::with_weights(2, std::move(weights), std::move(bias)).unwrap();

    const auto [d_weights, d_biases] =
        layer.weight_gradients(inputs.const_view(), d_outputs.const_view()).wait().unwrap();

    const auto d_weights_cpu = download(d_weights).unwrap();
    const auto d_biases_cpu = download(d_biases).unwrap();

    const std::vector<f32> expected_d_weights = {13.0f, 18.0f, 17.0f, 24.0f, 21.0f, 30.0f};
    const std::vector<f32> expected_d_biases = {4.0f, 6.0f};

    EXPECT_EQ(d_weights_cpu.rows(), 3u);
    EXPECT_EQ(d_weights_cpu.cols(), 2u);
    EXPECT_EQ(d_biases.element_count(), 2);
    ASSERT_TRUE(are_close(d_weights_cpu, expected_d_weights, 1e-5f));
    ASSERT_TRUE(are_close(d_biases_cpu, expected_d_biases, 1e-5f));
}
