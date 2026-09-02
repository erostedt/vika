#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(dense, layer_forward)
{
    using namespace vika;
    const auto inputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}).unwrap();
    auto weights = DeviceOwningTensorf::from({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}).unwrap();
    auto bias = DeviceOwningTensorf::from({1.0f, 2.0f}).unwrap();

    auto layer = DenseLayer::with_weights(2, std::move(weights), std::move(bias)).unwrap();
    const auto outputs = layer.forward({inputs.const_view()}).wait().unwrap();
    const auto out = download(outputs).unwrap();
    const std::vector<f32> expected = {59.0f, 66.0f, 140.0f, 156.0f};

    EXPECT_EQ(out.extent(0), 2u);
    EXPECT_EQ(out.extent(1), 2u);
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}

UTEST(dense, layer_forward_smaller_batch)
{
    using namespace vika;
    // Same weights and first two input rows as layer_forward, but the layer has spare capacity
    // (4) while the actual batch (2) is smaller. forward() should slice down to just those 2
    // rows - both in what it computes and in the shape of what it returns - instead of operating
    // on, or reporting, the full capacity.
    const auto inputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}).unwrap();
    auto weights = DeviceOwningTensorf::from({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}).unwrap();
    auto bias = DeviceOwningTensorf::from({1.0f, 2.0f}).unwrap();

    auto layer = DenseLayer::with_weights(4, std::move(weights), std::move(bias)).unwrap();
    const auto outputs = layer.forward({inputs.const_view()}).wait().unwrap();
    const auto out = download(outputs).unwrap();
    const std::vector<f32> expected = {59.0f, 66.0f, 140.0f, 156.0f};

    EXPECT_EQ(out.extent(0), 2u);
    EXPECT_EQ(out.extent(1), 2u);
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}

UTEST(dense, layer_forward_batch_exceeds_capacity)
{
    using namespace vika;
    const auto inputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}).unwrap();
    auto weights = DeviceOwningTensorf::from({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}).unwrap();
    auto bias = DeviceOwningTensorf::from({1.0f, 2.0f}).unwrap();

    // Layer only has capacity for 1 sample, but the input batch has 2.
    auto layer = DenseLayer::with_weights(1, std::move(weights), std::move(bias)).unwrap();
    ASSERT_TRUE(failed_with(layer.forward({inputs.const_view()}), ErrorKind::Shape));
}

UTEST(dense, layer_backward)
{
    using namespace vika;
    const auto d_outputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    auto weights = DeviceOwningTensorf::from({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, {3, 2}).unwrap();
    auto bias = DeviceOwningTensorf::from({1.0f, 2.0f}).unwrap();
    auto layer = DenseLayer::with_weights(2, std::move(weights), std::move(bias)).unwrap();

    const auto d_inputs = layer.backward(d_outputs.const_view())[0].wait().unwrap();

    const auto out = download(d_inputs).unwrap();
    const std::vector<f32> expected = {17.0f, 23.0f, 29.0f, 39.0f, 53.0f, 67.0f};

    EXPECT_EQ(out.extent(0), 2u);
    EXPECT_EQ(out.extent(1), 3u);
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}

UTEST(dense, layer_backward_smaller_batch)
{
    using namespace vika;
    const auto d_outputs = DeviceOwningTensorf::from({1.0f, 2.0f}, {1, 2}).unwrap();
    auto weights = DeviceOwningTensorf::from({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, {3, 2}).unwrap();
    auto bias = DeviceOwningTensorf::from({1.0f, 2.0f}).unwrap();
    // Built with capacity for 2 samples, but only 1 upstream gradient row is actually passed in.
    auto layer = DenseLayer::with_weights(2, std::move(weights), std::move(bias)).unwrap();

    const auto d_inputs = layer.backward(d_outputs.const_view())[0].wait().unwrap();

    const auto out = download(d_inputs).unwrap();
    const std::vector<f32> expected = {17.0f, 23.0f, 29.0f};

    EXPECT_EQ(out.extent(0), 1u);
    EXPECT_EQ(out.extent(1), 3u);
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}

UTEST(dense, layer_weight_gradients_grid_coverage)
{
    using namespace vika;
    // feature_count and neuron_count both exceed a single 16x16 block, and both exceed
    // batch_size - the exact condition under which weight_gradients() used to size its launch
    // grid from d_inputs' shape (batch_capacity x feature_count) instead of weights.grad's own
    // shape (feature_count x neuron_count), silently leaving rows/columns beyond the wrong
    // grid's coverage unwritten. All-ones inputs/upstream make every entry of the expected
    // gradient identical (batch_size), so any uncovered element is immediately visible.
    constexpr usize batch_size = 2;
    constexpr usize feature_count = 17;
    constexpr usize neuron_count = 17;

    const auto inputs =
        DeviceOwningTensorf::from(std::vector<f32>(batch_size * feature_count, 1.0f), {batch_size, feature_count})
            .unwrap();
    const auto d_outputs =
        DeviceOwningTensorf::from(std::vector<f32>(batch_size * neuron_count, 1.0f), {batch_size, neuron_count})
            .unwrap();

    auto weights =
        DeviceOwningTensorf::empty({feature_count, neuron_count}).unwrap();
    auto bias = DeviceOwningTensorf::empty({neuron_count}).unwrap();
    auto layer = DenseLayer::with_weights(batch_size, std::move(weights), std::move(bias)).unwrap();

    const auto [d_weights, d_biases] =
        layer.weight_gradients(inputs.const_view(), d_outputs.const_view()).wait().unwrap();

    const auto d_weights_cpu = download(d_weights).unwrap();
    const auto d_biases_cpu = download(d_biases).unwrap();

    const std::vector<f32> expected_d_weights(feature_count * neuron_count, static_cast<f32>(batch_size));
    const std::vector<f32> expected_d_biases(neuron_count, static_cast<f32>(batch_size));

    ASSERT_TRUE(are_close(d_weights_cpu, expected_d_weights, 1e-5f));
    ASSERT_TRUE(are_close(d_biases_cpu, expected_d_biases, 1e-5f));
}

UTEST(dense, layer_weight_gradients)
{
    using namespace vika;
    const auto inputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}).unwrap();
    const auto d_outputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();

    auto weights = DeviceOwningTensorf::from({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}).unwrap();
    auto bias = DeviceOwningTensorf::from({1.0f, 2.0f}).unwrap();

    auto layer = DenseLayer::with_weights(2, std::move(weights), std::move(bias)).unwrap();

    const auto [d_weights, d_biases] =
        layer.weight_gradients(inputs.const_view(), d_outputs.const_view()).wait().unwrap();

    const auto d_weights_cpu = download(d_weights).unwrap();
    const auto d_biases_cpu = download(d_biases).unwrap();

    const std::vector<f32> expected_d_weights = {13.0f, 18.0f, 17.0f, 24.0f, 21.0f, 30.0f};
    const std::vector<f32> expected_d_biases = {4.0f, 6.0f};

    EXPECT_EQ(d_weights_cpu.extent(0), 3u);
    EXPECT_EQ(d_weights_cpu.extent(1), 2u);
    EXPECT_EQ(d_biases.element_count(), 2u);
    ASSERT_TRUE(are_close(d_weights_cpu, expected_d_weights, 1e-5f));
    ASSERT_TRUE(are_close(d_biases_cpu, expected_d_biases, 1e-5f));
}

// Adam's bias correction divides by 1 - beta^t, which is zero at t = 0: every scale becomes inf
// and every weight NaN. Model::step() owns the counter and cannot pass 0, but a layer's update()
// takes t straight from its caller.
UTEST(dense, layer_adam_update_rejects_step_zero)
{
    using namespace vika;
    auto layer = DenseLayer::randomized(1, 2, 3, 42).unwrap();

    std::vector<AdamState> states;
    for (const auto &param : layer.parameters())
    {
        states.push_back(AdamState::create(param.value.extents).unwrap());
    }

    const auto before = download(layer.weights.value).unwrap();

    auto jobs = layer.update(states, AdamParameters{}, 0);
    for (auto &result : wait_on_all(jobs))
    {
        EXPECT_TRUE(failed_with(result, ErrorKind::Unsupported));
    }

    // Rejected before any launch, so the weights are untouched rather than NaN.
    const auto after = download(layer.weights.value).unwrap();
    ASSERT_TRUE(are_equal(after, before));
}

UTEST(dense, layer_adam_update)
{
    using namespace vika;
    auto weights = DeviceOwningTensorf::from({0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f}, {2, 3}).unwrap();
    auto biases = DeviceOwningTensorf::from({0.01f, -0.02f, 0.03f}).unwrap();

    auto layer = DenseLayer::with_weights(1, std::move(weights), std::move(biases)).unwrap();

    // update() now reads gradients from the layer's own weights.grad/biases.grad via parameters()
    // instead of taking them as arguments, so fixed gradients (standing in for what
    // weight_gradients() would have computed) are fed in by overwriting those buffers directly.
    layer.weights.grad = DeviceOwningTensorf::from({0.7f, -0.8f, 0.9f, -1.0f, 1.1f, -1.2f}, {2, 3}).unwrap();
    layer.biases.grad = DeviceOwningTensorf::from({0.05f, -0.06f, 0.07f}).unwrap();

    const auto parameters = AdamParameters{
        .learning_rate = 0.1f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .epsilon = 1e-8f,
    };

    const std::vector<std::vector<f32>> expected_weights = {
        {7.0780516e-07f, -1.0000070e-01f, 2.0000072e-01f, 4.9999928e-01f, -5.9999931e-01f, 6.9999933e-01f},
        {-9.99982506e-02f, -1.73598528e-06f, 1.00001745e-01f, 5.99998236e-01f, -6.99998260e-01f, 7.99998283e-01f},
        {-1.9999787e-01f, 9.9997871e-02f, 2.1308661e-06f, 6.9999784e-01f, -7.9999787e-01f, 8.9999789e-01f},
    };

    const std::vector<std::vector<f32>> expected_biases = {
        {-0.08999871f, 0.07999881f, -0.06999888f},
        {-0.18999726f, 0.17999741f, -0.16999754f},
        {-0.2899965f, 0.27999675f, -0.2699969f},
    };

    const std::vector<std::vector<f32>> expected_m_weights = {
        {0.07f, -0.08000001f, 0.09f, -0.1f, 0.11000001f, -0.12f},
        {0.133f, -0.15200001f, 0.171f, -0.19f, 0.209f, -0.22800002f},
        {0.1897f, -0.2168f, 0.2439f, -0.271f, 0.2981f, -0.32520002f},
    };

    const std::vector<std::vector<f32>> expected_v_weights = {
        {0.00049f, 0.00064f, 0.00081f, 0.001f, 0.00121f, 0.00144f},
        {0.00097951f, 0.00127936f, 0.00161919f, 0.001999f, 0.00241879f, 0.00287856f},
        {0.00146853f, 0.00191808f, 0.00242757f, 0.002997f, 0.00362637f, 0.00431568f},
    };

    const std::vector<std::vector<f32>> expected_m_biases = {
        {0.005f, -0.006f, 0.007f},
        {0.0095f, -0.0114f, 0.0133f},
        {0.01355f, -0.01626f, 0.01897f},
    };

    const std::vector<std::vector<f32>> expected_v_biases = {
        {2.5000004e-06f, 3.6000001e-06f, 4.9000005e-06f},
        {4.9975006e-06f, 7.1964005e-06f, 9.7951006e-06f},
        {7.4925033e-06f, 1.0789205e-05f, 1.4685305e-05f},
    };

    // parameters() order is {weights, biases} (see DenseLayer::parameters), so states[0]/[1]
    // are the weights'/biases' Adam state respectively.
    std::vector<AdamState> states;
    for (const auto &param : layer.parameters())
    {
        states.push_back(AdamState::create(param.value.extents).unwrap());
    }

    for (usize step = 1; step <= 3; ++step)
    {
        auto jobs = layer.update(states, parameters, step);
        for (auto &result : wait_on_all(jobs))
        {
            result.unwrap();
        }

        const auto host_weights = download(layer.weights.value).unwrap();
        const auto host_biases = download(layer.biases.value).unwrap();
        const auto host_m_weights = download(states[0].m).unwrap();
        const auto host_v_weights = download(states[0].v).unwrap();
        const auto host_m_biases = download(states[1].m).unwrap();
        const auto host_v_biases = download(states[1].v).unwrap();

        const auto index = step - 1;
        ASSERT_TRUE(are_close(host_weights, expected_weights[index], 1e-5f));
        ASSERT_TRUE(are_close(host_biases, expected_biases[index], 1e-5f));
        ASSERT_TRUE(are_close(host_m_weights, expected_m_weights[index], 1e-5f));
        ASSERT_TRUE(are_close(host_v_weights, expected_v_weights[index], 1e-5f));
        ASSERT_TRUE(are_close(host_m_biases, expected_m_biases[index], 1e-5f));
        ASSERT_TRUE(are_close(host_v_biases, expected_v_biases[index], 1e-5f));
    }
}
