#include "utest.h"

#include "comparison.cuh"
#include "vika.cuh"

UTEST(model, xor_compile_execution_order)
{
    using namespace vika;

    ComputationGraph graph{4};
    auto x = graph.input({2});
    x = graph.dense(x, 8, 42).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 43).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();

    EXPECT_EQ(model.layers.size(), 5u);
    EXPECT_EQ(model.execution_order.size(), 5u);
    EXPECT_EQ(model.input_node.value, 0u);
    EXPECT_EQ(model.output_node.value, 4u);

    EXPECT_EQ(model.execution_order.front().value, 0u);
    EXPECT_EQ(model.execution_order.back().value, 4u);

    for (usize i = 0; i < model.execution_order.size(); ++i)
    {
        const auto &preds = model.layer_inputs[model.execution_order[i].value];
        for (const auto &pred : preds)
        {
            bool pred_seen = false;
            for (usize j = 0; j < i; ++j)
            {
                if (model.execution_order[j].value == pred.value)
                {
                    pred_seen = true;
                    break;
                }
            }
            EXPECT_TRUE(pred_seen);
        }
    }
}

UTEST(model, line_cnn_compile_execution_order)
{
    using namespace vika;

    ComputationGraph graph{8};
    auto x = graph.input({8, 8, 1});
    x = graph.conv2d(x, 3, 3, 8, 1, 0, 42).unwrap();
    x = graph.maxpool2d(x, 2, 2, 2).unwrap();
    x = graph.flatten(x).unwrap();
    x = graph.dense(x, 16, 43).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 44).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();

    EXPECT_EQ(model.layers.size(), 8u);
    EXPECT_EQ(model.execution_order.size(), 8u);
    EXPECT_EQ(model.input_node.value, 0u);
    EXPECT_EQ(model.output_node.value, 7u);

    EXPECT_EQ(model.execution_order.front().value, 0u);
    EXPECT_EQ(model.execution_order.back().value, 7u);
}

UTEST(model, compile_invalid_output_node)
{
    using namespace vika;
    ComputationGraph graph{4};
    graph.input({2});
    const auto result = graph.compile(NodeId{99});
    EXPECT_TRUE(result.is_error());
}

UTEST(model, compile_no_input_node)
{
    using namespace vika;
    ComputationGraph graph{4};
    auto x = graph.input({2});
    x = graph.dense(x, 4, 42).unwrap();
    graph.nodes[0].spec = DenseSpec{2, 0};
    const auto result = graph.compile(x);
    EXPECT_TRUE(result.is_error());
}

UTEST(model, compile_multiple_input_nodes)
{
    using namespace vika;
    ComputationGraph graph{4};
    graph.input({2});
    auto x = graph.input({2});
    x = graph.dense(x, 4, 42).unwrap();
    const auto result = graph.compile(x);
    EXPECT_TRUE(result.is_error());
}

UTEST(model, forward_matches_manual_xor)
{
    using namespace vika;

    constexpr usize batch_size = 4;
    constexpr u32 seed1 = 42;
    constexpr u32 seed2 = 43;

    const auto cpu_inputs = HostTensor2f::from({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}, {batch_size, 2}).unwrap();
    const auto gpu_inputs = upload(cpu_inputs).unwrap();

    // graph API forward
    ComputationGraph graph{batch_size};
    auto x = graph.input({2});
    x = graph.dense(x, 8, seed1).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, seed2).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();
    const auto graph_out = model.forward(gpu_inputs.const_view()).unwrap();
    const auto graph_cpu = download(graph_out).unwrap();

    // manual API forward using same seeds
    auto dense1 = DenseLayer::randomized(batch_size, 2, 8, seed1).unwrap();
    auto sigmoid1 = SigmoidLayer::with_extents({batch_size, 8}).unwrap();
    auto dense2 = DenseLayer::randomized(batch_size, 8, 1, seed2).unwrap();
    auto sigmoid2 = SigmoidLayer::with_extents({batch_size, 1}).unwrap();

    const auto out1 = dense1.forward({gpu_inputs.const_view()}).wait().unwrap();
    const auto act1 = sigmoid1.forward({out1}).wait().unwrap();
    const auto out2 = dense2.forward({act1}).wait().unwrap();
    const auto manual_out = sigmoid2.forward({out2}).wait().unwrap();
    const auto manual_cpu = download(manual_out).unwrap();

    EXPECT_EQ(graph_cpu.size(), manual_cpu.size());
    ASSERT_TRUE(are_close(graph_cpu, manual_cpu, 1e-5f));
}

UTEST(model, forward_output_shape)
{
    using namespace vika;

    constexpr usize batch_size = 8;

    ComputationGraph graph{batch_size};
    auto x = graph.input({8, 8, 1});
    x = graph.conv2d(x, 3, 3, 8, 1, 0, 42).unwrap();
    x = graph.maxpool2d(x, 2, 2, 2).unwrap();
    x = graph.flatten(x).unwrap();
    x = graph.dense(x, 16, 43).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 44).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();

    const auto gpu_inputs = DeviceOwningTensorf::empty({batch_size, 8, 8, 1}).unwrap();
    const auto out = model.forward(gpu_inputs.const_view()).unwrap();

    EXPECT_EQ(out.rank, 2u);
    EXPECT_EQ(out.extents[0], batch_size);
    EXPECT_EQ(out.extents[1], 1u);
}

UTEST(model, xor_trains_to_convergence)
{
    using namespace vika;

    constexpr usize batch_size = 4;

    const auto cpu_inputs = HostTensor2f::from({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}, {batch_size, 2}).unwrap();
    const auto cpu_targets = HostTensor2f::from({0.0f, 1.0f, 1.0f, 0.0f}, {batch_size, 1}).unwrap();
    const auto gpu_inputs = upload(cpu_inputs).unwrap();
    const auto gpu_targets = upload(cpu_targets).unwrap();

    ComputationGraph graph{batch_size};
    auto x = graph.input({2});
    x = graph.dense(x, 8, 42).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 43).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();
    auto loss_fn = MSELoss::with_extents({batch_size, 1}).unwrap();

    const AdamParameters adam_params{.learning_rate = 0.01f, .beta1 = 0.9f, .beta2 = 0.999f, .epsilon = 1e-8f};
    auto optimizer = AdamOptimizer::from_model(model, adam_params).unwrap();

    for (usize t = 1; t <= 10000; ++t)
    {
        const auto out = model.forward(gpu_inputs.const_view()).unwrap();
        const auto loss_grad = loss_fn.backward(out, gpu_targets.const_view()).wait().unwrap();
        model.backward(loss_grad).unwrap();
        model.step(optimizer, t).unwrap();
    }

    const auto out = model.forward(gpu_inputs.const_view()).unwrap();
    const auto preds = download(out).unwrap();

    EXPECT_TRUE(preds(0, 0) < 0.1f);
    EXPECT_TRUE(preds(1, 0) > 0.9f);
    EXPECT_TRUE(preds(2, 0) > 0.9f);
    EXPECT_TRUE(preds(3, 0) < 0.1f);
}

UTEST(model, branching_add_forward_and_backward)
{
    using namespace vika;

    // input -> denseA -\
    //                    add -> sigmoid -> output
    // input -> denseB -/
    constexpr usize batch_size = 2;

    ComputationGraph graph{batch_size};
    auto x = graph.input({2});
    auto a = graph.dense(x, 3, 42).unwrap();
    auto b = graph.dense(x, 3, 43).unwrap();
    auto sum = graph.add({a, b}).unwrap();
    auto out = graph.sigmoid(sum).unwrap();

    auto model = graph.compile(out).unwrap();

    const auto cpu_inputs = HostTensor2f::from({1.0f, 2.0f, 3.0f, 4.0f}, {batch_size, 2}).unwrap();
    const auto gpu_inputs = upload(cpu_inputs).unwrap();

    const auto prediction = model.forward(gpu_inputs.const_view()).unwrap();
    EXPECT_EQ(prediction.extents[0], batch_size);
    EXPECT_EQ(prediction.extents[1], 3u);

    auto &dense_a = std::get<DenseLayer>(model.layers[a.value].kind);
    auto &dense_b = std::get<DenseLayer>(model.layers[b.value].kind);
    const auto weights_a_before = download(dense_a.weights.value).unwrap();
    const auto weights_b_before = download(dense_b.weights.value).unwrap();

    const auto cpu_targets = HostTensor2f::zero({batch_size, 3}).unwrap();
    const auto gpu_targets = upload(cpu_targets).unwrap();
    auto loss_fn = MSELoss::with_extents({batch_size, 3}).unwrap();
    const auto loss_grad = loss_fn.backward(prediction, gpu_targets.const_view()).wait().unwrap();
    model.backward(loss_grad).unwrap();

    auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.1f}).unwrap();
    model.step(optimizer, 1).unwrap();

    // Both branches must have received a real gradient through the Add node's fan-out - if
    // backward's routing to preds[0]/preds[1] were swapped, missing, or zeroed, at least one of
    // these would be unchanged (or both, if routing silently dropped both).
    const auto weights_a_after = download(dense_a.weights.value).unwrap();
    const auto weights_b_after = download(dense_b.weights.value).unwrap();
    EXPECT_FALSE(are_close(weights_a_before, weights_a_after, 1e-8f));
    EXPECT_FALSE(are_close(weights_b_before, weights_b_after, 1e-8f));
}

UTEST(model, branching_concat_forward_and_backward)
{
    using namespace vika;

    // input -> denseA (-> 3) -\
    //                          concat -> sigmoid -> output (-> 5)
    // input -> denseB (-> 2) -/
    constexpr usize batch_size = 2;

    ComputationGraph graph{batch_size};
    auto x = graph.input({2});
    auto a = graph.dense(x, 3, 42).unwrap();
    auto b = graph.dense(x, 2, 43).unwrap();
    auto joined = graph.concat({a, b}).unwrap();
    auto out = graph.sigmoid(joined).unwrap();

    auto model = graph.compile(out).unwrap();

    const auto cpu_inputs = HostTensor2f::from({1.0f, 2.0f, 3.0f, 4.0f}, {batch_size, 2}).unwrap();
    const auto gpu_inputs = upload(cpu_inputs).unwrap();

    const auto prediction = model.forward(gpu_inputs.const_view()).unwrap();
    EXPECT_EQ(prediction.extents[0], batch_size);
    EXPECT_EQ(prediction.extents[1], 5u);

    auto &dense_a = std::get<DenseLayer>(model.layers[a.value].kind);
    auto &dense_b = std::get<DenseLayer>(model.layers[b.value].kind);
    const auto weights_a_before = download(dense_a.weights.value).unwrap();
    const auto weights_b_before = download(dense_b.weights.value).unwrap();

    const auto cpu_targets = HostTensor2f::zero({batch_size, 5}).unwrap();
    const auto gpu_targets = upload(cpu_targets).unwrap();
    auto loss_fn = MSELoss::with_extents({batch_size, 5}).unwrap();
    const auto loss_grad = loss_fn.backward(prediction, gpu_targets.const_view()).wait().unwrap();
    model.backward(loss_grad).unwrap();

    auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.1f}).unwrap();
    model.step(optimizer, 1).unwrap();

    // Both branches must have received their own (differently-shaped) split of the gradient
    // through Concat's backward - if the column-offset splitting were wrong, at least one of
    // these would be unchanged.
    const auto weights_a_after = download(dense_a.weights.value).unwrap();
    const auto weights_b_after = download(dense_b.weights.value).unwrap();
    EXPECT_FALSE(are_close(weights_a_before, weights_a_after, 1e-8f));
    EXPECT_FALSE(are_close(weights_b_before, weights_b_after, 1e-8f));
}
