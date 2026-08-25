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

// compile() fills layers/layer_inputs by NodeId, not by position in the topological order, so a
// node vector that isn't already topologically ordered still wires up correctly. The builders
// cannot produce one (they reject a NodeId that does not exist yet, so edges only ever run from
// lower to higher indices), but nodes is public and a graph loaded from disk need not be ordered
// that way - here node 0 consumes node 1. This used to mis-index every layer and then throw.
UTEST(model, compile_out_of_order_nodes)
{
    using namespace vika;

    constexpr usize batch_size = 2;

    ComputationGraph graph{batch_size};
    graph.nodes.push_back(Node{DenseSpec{3, 42}, Extents{batch_size, 3}, {NodeId{1}}});
    graph.nodes.push_back(Node{InputSpec{}, Extents{batch_size, 4}, {}});

    auto model = graph.compile(NodeId{0}).unwrap();

    EXPECT_TRUE(std::holds_alternative<DenseLayer>(model.layers[0].kind));
    EXPECT_TRUE(std::holds_alternative<InputLayer>(model.layers[1].kind));

    EXPECT_EQ(model.layer_inputs[0].size(), 1u);
    EXPECT_EQ(model.layer_inputs[0][0].value, 1u);
    EXPECT_EQ(model.layer_inputs[1].size(), 0u);

    EXPECT_EQ(model.execution_order.size(), 2u);
    EXPECT_EQ(model.execution_order[0].value, 1u);
    EXPECT_EQ(model.execution_order[1].value, 0u);

    const auto inputs = DeviceOwningTensorf::zero({batch_size, 4}).unwrap();
    const auto out = model.forward(inputs.const_view()).unwrap();

    EXPECT_EQ(out.rank(), 2u);
    EXPECT_EQ(out.extents[0], batch_size);
    EXPECT_EQ(out.extents[1], 3u);
}

// An optimizer carries one state entry per node, so one built from a different model does not
// describe this one. While that state was a hash map, the mismatch was invisible: every lookup
// missed, every layer was skipped, and step() returned Ok having trained nothing.
UTEST(model, step_rejects_an_optimizer_built_from_another_model)
{
    using namespace vika;

    constexpr usize batch_size = 4;

    const auto build = [](usize dense_layers) {
        ComputationGraph graph{batch_size};
        auto x = graph.input({2});
        for (usize i = 0; i < dense_layers; ++i)
        {
            x = graph.dense(x, 3, 42).unwrap();
        }
        return graph.compile(x).unwrap();
    };

    auto model = build(1);
    auto other = build(3);

    const auto inputs = DeviceOwningTensorf::zero({batch_size, 2}).unwrap();
    const auto loss_grad = DeviceOwningTensorf::zero({batch_size, 3}).unwrap();

    auto foreign = AdamOptimizer::from_model(other, {}).unwrap();
    auto own = AdamOptimizer::from_model(model, {}).unwrap();

    // Both passes have run, so the only thing left for step() to object to is the optimizer.
    ASSERT_TRUE(model.forward(inputs.const_view()).is_ok());
    ASSERT_TRUE(model.backward(loss_grad.const_view()).is_ok());

    EXPECT_TRUE(model.step(foreign).is_error());
    EXPECT_EQ(foreign.steps_taken, 0u);

    // Same model, same passes - so a failure here would mean the guard rejects too much.
    ASSERT_TRUE(model.step(own).is_ok());
    EXPECT_EQ(own.steps_taken, 1u);
}

UTEST(model, forward_matches_manual_xor)
{
    using namespace vika;

    constexpr usize batch_size = 4;
    constexpr u32 seed1 = 42;
    constexpr u32 seed2 = 43;

    const auto cpu_inputs = HostTensorf::from({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}, {batch_size, 2}).unwrap();
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

// Calling these out of order used to take the process down (step indexed off the end of an empty
// forward_jobs/backward_jobs) or silently succeed on uninitialised layer state (backward reads the
// `outputs`/`argmax` that only forward fills).
UTEST(model, out_of_order_passes_return_errors)
{
    using namespace vika;

    constexpr usize batch_size = 4;

    ComputationGraph graph{batch_size};
    auto x = graph.input({2});
    x = graph.dense(x, 3, 42).unwrap();
    auto model = graph.compile(x).unwrap();
    auto optimizer = AdamOptimizer::from_model(model, {}).unwrap();

    const auto inputs = DeviceOwningTensorf::zero({batch_size, 2}).unwrap();
    const auto loss_grad = DeviceOwningTensorf::zero({batch_size, 3}).unwrap();

    EXPECT_TRUE(model.step(optimizer).is_error());
    EXPECT_TRUE(model.backward(loss_grad.const_view()).is_error());
    EXPECT_EQ(optimizer.steps_taken, 0u);   // a rejected step must not consume a step count
    EXPECT_TRUE(model.accumulate_output_gradient(model.output_node).is_error());
    EXPECT_TRUE(model.forward_output(model.output_node).is_error());

    ASSERT_TRUE(model.forward(inputs.const_view()).is_ok());
    EXPECT_TRUE(model.forward_output(NodeId{99}).is_error());

    // Forward has run, backward has not.
    EXPECT_TRUE(model.step(optimizer).is_error());

    // In order, and repeatable. The optimizer owns the step count, so the first successful step
    // is t = 1 no matter what a caller's own loop variable does.
    for (usize t = 0; t < 3; ++t)
    {
        ASSERT_TRUE(model.forward(inputs.const_view()).is_ok());
        ASSERT_TRUE(model.backward(loss_grad.const_view()).is_ok());
        ASSERT_TRUE(model.step(optimizer).is_ok());
    }
    EXPECT_EQ(optimizer.steps_taken, 3u);
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

    EXPECT_EQ(out.rank(), 2u);
    EXPECT_EQ(out.extents[0], batch_size);
    EXPECT_EQ(out.extents[1], 1u);
}

// The gap that let a broken Flatten2DLayer ship: every *_smaller_batch test was per-layer, and
// each of those layers was already correct. Nothing ran a whole model below its capacity.
//
// Checking shapes would not have caught it either - what makes this worth having is comparing
// values: a batch-2 forward must produce exactly what the first two rows of a batch-4 forward
// produce, since view slicing is supposed to change how much is computed, not what.
UTEST(model, smaller_batch_matches_the_first_rows_of_a_full_batch)
{
    using namespace vika;

    constexpr usize capacity = 4;

    // conv -> maxpool -> flatten -> dense: the line_cnn shape, and the only path through a
    // flatten in the suite.
    ComputationGraph graph{capacity};
    auto x = graph.input({4, 4, 1});
    x = graph.conv2d(x, 3, 3, 2, 1, 0, 7).unwrap();
    x = graph.maxpool2d(x, 2, 2, 2).unwrap();
    x = graph.flatten(x).unwrap();
    x = graph.dense(x, 1, 8).unwrap();
    auto model = graph.compile(x).unwrap();

    auto cpu_inputs = HostTensorf::zero({capacity, 4, 4, 1}).unwrap();
    for (usize n = 0; n < capacity; ++n)
    {
        for (usize h = 0; h < 4; ++h)
        {
            for (usize w = 0; w < 4; ++w)
            {
                cpu_inputs(n, h, w, 0) = static_cast<f32>(n * 100 + h * 10 + w);
            }
        }
    }
    auto inputs = upload(cpu_inputs).unwrap();

    const auto full_forward = model.forward(inputs.const_view());
    ASSERT_TRUE(full_forward.is_ok());
    const auto full = download(full_forward.unwrap()).unwrap();
    ASSERT_EQ(full.extent(0), capacity);

    const auto sliced = inputs.view().first_n(2).unwrap().const_view();
    const auto half_forward = model.forward(sliced);
    ASSERT_TRUE(half_forward.is_ok());
    const auto half_view = half_forward.unwrap();
    EXPECT_EQ(half_view.extents[0], 2u);
    const auto half = download(half_view).unwrap();

    ASSERT_EQ(half.extent(0), 2u);
    EXPECT_NEAR(half(0, 0), full(0, 0), 1e-6f);
    EXPECT_NEAR(half(1, 0), full(1, 0), 1e-6f);

    // ... and the rest of the step runs at that batch too, which is the half of finding 1 that
    // lived in backward().
    auto loss_fn = MSELoss::with_extents({capacity, 1}).unwrap();
    auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.01f}).unwrap();
    auto targets = DeviceOwningTensorf::zero({capacity, 1}).unwrap();
    const auto sliced_targets = targets.view().first_n(2).unwrap().const_view();

    ASSERT_TRUE(train_step(model, loss_fn, sliced, sliced_targets, optimizer).is_ok());
    EXPECT_EQ(optimizer.steps_taken, 1u);
}

UTEST(model, xor_trains_to_convergence)
{
    using namespace vika;

    constexpr usize batch_size = 4;

    const auto cpu_inputs = HostTensorf::from({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}, {batch_size, 2}).unwrap();
    const auto cpu_targets = HostTensorf::from({0.0f, 1.0f, 1.0f, 0.0f}, {batch_size, 1}).unwrap();
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
        model.step(optimizer).unwrap();
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

    // input -> denseA -.
    //                    add -> sigmoid -> output
    // input -> denseB -'
    constexpr usize batch_size = 2;

    ComputationGraph graph{batch_size};
    auto x = graph.input({2});
    auto a = graph.dense(x, 3, 42).unwrap();
    auto b = graph.dense(x, 3, 43).unwrap();
    auto sum = graph.add({a, b}).unwrap();
    auto out = graph.sigmoid(sum).unwrap();

    auto model = graph.compile(out).unwrap();

    const auto cpu_inputs = HostTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {batch_size, 2}).unwrap();
    const auto gpu_inputs = upload(cpu_inputs).unwrap();

    const auto prediction = model.forward(gpu_inputs.const_view()).unwrap();
    EXPECT_EQ(prediction.extents[0], batch_size);
    EXPECT_EQ(prediction.extents[1], 3u);

    auto &dense_a = std::get<DenseLayer>(model.layers[a.value].kind);
    auto &dense_b = std::get<DenseLayer>(model.layers[b.value].kind);
    const auto weights_a_before = download(dense_a.weights.value).unwrap();
    const auto weights_b_before = download(dense_b.weights.value).unwrap();

    const auto cpu_targets = HostTensorf::zero({batch_size, 3}).unwrap();
    const auto gpu_targets = upload(cpu_targets).unwrap();
    auto loss_fn = MSELoss::with_extents({batch_size, 3}).unwrap();
    const auto loss_grad = loss_fn.backward(prediction, gpu_targets.const_view()).wait().unwrap();
    model.backward(loss_grad).unwrap();

    auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.1f}).unwrap();
    model.step(optimizer).unwrap();

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

    // input -> denseA (-> 3) -.
    //                          concat -> sigmoid -> output (-> 5)
    // input -> denseB (-> 2) -'
    constexpr usize batch_size = 2;

    ComputationGraph graph{batch_size};
    auto x = graph.input({2});
    auto a = graph.dense(x, 3, 42).unwrap();
    auto b = graph.dense(x, 2, 43).unwrap();
    auto joined = graph.concat({a, b}).unwrap();
    auto out = graph.sigmoid(joined).unwrap();

    auto model = graph.compile(out).unwrap();

    const auto cpu_inputs = HostTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {batch_size, 2}).unwrap();
    const auto gpu_inputs = upload(cpu_inputs).unwrap();

    const auto prediction = model.forward(gpu_inputs.const_view()).unwrap();
    EXPECT_EQ(prediction.extents[0], batch_size);
    EXPECT_EQ(prediction.extents[1], 5u);

    auto &dense_a = std::get<DenseLayer>(model.layers[a.value].kind);
    auto &dense_b = std::get<DenseLayer>(model.layers[b.value].kind);
    const auto weights_a_before = download(dense_a.weights.value).unwrap();
    const auto weights_b_before = download(dense_b.weights.value).unwrap();

    const auto cpu_targets = HostTensorf::zero({batch_size, 5}).unwrap();
    const auto gpu_targets = upload(cpu_targets).unwrap();
    auto loss_fn = MSELoss::with_extents({batch_size, 5}).unwrap();
    const auto loss_grad = loss_fn.backward(prediction, gpu_targets.const_view()).wait().unwrap();
    model.backward(loss_grad).unwrap();

    auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.1f}).unwrap();
    model.step(optimizer).unwrap();

    // Both branches must have received their own (differently-shaped) split of the gradient
    // through Concat's backward - if the column-offset splitting were wrong, at least one of
    // these would be unchanged.
    const auto weights_a_after = download(dense_a.weights.value).unwrap();
    const auto weights_b_after = download(dense_b.weights.value).unwrap();
    EXPECT_FALSE(are_close(weights_a_before, weights_a_after, 1e-8f));
    EXPECT_FALSE(are_close(weights_b_before, weights_b_after, 1e-8f));
}

UTEST(model, fan_in_accumulation_across_multiple_backward_calls)
{
    using namespace vika;

    // input -> trunk -.
    //                   add -> sigmoid -> output
    //          trunk -'
    //
    // trunk's output has two consumers (branch_a, branch_b), and trunk itself has its own
    // predecessor (input) - unlike the branching tests above, where the fan-out point is the
    // input node itself. Model::backward skips accumulation entirely for any node with
    // preds.empty() (nothing further upstream to propagate to), and the input node always has
    // empty preds - so this is the only test that actually exercises
    // Model::accumulate_output_gradient's jobs.size() > 1 path, and the only one that calls
    // backward() more than once, to exercise reusing the same pre-allocated accumulation buffer
    // across repeated calls rather than just a single one.
    constexpr usize batch_size = 4;

    ComputationGraph graph{batch_size};
    auto x = graph.input({2});
    auto trunk = graph.dense(x, 4, 42).unwrap();
    auto branch_a = graph.dense(trunk, 3, 43).unwrap();
    auto branch_b = graph.dense(trunk, 3, 44).unwrap();
    auto sum = graph.add({branch_a, branch_b}).unwrap();
    auto out = graph.sigmoid(sum).unwrap();

    auto model = graph.compile(out).unwrap();

    const auto cpu_inputs =
        HostTensorf::from({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}, {batch_size, 2}).unwrap();
    const auto cpu_targets = HostTensorf::zero({batch_size, 3}).unwrap();
    const auto gpu_inputs = upload(cpu_inputs).unwrap();
    const auto gpu_targets = upload(cpu_targets).unwrap();

    auto loss_fn = MSELoss::with_extents({batch_size, 3}).unwrap();
    auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.1f}).unwrap();

    auto &dense_trunk = std::get<DenseLayer>(model.layers[trunk.value].kind);
    const auto trunk_weights_before = download(dense_trunk.weights.value).unwrap();

    f32 initial_loss = 0.0f;
    f32 final_loss = 0.0f;
    for (usize step = 1; step <= 50; ++step)
    {
        const auto prediction = model.forward(gpu_inputs.const_view()).unwrap();
        const auto loss_view = loss_fn.forward(prediction, gpu_targets.const_view()).wait().unwrap();
        const auto loss_grad = loss_fn.backward(prediction, gpu_targets.const_view()).wait().unwrap();
        model.backward(loss_grad).unwrap();
        model.step(optimizer).unwrap();

        const auto loss_cpu = download(loss_view).unwrap();
        if (step == 1)
        {
            initial_loss = loss_cpu[0];
        }
        final_loss = loss_cpu[0];
    }

    // trunk only ever receives a gradient by summing branch_a's and branch_b's contributions in
    // accumulate_output_gradient - if that summing were wrong, or the reused accumulation buffer
    // got corrupted across repeated calls, trunk's weights would either not move or move
    // incorrectly, and loss wouldn't reliably decrease over 50 steps of reusing the same buffer.
    const auto trunk_weights_after = download(dense_trunk.weights.value).unwrap();
    EXPECT_FALSE(are_close(trunk_weights_before, trunk_weights_after, 1e-8f));
    EXPECT_TRUE(final_loss < initial_loss);
}
