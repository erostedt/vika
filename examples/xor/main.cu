#include <cstdio>

#define VIKA_IMPLEMENTATION
#include "vika.cuh"

int main()
{
    using namespace vika;

    constexpr usize batch_size = 4;
    constexpr usize hidden_size = 8;
    constexpr usize epochs = 10000;

    // XOR dataset
    const auto cpu_inputs = HostTensor2f::from(
        {
            0.0f,
            0.0f,
            0.0f,
            1.0f,
            1.0f,
            0.0f,
            1.0f,
            1.0f,
        },
        {batch_size, 2});

    const auto cpu_targets = HostTensor2f::from(
        {
            0.0f,
            1.0f,
            1.0f,
            0.0f,
        },
        {batch_size, 1});

    const auto inputs = upload(cpu_inputs).unwrap();
    const auto targets = upload(cpu_targets).unwrap();

    // Layers
    auto dense1 = DenseLayer::randomized(batch_size, 2, hidden_size, 42u).unwrap();
    auto sigmoid1 = SigmoidLayer::with_extents({batch_size, hidden_size}).unwrap();
    auto dense2 = DenseLayer::randomized(batch_size, hidden_size, 1, 43u).unwrap();
    auto sigmoid2 = SigmoidLayer::with_extents({batch_size, 1}).unwrap();
    auto loss_fn = MSELoss<2>::with_extents({batch_size, 1}).unwrap();

    const AdamParameters adam{
        .learning_rate = 0.01f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .epsilon = 1e-8f,
    };

    for (usize step = 1; step <= epochs; ++step)
    {
        // Forward
        const auto out1 = dense1.forward(inputs.const_view()).wait().unwrap();
        const auto act1 = sigmoid1.forward(out1).wait().unwrap();
        const auto out2 = dense2.forward(act1).wait().unwrap();
        const auto act2 = sigmoid2.forward(out2).wait().unwrap();

        // Loss
        const auto loss_view = loss_fn.forward(act2, targets.const_view()).wait().unwrap();

        // Backward
        const auto d_act2 = loss_fn.backward(act2, targets.const_view()).wait().unwrap();
        const auto d_out2 = sigmoid2.backward(d_act2).wait().unwrap();
        const auto d_act1 = dense2.backward(d_out2).wait().unwrap();
        const auto [dw2, db2] = dense2.weight_gradients(act1, d_out2).wait().unwrap();
        const auto d_out1 = sigmoid1.backward(d_act1).wait().unwrap();
        const auto [dw1, db1] = dense1.weight_gradients(inputs.const_view(), d_out1).wait().unwrap();

        // Update
        dense2.update(dw2, db2, adam, step).wait().unwrap();
        dense1.update(dw1, db1, adam, step).wait().unwrap();

        if (step % 1000 == 0)
        {
            const auto loss_cpu = download(loss_view).unwrap();
            printf("step %5zu | loss: %.6f\n", step, loss_cpu[0]);
        }
    }

    // Final predictions
    const auto out1 = dense1.forward(inputs.const_view()).wait().unwrap();
    const auto act1 = sigmoid1.forward(out1).wait().unwrap();
    const auto out2 = dense2.forward(act1).wait().unwrap();
    const auto act2 = sigmoid2.forward(out2).wait().unwrap();
    const auto preds = download(act2).unwrap();

    printf("\nFinal predictions:\n");
    printf("  [0, 0] -> %.4f (expected 0)\n", preds(0, 0));
    printf("  [0, 1] -> %.4f (expected 1)\n", preds(1, 0));
    printf("  [1, 0] -> %.4f (expected 1)\n", preds(2, 0));
    printf("  [1, 1] -> %.4f (expected 0)\n", preds(3, 0));

    return 0;
}
