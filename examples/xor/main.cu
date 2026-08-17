#include <cstdio>

#define VIKA_IMPLEMENTATION
#include "vika.cuh"

int main()
{
    using namespace vika;

    constexpr usize batch_size = 4;
    constexpr usize epochs = 10000;

    const auto cpu_inputs =
        HostTensor2f::from({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f}, {batch_size, 2}).unwrap();
    const auto cpu_targets = HostTensor2f::from({0.0f, 1.0f, 1.0f, 0.0f}, {batch_size, 1}).unwrap();
    const auto inputs = upload(cpu_inputs).unwrap();
    const auto targets = upload(cpu_targets).unwrap();

    ComputationGraph graph{batch_size};
    auto x = graph.input({2});
    x = graph.dense(x, 8, 42u).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 43u).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();
    auto loss_fn = MSELoss::with_extents({batch_size, 1}).unwrap();

    auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.01f}).unwrap();

    for (usize t = 1; t <= epochs; ++t)
    {
        const auto loss = train_step(model, loss_fn, inputs.const_view(), targets.const_view(), optimizer, t).unwrap();

        if (t % 1000 == 0)
        {
            printf("step %5zu | loss: %.6f\n", t, loss);
        }
    }

    const auto out = model.forward(inputs.const_view()).unwrap();
    const auto preds = download(out).unwrap();

    printf("\nFinal predictions:\n");
    printf("  [0, 0] -> %.4f (expected 0)\n", preds(0, 0));
    printf("  [0, 1] -> %.4f (expected 1)\n", preds(1, 0));
    printf("  [1, 0] -> %.4f (expected 1)\n", preds(2, 0));
    printf("  [1, 1] -> %.4f (expected 0)\n", preds(3, 0));

    return 0;
}
