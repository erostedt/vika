#include <cstdio>

#define VIKA_IMPLEMENTATION
#include "vika.cuh"

int main()
{
    using namespace vika;

    // 8x8 single-channel images
    // 8 samples with a horizontal line (label = 0.0)
    // 8 samples with a vertical line   (label = 1.0)
    constexpr usize IMG_H = 8;
    constexpr usize IMG_W = 8;
    constexpr usize N_HORIZ = IMG_H;
    constexpr usize N_VERT = IMG_W;
    constexpr usize batch_size = N_HORIZ + N_VERT;

    auto cpu_inputs = HostTensor4f::zero({batch_size, IMG_H, IMG_W, 1}).unwrap();
    auto cpu_targets = HostTensor2f::zero({batch_size, 1}).unwrap();

    for (usize r = 0; r < N_HORIZ; ++r)
    {
        for (usize w = 0; w < IMG_W; ++w)
        {
            cpu_inputs(r, r, w, 0) = 1.0f;
        }
        cpu_targets(r, 0) = 0.0f;
    }

    for (usize c = 0; c < N_VERT; ++c)
    {
        for (usize h = 0; h < IMG_H; ++h)
        {
            cpu_inputs(N_HORIZ + c, h, c, 0) = 1.0f;
        }
        cpu_targets(N_HORIZ + c, 0) = 1.0f;
    }

    const auto inputs = upload(cpu_inputs).unwrap();
    const auto targets = upload(cpu_targets).unwrap();

    // Architecture:
    //   Conv2D(3x3, 8 filters, stride=1, padding=0): [N, 8, 8, 1] -> [N, 6, 6, 8]
    //   MaxPool2D(2x2, stride=2):                    [N, 6, 6, 8] -> [N, 3, 3, 8]
    //   Flatten:                                     [N, 3, 3, 8] -> [N, 72]
    //   Dense(72 -> 16) -> Sigmoid -> Dense(16 -> 1) -> Sigmoid
    ComputationGraph graph{batch_size};
    auto x = graph.input({IMG_H, IMG_W, 1});
    x = graph.conv2d(x, 3, 3, 8, 1, 0, 1u).unwrap();
    x = graph.maxpool2d(x, 2, 2, 2).unwrap();
    x = graph.flatten(x).unwrap();
    x = graph.dense(x, 16, 2u).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 3u).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();
    auto loss_fn = MSELoss::with_extents({batch_size, 1}).unwrap();

    auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.01f}).unwrap();

    constexpr usize epochs = 5000;

    for (usize t = 1; t <= epochs; ++t)
    {
        const auto out = model.forward(inputs.const_view()).unwrap();
        const auto loss_view = loss_fn.forward(out, targets.const_view()).wait().unwrap();
        const auto loss_grad = loss_fn.backward(out, targets.const_view()).wait().unwrap();
        model.backward(loss_grad).unwrap();
        model.step(optimizer, t).unwrap();

        if (t % 500 == 0)
        {
            const auto loss_cpu = download(loss_view).unwrap();
            printf("step %5zu | loss: %.6f\n", t, loss_cpu[0]);
        }
    }

    const auto out = model.forward(inputs.const_view()).unwrap();
    const auto preds = download(out).unwrap();

    printf("\nFinal predictions (0=horizontal, 1=vertical):\n");
    for (usize r = 0; r < N_HORIZ; ++r)
    {
        printf("  horiz line at row %zu -> %.4f (expected 0.0)\n", r, preds(r, 0));
    }
    for (usize c = 0; c < N_VERT; ++c)
    {
        printf("  vert  line at col %zu -> %.4f (expected 1.0)\n", c, preds(N_HORIZ + c, 0));
    }

    return 0;
}
