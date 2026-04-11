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
    constexpr usize N_HORIZ = IMG_H; // one sample per row position
    constexpr usize N_VERT = IMG_W;  // one sample per column position
    constexpr usize batch_size = N_HORIZ + N_VERT;

    auto cpu_inputs = HostTensor4f::zero({batch_size, IMG_H, IMG_W, 1});
    auto cpu_targets = HostTensor2f::zero({batch_size, 1});

    // Horizontal lines: sample r has a bright row at r
    for (usize r = 0; r < N_HORIZ; ++r)
    {
        for (usize w = 0; w < IMG_W; ++w)
        {
            cpu_inputs(r, r, w, 0) = 1.0f;
        }
        cpu_targets(r, 0) = 0.0f;
    }

    // Vertical lines: sample N_HORIZ+c has a bright column at c
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
    //   Flatten2D:                                   [N, 3, 3, 8] -> [N, 72]
    //   Dense(72 -> 16)
    //   Sigmoid
    //   Dense(16 -> 1)
    //   Sigmoid
    constexpr usize CONV_FILTERS = 8;
    constexpr usize CONV_OUT_H = IMG_H - 3 + 1;                         // 6
    constexpr usize CONV_OUT_W = IMG_W - 3 + 1;                         // 6
    constexpr usize POOL_OUT_H = (CONV_OUT_H - 2) / 2 + 1;              // 3
    constexpr usize POOL_OUT_W = (CONV_OUT_W - 2) / 2 + 1;              // 3
    constexpr usize FLAT_SIZE = POOL_OUT_H * POOL_OUT_W * CONV_FILTERS; // 72
    constexpr usize HIDDEN = 16;

    auto conv = Conv2DLayer::randomized(batch_size, IMG_H, IMG_W, 3, 3, 1, CONV_FILTERS, 1, 0, 1u).unwrap();
    auto pool = MaxPool2DLayer::with_extents(batch_size, CONV_OUT_H, CONV_OUT_W, CONV_FILTERS, 2, 2, 2).unwrap();
    auto flatten = Flatten2DLayer::with_extents({batch_size, POOL_OUT_H, POOL_OUT_W, CONV_FILTERS});
    auto dense1 = DenseLayer::randomized(batch_size, FLAT_SIZE, HIDDEN, 2u).unwrap();
    auto sigmoid1 = SigmoidLayer::with_extents({batch_size, HIDDEN}).unwrap();
    auto dense2 = DenseLayer::randomized(batch_size, HIDDEN, 1, 3u).unwrap();
    auto sigmoid2 = SigmoidLayer::with_extents({batch_size, 1}).unwrap();
    auto loss_fn = MSELoss::with_extents({batch_size, 1}).unwrap();

    const AdamParameters adam{
        .learning_rate = 0.01f,
        .beta1 = 0.9f,
        .beta2 = 0.999f,
        .epsilon = 1e-8f,
    };

    constexpr usize epochs = 5000;

    for (usize step = 1; step <= epochs; ++step)
    {
        // Forward
        const auto conv_out = conv.forward(inputs.const_view()).wait().unwrap();
        const auto pool_out = pool.forward(conv_out).wait().unwrap();
        const auto flat_out = flatten.forward(pool_out);
        const auto d1_out = dense1.forward(flat_out).wait().unwrap();
        const auto act1 = sigmoid1.forward(d1_out).wait().unwrap();
        const auto d2_out = dense2.forward(act1).wait().unwrap();
        const auto preds = sigmoid2.forward(d2_out).wait().unwrap();

        // Loss
        const auto loss_view = loss_fn.forward(preds, targets.const_view()).wait().unwrap();

        // Backward
        const auto d_preds = loss_fn.backward(preds, targets.const_view()).wait().unwrap();
        const auto d_d2_out = sigmoid2.backward(d_preds).wait().unwrap();
        const auto d_act1 = dense2.backward(d_d2_out).wait().unwrap();
        const auto [dw2, db2] = dense2.weight_gradients(act1, d_d2_out).wait().unwrap();
        const auto d_d1_out = sigmoid1.backward(d_act1).wait().unwrap();
        const auto d_flat = dense1.backward(d_d1_out).wait().unwrap();
        const auto [dw1, db1] = dense1.weight_gradients(flat_out, d_d1_out).wait().unwrap();
        const auto d_pool = flatten.backward(d_flat);
        const auto d_conv = pool.backward(d_pool).wait().unwrap();
        const auto [df, dbias] = conv.weight_gradients(inputs.const_view(), d_conv).wait().unwrap();

        // Update
        dense2.update(dw2, db2, adam, step).wait().unwrap();
        dense1.update(dw1, db1, adam, step).wait().unwrap();
        conv.update(df, dbias, adam, step).wait().unwrap();

        if (step % 500 == 0)
        {
            const auto loss_cpu = download(loss_view).unwrap();
            printf("step %5zu | loss: %.6f\n", step, loss_cpu[0]);
        }
    }

    // Final predictions
    const auto conv_out = conv.forward(inputs.const_view()).wait().unwrap();
    const auto pool_out = pool.forward(conv_out).wait().unwrap();
    const auto flat_out = flatten.forward(pool_out);
    const auto d1_out = dense1.forward(flat_out).wait().unwrap();
    const auto act1 = sigmoid1.forward(d1_out).wait().unwrap();
    const auto d2_out = dense2.forward(act1).wait().unwrap();
    const auto preds = sigmoid2.forward(d2_out).wait().unwrap();
    const auto cpu_preds = download(preds).unwrap();

    printf("\nFinal predictions (0=horizontal, 1=vertical):\n");
    for (usize r = 0; r < N_HORIZ; ++r)
    {
        printf("  horiz line at row %zu -> %.4f (expected 0.0)\n", r, cpu_preds(r, 0));
    }
    for (usize c = 0; c < N_VERT; ++c)
    {
        printf("  vert  line at col %zu -> %.4f (expected 1.0)\n", c, cpu_preds(N_HORIZ + c, 0));
    }

    return 0;
}
