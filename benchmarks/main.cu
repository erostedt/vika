// Deliberately not wired into CTest: this is a tool you run on purpose, not a check.
//
//     cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release && cmake --build build-release -j
//     ./build-release/benchmarks/run_benchmarks
//
// Each row times what a caller actually pays: the operation *and* the wait() that follows it,
// because this library synchronises inside every call, and that cost is the subject of several
// open findings. It answers "did this change help, and by how much" - when the question is which
// kernel dominates a row, reach for ncu or nsys rather than growing a profiler in here.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#define VIKA_IMPLEMENTATION
#include "vika.cuh"

using namespace vika;

namespace
{

// Median rather than mean: one descheduled iteration should not move the number. min is reported
// alongside as the cleanest view of the work itself, with scheduling noise squeezed out.
template <typename Body>
auto measure(const char *name, usize iterations, Body &&body) -> void
{
    const usize warmup = iterations / 10 + 3;
    for (usize i = 0; i < warmup; ++i)
    {
        body();
    }

    std::vector<double> samples;
    samples.reserve(iterations);
    for (usize i = 0; i < iterations; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        body();
        const auto end = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    std::sort(std::begin(samples), std::end(samples));
    printf("  %-38s %7zu %12.4f %12.4f\n", name, iterations, samples[samples.size() / 2], samples.front());
}

auto header() -> void
{
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    printf("device: %s\n", properties.name);

#ifdef NDEBUG
    printf("build:  release\n");
#else
    printf("build:  DEBUG - host-side cost is inflated (roughly a third, on the xor example),\n");
    printf("        so treat the absolute numbers as indicative and compare like with like.\n");
#endif
    printf("\n  %-38s %7s %12s %12s\n", "benchmark", "iters", "median ms", "min ms");
    printf("  %-38s %7s %12s %12s\n", "--------------------------------------", "-------", "------------",
           "------------");
}

// One full training step through the line_cnn architecture: the end-to-end number every per-op
// row below should be read against.
auto benchmark_train_step() -> void
{
    constexpr usize batch = 16;

    ComputationGraph graph{batch};
    auto x = graph.input({8, 8, 1});
    x = graph.conv2d(x, 3, 3, 8, 1, 0, 1).unwrap();
    x = graph.maxpool2d(x, 2, 2, 2).unwrap();
    x = graph.flatten(x).unwrap();
    x = graph.dense(x, 16, 2).unwrap();
    x = graph.sigmoid(x).unwrap();
    x = graph.dense(x, 1, 3).unwrap();
    x = graph.sigmoid(x).unwrap();

    auto model = graph.compile(x).unwrap();
    auto loss_fn = MSELoss::with_extents({batch, 1}).unwrap();
    auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.01f}).unwrap();

    const auto inputs = DeviceOwningTensorf::zero({batch, 8, 8, 1}).unwrap();
    const auto targets = DeviceOwningTensorf::zero({batch, 1}).unwrap();

    measure("train_step (line_cnn, batch 16)", 200, [&] {
        train_step(model, loss_fn, inputs.const_view(), targets.const_view(), optimizer).unwrap();
    });
}

auto benchmark_dense() -> void
{
    constexpr usize batch = 256;
    constexpr usize features = 1024;
    constexpr usize neurons = 1024;

    auto layer = DenseLayer::randomized(batch, features, neurons, 1).unwrap();
    const auto inputs = DeviceOwningTensorf::zero({batch, features}).unwrap();
    const auto upstream = DeviceOwningTensorf::zero({batch, neurons}).unwrap();

    std::vector<AdamState> states;
    for (const auto &parameter : layer.parameters())
    {
        states.push_back(AdamState::create(parameter.value.extents).unwrap());
    }

    measure("dense forward (256x1024 -> 1024)", 200,
            [&] { layer.forward({inputs.const_view()}).wait().unwrap(); });
    measure("dense backward", 200, [&] { layer.backward(upstream.const_view())[0].wait().unwrap(); });
    measure("dense weight_gradients", 200,
            [&] { layer.weight_gradients(inputs.const_view(), upstream.const_view()).wait().unwrap(); });
    measure("dense adam update", 200, [&] {
        for (auto &result : wait_on_all(layer.update(states, AdamParameters{}, 1)))
        {
            result.unwrap();
        }
    });
}

auto benchmark_conv() -> void
{
    constexpr usize batch = 16;
    constexpr usize size = 32;
    constexpr usize channels_in = 32;
    constexpr usize channels_out = 64;

    auto layer = Conv2DLayer::randomized(batch, size, size, 3, 3, channels_in, channels_out, 1, 0, 1).unwrap();
    const auto inputs = DeviceOwningTensorf::zero({batch, size, size, channels_in}).unwrap();
    const auto upstream = DeviceOwningTensorf::zero({batch, size - 2, size - 2, channels_out}).unwrap();

    measure("conv2d forward (16x32x32x32 -> 64)", 100,
            [&] { layer.forward({inputs.const_view()}).wait().unwrap(); });
    measure("conv2d backward", 100, [&] { layer.backward(upstream.const_view())[0].wait().unwrap(); });
    measure("conv2d weight_gradients", 100,
            [&] { layer.weight_gradients(inputs.const_view(), upstream.const_view()).wait().unwrap(); });
}

auto benchmark_pointwise() -> void
{
    constexpr usize batch = 256;
    constexpr usize features = 1024;

    auto sigmoid = SigmoidLayer::with_extents({batch, features}).unwrap();
    auto pool = MaxPool2DLayer::with_extents(16, 32, 32, 32, 2, 2, 2).unwrap();
    auto loss = MSELoss::with_extents({batch, features}).unwrap();

    const auto flat = DeviceOwningTensorf::zero({batch, features}).unwrap();
    const auto images = DeviceOwningTensorf::zero({16, 32, 32, 32}).unwrap();
    const auto pooled = DeviceOwningTensorf::zero({16, 16, 16, 32}).unwrap();

    measure("sigmoid forward (256x1024)", 500, [&] { sigmoid.forward({flat.const_view()}).wait().unwrap(); });
    measure("sigmoid backward", 500, [&] { sigmoid.backward(flat.const_view())[0].wait().unwrap(); });
    measure("maxpool2d forward (16x32x32x32)", 500, [&] { pool.forward({images.const_view()}).wait().unwrap(); });
    measure("maxpool2d backward", 500, [&] { pool.backward(pooled.const_view())[0].wait().unwrap(); });
    measure("mse forward (262144 elements)", 500,
            [&] { loss.forward(flat.const_view(), flat.const_view()).wait().unwrap(); });
    measure("mse backward", 500, [&] { loss.backward(flat.const_view(), flat.const_view()).wait().unwrap(); });
}

} // namespace

auto main() -> int
{
    header();
    benchmark_train_step();
    benchmark_dense();
    benchmark_conv();
    benchmark_pointwise();
    return 0;
}
