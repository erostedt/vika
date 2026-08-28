#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(add, layer_forward)
{
    using namespace vika;
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({10.0f, 20.0f, 30.0f, 40.0f}, {2, 2}).unwrap();

    auto layer = AddLayer::with_extents({2, 2}, 2).unwrap();
    const auto out = layer.forward({a.const_view(), b.const_view()}).wait().unwrap();
    const auto out_cpu = download(out).unwrap();

    const std::vector<f32> expected = {11.0f, 22.0f, 33.0f, 44.0f};
    ASSERT_TRUE(are_close(out_cpu, expected, 1e-5f));
}

UTEST(add, layer_forward_n_ary)
{
    using namespace vika;
    // 4 inputs, not 2 - exercises the accumulate loop beyond the first/last special cases.
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({10.0f, 20.0f, 30.0f, 40.0f}, {2, 2}).unwrap();
    const auto c = DeviceOwningTensorf::from({100.0f, 200.0f, 300.0f, 400.0f}, {2, 2}).unwrap();
    const auto d = DeviceOwningTensorf::from({1000.0f, 2000.0f, 3000.0f, 4000.0f}, {2, 2}).unwrap();

    auto layer = AddLayer::with_extents({2, 2}, 4).unwrap();
    const auto out =
        layer.forward({a.const_view(), b.const_view(), c.const_view(), d.const_view()}).wait().unwrap();
    const auto out_cpu = download(out).unwrap();

    const std::vector<f32> expected = {1111.0f, 2222.0f, 3333.0f, 4444.0f};
    ASSERT_TRUE(are_close(out_cpu, expected, 1e-5f));
}

UTEST(add, layer_backward_n_ary)
{
    using namespace vika;
    const auto upstream = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();

    auto layer = AddLayer::with_extents({2, 2}, 4).unwrap();
    auto jobs = layer.backward(upstream.const_view());

    ASSERT_TRUE(jobs.size() == 4);
    const std::vector<f32> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    for (auto &job : jobs)
    {
        const auto d_x = download(job.wait().unwrap()).unwrap();
        ASSERT_TRUE(are_close(d_x, expected, 1e-5f));
    }
}

UTEST(add, layer_forward_smaller_batch)
{
    using namespace vika;
    // Same values as layer_forward's first row, but the layer has spare capacity (2) while the
    // actual batch (1) is smaller.
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f}, {1, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({10.0f, 20.0f}, {1, 2}).unwrap();

    auto layer = AddLayer::with_extents({2, 2}, 2).unwrap();
    const auto out = layer.forward({a.const_view(), b.const_view()}).wait().unwrap();
    const auto out_cpu = download(out).unwrap();

    EXPECT_EQ(out.extents[0], 1u);
    const std::vector<f32> expected = {11.0f, 22.0f};
    ASSERT_TRUE(are_close(out_cpu, expected, 1e-5f));
}

UTEST(add, layer_forward_batch_exceeds_capacity)
{
    using namespace vika;
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({10.0f, 20.0f, 30.0f, 40.0f}, {2, 2}).unwrap();

    // Layer only has capacity for 1 sample, but the inputs have batch 2.
    auto layer = AddLayer::with_extents({1, 2}, 2).unwrap();
    ASSERT_TRUE(failed_with(layer.forward({a.const_view(), b.const_view()}), ErrorKind::Shape, "rows but tensor only has"));
}

UTEST(add, layer_forward_wrong_input_count)
{
    using namespace vika;
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();

    auto layer = AddLayer::with_extents({2, 2}, 2).unwrap();
    ASSERT_TRUE(failed_with(layer.forward({a.const_view()}), ErrorKind::Shape, "expects exactly 2 input(s), got 1"));
    ASSERT_TRUE(failed_with(layer.forward({a.const_view(), a.const_view(), a.const_view()}), ErrorKind::Shape, "expects exactly 2 input(s), got 3"));
}

UTEST(add, layer_forward_shape_mismatch)
{
    using namespace vika;
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}).unwrap();

    auto layer = AddLayer::with_extents({2, 2}, 2).unwrap();
    ASSERT_TRUE(failed_with(layer.forward({a.const_view(), b.const_view()}), ErrorKind::Shape, "forward: input 1 is"));
}

UTEST(add, layer_backward)
{
    using namespace vika;
    const auto upstream = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();

    auto layer = AddLayer::with_extents({2, 2}, 2).unwrap();
    auto jobs = layer.backward(upstream.const_view());

    ASSERT_TRUE(jobs.size() == 2);
    const auto d_a = download(jobs[0].wait().unwrap()).unwrap();
    const auto d_b = download(jobs[1].wait().unwrap()).unwrap();

    // d/da(a+b) = d/db(a+b) = 1: the same upstream gradient flows unchanged to both inputs.
    const std::vector<f32> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_TRUE(are_close(d_a, expected, 1e-5f));
    ASSERT_TRUE(are_close(d_b, expected, 1e-5f));
}
