#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(concat, layer_forward)
{
    using namespace vika;
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3}).unwrap();

    auto layer = ConcatLayer::with_extents({{2, 2}, {2, 3}}).unwrap();
    const auto out = layer.forward({a.const_view(), b.const_view()}).wait().unwrap();
    const auto out_cpu = download(out).unwrap();

    EXPECT_EQ(out.extents[0], 2u);
    EXPECT_EQ(out.extents[1], 5u);
    const std::vector<f32> expected = {1.0f, 2.0f, 10.0f, 20.0f, 30.0f, 3.0f, 4.0f, 40.0f, 50.0f, 60.0f};
    ASSERT_TRUE(are_close(out_cpu, expected, 1e-5f));
}

UTEST(concat, layer_forward_n_ary)
{
    using namespace vika;
    // 3 inputs, not 2 - exercises the accumulate-offset loop beyond the first/last special cases.
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f}, {1, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({10.0f}, {1, 1}).unwrap();
    const auto c = DeviceOwningTensorf::from({100.0f, 200.0f, 300.0f}, {1, 3}).unwrap();

    auto layer = ConcatLayer::with_extents({{1, 2}, {1, 1}, {1, 3}}).unwrap();
    const auto out = layer.forward({a.const_view(), b.const_view(), c.const_view()}).wait().unwrap();
    const auto out_cpu = download(out).unwrap();

    EXPECT_EQ(out.extents[1], 6u);
    const std::vector<f32> expected = {1.0f, 2.0f, 10.0f, 100.0f, 200.0f, 300.0f};
    ASSERT_TRUE(are_close(out_cpu, expected, 1e-5f));
}

UTEST(concat, layer_forward_smaller_batch)
{
    using namespace vika;
    // Layer built with capacity 2, but the actual batch (1) is smaller.
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f}, {1, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({10.0f, 20.0f, 30.0f}, {1, 3}).unwrap();

    auto layer = ConcatLayer::with_extents({{2, 2}, {2, 3}}).unwrap();
    const auto out = layer.forward({a.const_view(), b.const_view()}).wait().unwrap();
    const auto out_cpu = download(out).unwrap();

    EXPECT_EQ(out.extents[0], 1u);
    EXPECT_EQ(out.extents[1], 5u);
    const std::vector<f32> expected = {1.0f, 2.0f, 10.0f, 20.0f, 30.0f};
    ASSERT_TRUE(are_close(out_cpu, expected, 1e-5f));
}

UTEST(concat, layer_forward_batch_exceeds_capacity)
{
    using namespace vika;
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3}).unwrap();

    // Layer only has capacity for 1 sample, but the inputs have batch 2.
    auto layer = ConcatLayer::with_extents({{1, 2}, {1, 3}}).unwrap();
    ASSERT_TRUE(layer.forward({a.const_view(), b.const_view()}).is_error());
}

UTEST(concat, layer_forward_wrong_input_count)
{
    using namespace vika;
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    const auto b = DeviceOwningTensorf::from({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, {2, 3}).unwrap();

    auto layer = ConcatLayer::with_extents({{2, 2}, {2, 3}}).unwrap();
    ASSERT_TRUE(layer.forward({a.const_view()}).is_error());
    ASSERT_TRUE(layer.forward({a.const_view(), b.const_view(), a.const_view()}).is_error());
}

UTEST(concat, layer_forward_shape_mismatch)
{
    using namespace vika;
    const auto a = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    // Wrong last-dim width for the second slot (layer expects 3, this has 4).
    const auto wrong_b = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 4}).unwrap();

    auto layer = ConcatLayer::with_extents({{2, 2}, {2, 3}}).unwrap();
    ASSERT_TRUE(layer.forward({a.const_view(), wrong_b.const_view()}).is_error());
}

UTEST(concat, with_extents_rank_mismatch)
{
    using namespace vika;
    // {2, 2} is rank 2, {2, 2, 2} is rank 3 - must fail at construction.
    ASSERT_TRUE(ConcatLayer::with_extents({{2, 2}, {2, 2, 2}}).is_error());
}

UTEST(concat, with_extents_non_last_dim_mismatch)
{
    using namespace vika;
    // Same rank, but dimension 1 (not the last) disagrees - must fail at construction.
    ASSERT_TRUE(ConcatLayer::with_extents({{2, 3, 4}, {2, 5, 4}}).is_error());
}

UTEST(concat, layer_backward)
{
    using namespace vika;
    const auto upstream =
        DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, {2, 5}).unwrap();

    auto layer = ConcatLayer::with_extents({{2, 2}, {2, 3}}).unwrap();
    auto jobs = layer.backward(upstream.const_view());

    ASSERT_TRUE(jobs.size() == 2);
    const auto d_a = download(jobs[0].wait().unwrap()).unwrap();
    const auto d_b = download(jobs[1].wait().unwrap()).unwrap();

    const std::vector<f32> expected_d_a = {1.0f, 2.0f, 6.0f, 7.0f};
    const std::vector<f32> expected_d_b = {3.0f, 4.0f, 5.0f, 8.0f, 9.0f, 10.0f};
    ASSERT_TRUE(are_close(d_a, expected_d_a, 1e-5f));
    ASSERT_TRUE(are_close(d_b, expected_d_b, 1e-5f));
}
