#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"
#include <vector>

// Flatten only reinterprets its input's extents - it owns no buffer and launches no kernel - so
// what these check is the shape arithmetic and, above all, that the batch dimension is taken from
// the tensor in hand rather than from the capacity the layer was built for.
static auto counting_input(vika::usize batch) -> vika::DeviceOwningTensorf
{
    using namespace vika;
    std::vector<f32> values(batch * 4);
    std::iota(std::begin(values), std::end(values), 1.0f);
    return DeviceOwningTensorf::from(values, {batch, 2, 2, 1}).unwrap();
}

UTEST(flatten, forward_collapses_every_dimension_but_the_batch)
{
    using namespace vika;
    const auto inputs = counting_input(2);
    const auto layer = Flatten2DLayer::with_extents({2, 2, 2, 1}).unwrap();

    const auto forwarded = layer.forward({inputs.const_view()}).wait();
    ASSERT_TRUE(forwarded.is_ok());
    const auto out = forwarded.unwrap();
    EXPECT_EQ(out.rank(), 2u);
    EXPECT_EQ(out.extents[0], 2u);
    EXPECT_EQ(out.extents[1], 4u);

    // A view over the same memory, not a copy.
    EXPECT_EQ(out.data, inputs.data());
    ASSERT_TRUE(are_close(download(out).unwrap(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, 1e-6f));
}

UTEST(flatten, forward_smaller_batch)
{
    using namespace vika;
    // Built for a batch of 4, handed 2. This is the case that was rejected outright until the
    // full-extent comparison became a trailing-extent one.
    const auto inputs = counting_input(2);
    const auto layer = Flatten2DLayer::with_extents({4, 2, 2, 1}).unwrap();

    const auto forwarded = layer.forward({inputs.const_view()}).wait();
    ASSERT_TRUE(forwarded.is_ok());
    const auto out = forwarded.unwrap();
    EXPECT_EQ(out.extents[0], 2u);
    EXPECT_EQ(out.extents[1], 4u);
}

UTEST(flatten, backward_restores_the_input_shape)
{
    using namespace vika;
    const auto upstream =
        DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 4}).unwrap();
    const auto layer = Flatten2DLayer::with_extents({2, 2, 2, 1}).unwrap();

    const auto backwarded = layer.backward(upstream.const_view())[0].wait();
    ASSERT_TRUE(backwarded.is_ok());
    const auto grad = backwarded.unwrap();
    EXPECT_EQ(grad.rank(), 4u);
    EXPECT_EQ(grad.extents[0], 2u);
    EXPECT_EQ(grad.extents[1], 2u);
    EXPECT_EQ(grad.extents[2], 2u);
    EXPECT_EQ(grad.extents[3], 1u);
}

UTEST(flatten, backward_smaller_batch)
{
    using namespace vika;
    // The batch must come from the upstream gradient, not from the layer's capacity - returning
    // the capacity shape would claim more rows than the caller actually has.
    const auto upstream = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4}).unwrap();
    const auto layer = Flatten2DLayer::with_extents({4, 2, 2, 1}).unwrap();

    const auto backwarded = layer.backward(upstream.const_view())[0].wait();
    ASSERT_TRUE(backwarded.is_ok());
    const auto grad = backwarded.unwrap();
    EXPECT_EQ(grad.extents[0], 1u);
    EXPECT_EQ(grad.extents[1], 2u);
    EXPECT_EQ(grad.extents[2], 2u);
    EXPECT_EQ(grad.extents[3], 1u);
}

UTEST(flatten, rejects_wrong_trailing_extents_in_both_directions)
{
    using namespace vika;
    const auto layer = Flatten2DLayer::with_extents({2, 2, 2, 1}).unwrap();

    const auto wrong_input = DeviceOwningTensorf::zero({2, 3, 3, 1}).unwrap();
    EXPECT_TRUE(failed_with(layer.forward({wrong_input.const_view()}).wait(), ErrorKind::Shape, "forward: input.extents is"));

    const auto wrong_upstream = DeviceOwningTensorf::zero({2, 5}).unwrap();
    EXPECT_TRUE(failed_with(layer.backward(wrong_upstream.const_view())[0].wait(), ErrorKind::Shape, "backward: upstream.extents is"));

    const auto input = counting_input(2);
    EXPECT_TRUE(failed_with(layer.forward({input.const_view(), input.const_view()}).wait(), ErrorKind::Shape, "forward: expects exactly"));
}

UTEST(flatten, with_extents_requires_rank_2_or_higher)
{
    using namespace vika;
    // Below rank 2 there is nothing to flatten, and output_extents() reads extents[0]. This was
    // the one layer factory that could not report anything.
    EXPECT_TRUE(failed_with(Flatten2DLayer::with_extents({}), ErrorKind::Shape, "got {}"));
    EXPECT_TRUE(failed_with(Flatten2DLayer::with_extents({4}), ErrorKind::Shape, "got {4}"));
    EXPECT_TRUE(Flatten2DLayer::with_extents({4, 2}).is_ok());
    EXPECT_TRUE(Flatten2DLayer::with_extents({4, 2, 2, 1}).is_ok());
}
