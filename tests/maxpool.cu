#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"
#include <vector>

// A 4x4 input, 2x2 pooling at stride 2, so the four windows are disjoint:
//
//      1  2 |  3  4          window maxima     6  8      argmax (flat, row-major)   5  7
//      5  6 |  7  8      ->                   14 16                                13 15
//     ------+------
//      9 10 | 11 12
//     13 14 | 15 16
static auto counting_input() -> vika::DeviceOwningTensorf
{
    using namespace vika;
    std::vector<f32> values(16);
    std::iota(std::begin(values), std::end(values), 1.0f);
    return DeviceOwningTensorf::from(values, Extents::of(1, 4, 4, 1)).unwrap();
}

UTEST(maxpool, forward_picks_the_window_maximum)
{
    using namespace vika;
    const auto inputs = counting_input();
    auto layer = MaxPool2DLayer::with_extents(1, 4, 4, 1, 2, 2, 2).unwrap();

    const auto forwarded = layer.forward({inputs.const_view()}).wait();
    ASSERT_TRUE(forwarded.is_ok());
    const auto out = download(forwarded.unwrap()).unwrap();

    EXPECT_EQ(out.extent(0), 1u);
    EXPECT_EQ(out.extent(1), 2u);
    EXPECT_EQ(out.extent(2), 2u);
    ASSERT_TRUE(are_close(out, {6.0f, 8.0f, 14.0f, 16.0f}, 1e-6f));

    // argmax stores the flat (row, col) offset into one input image, which backward turns back
    // into coordinates - so its exact encoding matters, not just that the maxima were right.
    const auto argmax = download(layer.argmax).unwrap();
    ASSERT_TRUE(are_equal(argmax, HostTensoru::copy_from({5u, 7u, 13u, 15u}, Extents::of(1, 2, 2, 1)).unwrap()));
}

UTEST(maxpool, backward_routes_the_gradient_to_the_argmax)
{
    using namespace vika;
    const auto inputs = counting_input();
    auto layer = MaxPool2DLayer::with_extents(1, 4, 4, 1, 2, 2, 2).unwrap();
    layer.forward({inputs.const_view()}).wait().unwrap();

    const auto upstream = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, Extents::of(1, 2, 2, 1)).unwrap();
    const auto gpu_grad = layer.backward(upstream.const_view())[0].wait().unwrap();
    const auto grad = download(gpu_grad).unwrap();

    // Every window's gradient lands on the one element that won it; everything else stays zero.
    std::vector<f32> expected(16, 0.0f);
    expected[5] = 1.0f;
    expected[7] = 2.0f;
    expected[13] = 3.0f;
    expected[15] = 4.0f;
    ASSERT_TRUE(are_close(grad, expected, 1e-6f));
}

UTEST(maxpool, backward_ignores_an_out_of_range_argmax)
{
    using namespace vika;
    // backward() reads an argmax that only forward() writes, and argmax comes from empty() - so a
    // standalone backward-before-forward reads whatever was in that memory. Unbounded, that index
    // scatters an atomicAdd anywhere; bounded, the contribution is dropped.
    auto layer = MaxPool2DLayer::with_extents(1, 4, 4, 1, 2, 2, 2).unwrap();

    auto poison = HostTensoru::zero(Extents::of(1, 2, 2, 1)).unwrap();
    for (usize i = 0; i < poison.size(); ++i)
    {
        poison[i] = 0xFFFF0000u;
    }
    copy(poison, layer.argmax).unwrap();

    const auto upstream = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, Extents::of(1, 2, 2, 1)).unwrap();
    const auto backwarded = layer.backward(upstream.const_view())[0].wait();
    ASSERT_TRUE(backwarded.is_ok());
    const auto grad = download(backwarded.unwrap()).unwrap();

    ASSERT_TRUE(are_close(grad, std::vector<f32>(16, 0.0f), 1e-6f));
}

UTEST(maxpool, forward_smaller_batch)
{
    using namespace vika;
    // Capacity 2, actual batch 1.
    const auto inputs = counting_input();
    auto layer = MaxPool2DLayer::with_extents(2, 4, 4, 1, 2, 2, 2).unwrap();

    const auto forwarded = layer.forward({inputs.const_view()}).wait();
    ASSERT_TRUE(forwarded.is_ok());
    const auto out = download(forwarded.unwrap()).unwrap();

    EXPECT_EQ(out.extent(0), 1u);
    ASSERT_TRUE(are_close(out, {6.0f, 8.0f, 14.0f, 16.0f}, 1e-6f));
}

UTEST(maxpool, forward_rejects_wrong_trailing_extents)
{
    using namespace vika;
    auto layer = MaxPool2DLayer::with_extents(1, 4, 4, 1, 2, 2, 2).unwrap();
    const auto wrong = DeviceOwningTensorf::zero(Extents::of(1, 5, 5, 1)).unwrap();
    EXPECT_TRUE(failed_with(layer.forward({wrong.const_view()}).wait(), ErrorKind::Shape));
}

UTEST(maxpool, with_extents_rejects_zero_stride)
{
    using namespace vika;
    EXPECT_TRUE(failed_with(MaxPool2DLayer::with_extents(1, 4, 4, 1, 2, 2, 0), ErrorKind::Shape));
    EXPECT_TRUE(MaxPool2DLayer::with_extents(1, 4, 4, 1, 2, 2, 2).is_ok());
}

UTEST(maxpool, with_extents_rejects_a_pool_larger_than_the_input)
{
    using namespace vika;
    EXPECT_TRUE(failed_with(MaxPool2DLayer::with_extents(1, 2, 2, 1, 5, 5, 1), ErrorKind::Shape));
    EXPECT_TRUE(MaxPool2DLayer::with_extents(1, 2, 2, 1, 2, 2, 1).is_ok());
}
