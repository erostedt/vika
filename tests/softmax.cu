#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(softmax, forward)
{
    using namespace vika;
    const auto inputs = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f}, Extents::of(2, 3)).unwrap();
    auto layer = SoftmaxLayer::with_extents(Extents::of(2, 3)).unwrap();
    const auto outputs = layer.forward({inputs.const_view()}).wait().unwrap();

    const auto out = download(outputs).unwrap();
    const std::vector<f32> expected = {0.09003057317038046f, 0.24472847105479767f, 0.6652409557748219f,
                                       1.0f / 3.0f,          1.0f / 3.0f,          1.0f / 3.0f};
    ASSERT_TRUE(are_close(out, expected, 1e-5f));

    // Every row must itself sum to 1 - the defining property of softmax, not just a byproduct of
    // matching the hand-computed values above.
    for (usize row = 0; row < 2; ++row)
    {
        f32 sum = 0.0f;
        for (usize col = 0; col < 3; ++col)
        {
            sum += out[row * 3 + col];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f);
    }
}

UTEST(softmax, forward_is_shift_invariant)
{
    using namespace vika;
    // Same row as `forward`'s first row, shifted by a large constant - only correct if
    // softmax_forward actually subtracts the row max before exponentiating, otherwise expf(1e4)
    // overflows to inf and the result is nan instead of matching the unshifted row.
    const auto inputs = DeviceOwningTensorf::from({10001.0f, 10002.0f, 10003.0f}, Extents::of(1, 3)).unwrap();
    auto layer = SoftmaxLayer::with_extents(Extents::of(1, 3)).unwrap();
    const auto outputs = layer.forward({inputs.const_view()}).wait().unwrap();

    const auto out = download(outputs).unwrap();
    const std::vector<f32> expected = {0.09003057317038046f, 0.24472847105479767f, 0.6652409557748219f};
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}

UTEST(softmax, backward)
{
    using namespace vika;

    const auto upstream_gradient = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f}, Extents::of(1, 3)).unwrap();
    auto layer = SoftmaxLayer::with_extents(Extents::of(1, 3)).unwrap();
    copy(HostTensorf::from({0.2f, 0.3f, 0.5f}, Extents::of(1, 3)).unwrap(), layer.outputs).unwrap();

    const auto outputs = layer.backward(upstream_gradient.const_view())[0].wait().unwrap();

    const auto out = download(outputs).unwrap();
    // d_i = y_i * (upstream_i - dot), dot = sum(y * upstream) = 0.2*1 + 0.3*2 + 0.5*3 = 2.3
    const std::vector<f32> expected = {-0.26f, -0.09f, 0.35f};
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}
