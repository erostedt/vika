#include "utest.h"

#include <cmath>

#include "comparison.cuh"

#include "vika.cuh"

UTEST(cce, forward)
{
    using namespace vika;
    // Two one-hot-labeled rows, predictions already probabilities (as if softmax's own output) -
    // CCELoss does not fuse a softmax internally, see cce_kernel's doc comment.
    const auto predictions =
        DeviceOwningTensorf::from({0.7f, 0.2f, 0.1f, 0.1f, 0.1f, 0.8f}, Extents::of(2, 3)).unwrap();
    const auto targets = DeviceOwningTensorf::from({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, Extents::of(2, 3)).unwrap();

    auto loss_fn = CCELoss::with_extents(Extents::of(2, 3)).unwrap();
    const auto loss = loss_fn.forward(predictions.const_view(), targets.const_view()).wait().unwrap();
    const auto loss_cpu = download(loss).unwrap();

    // -(log(0.7) + log(0.8)) / 2, averaged per sample (row count), not per element - see
    // cce_kernel's doc comment for why.
    const f32 expected = -(std::log(0.7f) + std::log(0.8f)) / 2.0f;
    EXPECT_NEAR(loss_cpu[0], expected, 1e-5f);
}

UTEST(cce, forward_clamps_away_from_log_zero)
{
    using namespace vika;
    // A prediction of exactly 0 at the one-hot target class would make log() produce -inf without
    // the epsilon clamp in cce_kernel - this must come out finite instead.
    const auto predictions = DeviceOwningTensorf::from({0.0f, 1.0f}, Extents::of(1, 2)).unwrap();
    const auto targets = DeviceOwningTensorf::from({1.0f, 0.0f}, Extents::of(1, 2)).unwrap();

    auto loss_fn = CCELoss::with_extents(Extents::of(1, 2)).unwrap();
    const auto loss = loss_fn.forward(predictions.const_view(), targets.const_view()).wait().unwrap();
    const auto loss_cpu = download(loss).unwrap();

    ASSERT_TRUE(std::isfinite(loss_cpu[0]));
}

UTEST(cce, backward)
{
    using namespace vika;
    const auto predictions =
        DeviceOwningTensorf::from({0.7f, 0.2f, 0.1f, 0.1f, 0.1f, 0.8f}, Extents::of(2, 3)).unwrap();
    const auto targets = DeviceOwningTensorf::from({1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, Extents::of(2, 3)).unwrap();

    auto loss_fn = CCELoss::with_extents(Extents::of(2, 3)).unwrap();
    const auto d_inputs = loss_fn.backward(predictions.const_view(), targets.const_view()).wait().unwrap();
    const auto out = download(d_inputs).unwrap();

    // d/dp_ic = -target_ic / p_ic / row_count, zero everywhere targets is zero.
    const std::vector<f32> expected = {-1.0f / 0.7f / 2.0f, 0.0f, 0.0f, 0.0f, 0.0f, -1.0f / 0.8f / 2.0f};
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}
