#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"
#include <vector>

// MSELoss had no test of its own - it was only ever exercised end-to-end by the training tests,
// which converge whether or not the constant factors are right.
//
// predictions {1, 2, 3, 4} against zero targets:
//   forward  = mean(diff^2)      = (1 + 4 + 9 + 16) / 4 = 7.5
//   backward = 2 * diff / count  = {0.5, 1.0, 1.5, 2.0}
UTEST(mse, forward_averages_over_every_element)
{
    using namespace vika;
    const auto predictions = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    const auto targets = DeviceOwningTensorf::zero({2, 2}).unwrap();
    auto loss = MSELoss::with_extents({2, 2}).unwrap();

    const auto forwarded = loss.forward(predictions.const_view(), targets.const_view()).wait();
    ASSERT_TRUE(forwarded.is_ok());
    const auto value = download(forwarded.unwrap()).unwrap();

    EXPECT_EQ(value.size(), 1u);
    EXPECT_NEAR(value[0], 7.5f, 1e-5f);
}

UTEST(mse, backward_is_the_derivative_of_that_average)
{
    using namespace vika;
    const auto predictions = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}).unwrap();
    const auto targets = DeviceOwningTensorf::zero({2, 2}).unwrap();
    auto loss = MSELoss::with_extents({2, 2}).unwrap();

    const auto backwarded = loss.backward(predictions.const_view(), targets.const_view()).wait();
    ASSERT_TRUE(backwarded.is_ok());
    const auto grad = download(backwarded.unwrap()).unwrap();

    ASSERT_TRUE(are_close(grad, {0.5f, 1.0f, 1.5f, 2.0f}, 1e-6f));
}

UTEST(mse, smaller_batch_averages_over_the_rows_it_was_given)
{
    using namespace vika;
    // Built for a batch of 2, handed 1: the average is over that row's 2 elements, not over the
    // capacity - (1 + 4) / 2 = 2.5 - and the gradient is sliced to match.
    const auto predictions = DeviceOwningTensorf::from({1.0f, 2.0f}, {1, 2}).unwrap();
    const auto targets = DeviceOwningTensorf::zero({1, 2}).unwrap();
    auto loss = MSELoss::with_extents({2, 2}).unwrap();

    const auto value = download(loss.forward(predictions.const_view(), targets.const_view()).wait().unwrap()).unwrap();
    EXPECT_NEAR(value[0], 2.5f, 1e-5f);

    const auto grad = download(loss.backward(predictions.const_view(), targets.const_view()).wait().unwrap()).unwrap();
    EXPECT_EQ(grad.extent(0), 1u);
    ASSERT_TRUE(are_close(grad, {1.0f, 2.0f}, 1e-6f));
}

UTEST(mse, rejects_predictions_and_targets_of_different_batch)
{
    using namespace vika;
    // The kernel sizes its launch off predictions and indexes straight into targets, so a
    // mismatch here used to be an out-of-bounds read reported as success.
    const auto predictions = DeviceOwningTensorf::zero({2, 2}).unwrap();
    const auto targets = DeviceOwningTensorf::zero({1, 2}).unwrap();
    auto loss = MSELoss::with_extents({2, 2}).unwrap();

    EXPECT_TRUE(loss.forward(predictions.const_view(), targets.const_view()).wait().is_error());
    EXPECT_TRUE(loss.backward(predictions.const_view(), targets.const_view()).wait().is_error());
}
