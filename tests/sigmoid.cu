#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(sigmoid, forward)
{
    using namespace vika;
    const auto inputs = DeviceOwningTensor2f::from({-2, -1, 0, 1, 2, 3}, {2, 3}).unwrap();
    auto layer = SigmoidLayer::with_extents({2, 3}).unwrap();
    const auto outputs = layer.forward(inputs.const_view()).wait().unwrap();

    const auto out = download(outputs).unwrap();
    const std::vector<f32> expected = {0.11920292202211755, 0.2689414213699951, 0.5,
                                       0.7310585786300049,  0.8807970779778823, 0.9525741268224334};

    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}

UTEST(sigmoid, backward)
{
    using namespace vika;

    const auto upstream_gradient = DeviceOwningTensor2f::from({1.0, 2.0, 3.0, 4.0}, {2, 2}).unwrap();
    auto layer = SigmoidLayer::with_extents({2, 2}).unwrap();
    copy(HostTensor2f::from({0.0, 0.5, 0.7310585786300048793, 1.0}, {2, 2}), layer.outputs);

    const auto outputs = layer.backward(upstream_gradient.const_view()).wait().unwrap();

    const auto out = download(outputs).unwrap();
    const std::vector<f32> expected = {0.0, 0.5, 0.5898357997244455576, 0.0};
    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}
