#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(sigmoid, small)
{
    using namespace vika;
    auto a = CudaOwningTensor1f::from({-3, -2, -1, 0, 1, 2, 3}).unwrap();
    auto b = CudaOwningTensor1f::empty_like(a).unwrap();

    dim3 block(1);
    dim3 grid(a.element_count());
    sigmoid_kernel<<<grid, block>>>(a.const_view(), b.view());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto out = download(b).unwrap();
    const std::vector<f32> expected = {0.04742587317756678, 0.11920292202211755, 0.2689414213699951, 0.5,
                                       0.7310585786300049,  0.8807970779778823,  0.9525741268224334};

    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}
