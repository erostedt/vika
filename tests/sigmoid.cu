#include "utest.h"

#include "comparison.cuh"

#include "vika.cuh"

UTEST(sigmoid, small)
{
    using namespace vika;
    std::vector<f32> data_a = {-3, -2, -1, 0, 1, 2, 3};
    auto a = CudaOwningTensor1f::empty({data_a.size()}).unwrap();
    auto b = CudaOwningTensor1f::empty({data_a.size()}).unwrap();
    ASSERT_EQ(a.upload(data_a), cudaSuccess);

    dim3 block(1);
    dim3 grid(data_a.size());
    sigmoid_kernel<<<grid, block>>>(a.const_view(), b.view());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto out = download(b).unwrap();
    const std::vector<f32> expected = {0.04742587317756678, 0.11920292202211755, 0.2689414213699951, 0.5,
                                       0.7310585786300049,  0.8807970779778823,  0.9525741268224334};

    ASSERT_TRUE(are_close(out, expected, 1e-5f));
}
