#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <variant>
#include <vector>

using i32 = int32_t;
using u32 = uint32_t;
using f32 = float;
using usize = size_t;

#define CHECK_MSG(expr, msg)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(expr))                                                                                                   \
        {                                                                                                              \
            fprintf(stderr, "Check failed: %s\n", msg);                                                                \
            assert(expr);                                                                                              \
        }                                                                                                              \
    } while (0)

namespace vika
{

auto to_string(cudaError_t e) -> std::string
{
    return cudaGetErrorString(e);
}

template <typename T, typename E>
class Result
{
  public:
    Result(const T &value) : _union(std::in_place_index<0>, value)
    {
    }

    Result(T &&value) : _union(std::in_place_index<0>, std::move(value))
    {
    }

    Result(const E &error) : _union(std::in_place_index<1>, error)
    {
    }

    Result(E &&error) : _union(std::in_place_index<1>, std::move(error))
    {
    }

    auto unwrap() && -> T
    {
        CHECK_MSG(is_some(), to_string(std::get<1>(_union)));
        return std::move(std::get<0>(_union));
    }

    auto is_some() const -> bool
    {
        return _union.index() == 0;
    }

    auto is_error() const -> bool
    {
        return _union.index() == 1;
    }

  private:
    template <typename... Args>
    explicit Result(std::in_place_index_t<0>, Args &&...args)
        : _union(std::in_place_index<0>, std::forward<Args>(args)...)
    {
    }

    template <typename... Args>
    explicit Result(std::in_place_index_t<1>, Args &&...args)
        : _union(std::in_place_index<1>, std::forward<Args>(args)...)
    {
    }

    std::variant<T, E> _union;
};

template <typename T>
struct CudaDeleter
{
    auto operator()(T *ptr)
    {
        cudaFree(ptr);
    }
};

template <usize Dimensions>
auto element_count(const std::array<usize, Dimensions> &extents) -> usize
{
    using namespace std;
    return accumulate(begin(extents), end(extents), 1ul, std::multiplies<>{});
}

template <typename T, usize Dimensions>
auto byte_count(const std::array<usize, Dimensions> &extents) -> usize
{
    return element_count(extents) * sizeof(T);
}

template <typename T, usize Dimensions>
struct CudaTensorView;

template <typename T, usize Dimensions>
class CudaOwningTensor
{
    using Self = CudaOwningTensor<T, Dimensions>;

  public:
    static auto create(const std::array<usize, Dimensions> &extents) -> Result<Self, cudaError_t>
    {
        T *ptr = nullptr;
        const auto err = cudaMalloc(&ptr, vika::byte_count<T>(extents));
        if (err)
        {
            return err;
        }
        return Self(ptr, extents);
    }

    auto upload(const std::vector<T> &data) -> cudaError_t
    {
        CHECK_MSG(data.size() == element_count(), "element_mismatch");
        return cudaMemcpy(_data.get(), data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
    }

    auto download(std::vector<T> &data) -> cudaError_t
    {
        const auto count = element_count();
        if (data.size() != count)
        {
            data.resize(count);
        }
        return cudaMemcpy(data.data(), _data.get(), byte_count(), cudaMemcpyDeviceToHost);
    }

    auto download() -> Result<std::vector<T>, cudaError_t>
    {
        std::vector<T> data(element_count());
        cudaError_t err = cudaMemcpy(data.data(), _data.get(), byte_count(), cudaMemcpyDeviceToHost);
        if (err)
        {
            return err;
        }
        return data;
    }

    auto element_count() const -> usize
    {
        return vika::element_count<Dimensions>(_extents);
    }

    auto byte_count() const -> usize
    {
        return vika::byte_count<T>(_extents);
    }

    auto data() -> T *
    {
        return _data.get();
    }

    auto view() -> CudaTensorView<T, Dimensions>
    {
        CudaTensorView<T, Dimensions> tensor_view = {.data = _data.get()};
        for (usize i = 0; i < _extents.size(); ++i)
        {
            tensor_view.extents[i] = _extents[i];
        }

        return tensor_view;
    }

    auto const_view() -> CudaTensorView<const T, Dimensions>
    {
        CudaTensorView<const T, Dimensions> tensor_view = {.data = _data.get()};
        for (usize i = 0; i < _extents.size(); ++i)
        {
            tensor_view.extents[i] = _extents[i];
        }

        return tensor_view;
    }

  private:
    CudaOwningTensor(T *data, const std::array<usize, Dimensions> &extents) : _data(data), _extents(extents)
    {
    }

  private:
    std::unique_ptr<T[], CudaDeleter<T>> _data;
    std::array<usize, Dimensions> _extents;
};

template <typename T, usize Dimensions>
struct CudaTensorView
{
    T *data;
    usize extents[Dimensions];

    template <usize D = Dimensions, typename = std::enable_if_t<D == 1>>
    __host__ __device__ inline T &operator()(usize x)
    {
        return data[x];
    }

    template <usize D = Dimensions, typename = std::enable_if_t<D == 1>>
    __host__ __device__ inline const T &operator()(usize x) const
    {
        return data[x];
    }

    template <usize D = Dimensions, typename = std::enable_if_t<D == 2>>
    __host__ __device__ inline T &operator()(usize x, usize y)
    {
        return data[x * extents[1] + y];
    }

    template <usize D = Dimensions, typename = std::enable_if_t<D == 2>>
    __host__ __device__ inline const T &operator()(usize x, usize y) const
    {
        return data[x * extents[1] + y];
    }

    template <usize D = Dimensions, typename = std::enable_if_t<D == 3>>
    __host__ __device__ inline T &operator()(usize x, usize y, usize z)
    {
        return data[(x * extents[1] + y) * extents[2] + z];
    }

    template <usize D = Dimensions, typename = std::enable_if_t<D == 3>>
    __host__ __device__ inline const T &operator()(usize x, usize y, usize z) const
    {
        return data[(x * extents[1] + y) * extents[2] + z];
    }

    template <usize D = Dimensions, typename = std::enable_if_t<D == 4>>
    __host__ __device__ inline T &operator()(usize x, usize y, usize z, usize w)
    {
        return data[((x * extents[1] + y) * extents[2] + z) * extents[3] + w];
    }

    template <usize D = Dimensions, typename = std::enable_if_t<D == 4>>
    __host__ __device__ inline const T &operator()(usize x, usize y, usize z, usize w) const
    {
        return data[((x * extents[1] + y) * extents[2] + z) * extents[3] + w];
    }
};

using CudaOwningTensor1f = CudaOwningTensor<f32, 1>;
using CudaOwningTensor2f = CudaOwningTensor<f32, 2>;
using CudaOwningTensor3f = CudaOwningTensor<f32, 3>;
using CudaOwningTensor4f = CudaOwningTensor<f32, 4>;

using CudaTensorView1f = CudaTensorView<f32, 1>;
using CudaTensorView2f = CudaTensorView<f32, 2>;
using CudaTensorView3f = CudaTensorView<f32, 3>;
using CudaTensorView4f = CudaTensorView<f32, 4>;

__global__ void matmul_kernel(CudaTensorView<const f32, 2> a, CudaTensorView<const f32, 2> b,
                              CudaTensorView<f32, 2> out)
{
    const usize row = blockIdx.y * blockDim.y + threadIdx.y;
    const usize col = blockIdx.x * blockDim.x + threadIdx.x;

    const usize m = a.extents[0];
    const usize k = a.extents[1];
    const usize n = b.extents[1];

    if (row >= m || col >= n)
    {
        return;
    }

    f32 sum = 0;
    for (usize i = 0; i < k; ++i)
    {
        sum += a(row, i) * b(i, col);
    }

    out(row, col) = sum;
}
}; // namespace vika

// TODO (ecrt):
// - Tiled matmul
// - CpuTensor
//
// - Sigmoid forward
// - Sigmoid backward
// - Sigmoid weight update
// - Sigmoid Layer
//
// - Flatten Forward
// - Flatten Backward
// - Flatten weight update
// - Flatten Layer
//
// - Dense Forward
// - Dense Backward
// - Dense weight update
// - Dense Layer
//
// - Conv Forward
// - Conv Backward
// - Conv weight update
// - Conv Layer
//
// - Maxpool Forward
// - Maxpool Backward
// - Maxpool weight update
// - Maxpool Layer
//
// - Softmax Forward
// - Softmax Backward
// - Softmax weight update
// - Softmax Layer
//
// - CategoricalCrossEntropy Forward
// - CategoricalCrossEntropy Backward
// - CategoricalCrossEntropy weight update
// - CategoricalCrossEntropy Layer
//
// - Adam optimizer
// - Link layers
// - Pick device
// - Events/Async/Streams?
//
// - stb header only
