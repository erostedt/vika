#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
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

auto to_string(cudaError_t e) -> std::string;

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

template <usize Rank>
inline auto element_count(const std::array<usize, Rank> &extents) -> usize
{
    using namespace std;
    return accumulate(begin(extents), end(extents), 1ul, std::multiplies<>{});
}

template <typename T, usize Rank>
inline auto byte_count(const std::array<usize, Rank> &extents) -> usize
{
    return element_count(extents) * sizeof(T);
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
class HostTensor
{
    using Extents = std::array<usize, Rank>;
    using Self = HostTensor<T, Rank>;
    using iterator = typename std::vector<T>::iterator;
    using const_iterator = typename std::vector<T>::const_iterator;

  public:
    auto operator[](usize i) -> T &
    {
        return _data[i];
    }

    auto operator[](usize i) const -> const T &
    {
        return _data[i];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 2>>
    auto operator()(usize r, usize c) -> T &
    {
        return _data[r * cols() + c];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 2>>
    auto operator()(usize r, usize c) const -> const T &
    {
        return _data[r * cols() + c];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 4>>
    auto operator()(usize n, usize h, usize w, usize c) -> T &
    {
        const auto height = _extents[1];
        const auto width = _extents[2];
        const auto channels = _extents[3];
        return _data[((n * height + h) * width + w) * channels + c];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 4>>
    auto operator()(usize n, usize h, usize w, usize c) const -> const T &
    {
        const auto height = _extents[1];
        const auto width = _extents[2];
        const auto channels = _extents[3];
        return _data[((n * height + h) * width + w) * channels + c];
    }

    auto extents() const -> const Extents &
    {
        return _extents;
    }

    template <usize Dimension, typename = std::enable_if_t<(Dimension < Rank)>>
    auto extent() const -> usize
    {
        return _extents[Dimension];
    }

    auto data() -> T *
    {
        return _data.data();
    }

    auto data() const -> const T *
    {
        return _data.data();
    }

    static auto size(const Extents &extents) -> usize
    {
        using namespace std;
        return accumulate(std::begin(extents), std::end(extents), usize{1}, multiplies<>{});
    }

    auto size() const -> usize
    {
        return std::size(_data);
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 2>>
    auto rows() const -> usize
    {
        return _extents[0];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 2>>
    auto cols() const -> usize
    {
        return _extents[1];
    }

    auto begin() -> iterator
    {
        return std::begin(_data);
    }

    auto begin() const -> const_iterator
    {
        return std::cbegin(_data);
    }

    auto end() -> iterator
    {
        return std::end(_data);
    }

    auto end() const -> const_iterator
    {
        return std::cend(_data);
    }

    static auto zero(const Extents &extents) -> Self
    {
        return HostTensor(std::vector<T>(HostTensor<T, Rank>::size(extents), T{}), extents);
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 1>>
    static auto zero(usize element_count) -> Self
    {
        return Self::zero(Extents{element_count});
    }

    template <typename OtherType>
    static auto zero_like(const HostTensor<OtherType, Rank> &tensor) -> Self
    {
        return Self::zero(tensor.extents());
    }

    template <usize NewRank>
    static auto reshaped(const HostTensor<T, Rank> &tensor, const std::array<usize, NewRank> &extents)
        -> HostTensor<T, NewRank>
    {
        assert(tensor.size() == (HostTensor<T, NewRank>::size(extents)));
        auto data = tensor._data;
        return HostTensor<T, NewRank>(std::move(data), extents);
    }

    static auto from(std::initializer_list<T> data, const Extents &extents) -> Self
    {
        return copy_from(data, extents);
    }

    static auto copy_from(std::vector<T> data, const Extents &extents) -> Self
    {
        assert(std::size(data) == Self::size(extents));
        return HostTensor(std::move(data), extents);
    }

  private:
    template <typename OtherT, usize OtherRank, typename>
    friend class HostTensor;

    HostTensor(std::vector<T> &&data, const Extents &extents) : _data(std::move(data)), _extents(extents)
    {
        assert(size(extents) > 0);
    }

  private:
    std::vector<T> _data{};
    const Extents _extents{};
};

template <usize Rank>
using HostTensorf = HostTensor<f32, Rank>;
using HostTensor1f = HostTensorf<1>;
using HostTensor2f = HostTensorf<2>;
using HostTensor3f = HostTensorf<3>;
using HostTensor4f = HostTensorf<4>;
using HostTensor4u = HostTensor<u32, 4>;
using Vectorf = HostTensor1f;
using Vectoru = HostTensor<u32, 1>;
using Matrixf = HostTensor2f;

template <typename T>
struct CudaDeleter
{
    auto operator()(T *ptr)
    {
        cudaFree(ptr);
    }
};

template <typename T, usize Rank>
struct CudaTensorView;

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
class CudaOwningTensor
{
    using Self = CudaOwningTensor<T, Rank>;

  public:
    static auto empty(const std::array<usize, Rank> &extents) -> Result<Self, cudaError_t>
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
        return vika::element_count<Rank>(_extents);
    }

    auto byte_count() const -> usize
    {
        return vika::byte_count<T>(_extents);
    }

    auto data() -> T *
    {
        return _data.get();
    }

    auto view() -> CudaTensorView<T, Rank>
    {
        CudaTensorView<T, Rank> tensor_view = {.data = _data.get()};
        for (usize i = 0; i < _extents.size(); ++i)
        {
            tensor_view.extents[i] = _extents[i];
        }

        return tensor_view;
    }

    auto const_view() -> CudaTensorView<const T, Rank>
    {
        CudaTensorView<const T, Rank> tensor_view = {.data = _data.get()};
        for (usize i = 0; i < _extents.size(); ++i)
        {
            tensor_view.extents[i] = _extents[i];
        }

        return tensor_view;
    }

  private:
    CudaOwningTensor(T *data, const std::array<usize, Rank> &extents) : _data(data), _extents(extents)
    {
    }

  private:
    std::unique_ptr<T[], CudaDeleter<T>> _data;
    std::array<usize, Rank> _extents;
};

template <typename T, usize Rank>
struct CudaTensorView
{
    T *data;
    usize extents[Rank];

    template <usize R = Rank, typename = std::enable_if_t<R == 1>>
    __host__ __device__ inline T &operator()(usize x)
    {
        return data[x];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 1>>
    __host__ __device__ inline const T &operator()(usize x) const
    {
        return data[x];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 2>>
    __host__ __device__ inline T &operator()(usize x, usize y)
    {
        return data[x * extents[1] + y];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 2>>
    __host__ __device__ inline const T &operator()(usize x, usize y) const
    {
        return data[x * extents[1] + y];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 3>>
    __host__ __device__ inline T &operator()(usize x, usize y, usize z)
    {
        return data[(x * extents[1] + y) * extents[2] + z];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 3>>
    __host__ __device__ inline const T &operator()(usize x, usize y, usize z) const
    {
        return data[(x * extents[1] + y) * extents[2] + z];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 4>>
    __host__ __device__ inline T &operator()(usize x, usize y, usize z, usize w)
    {
        return data[((x * extents[1] + y) * extents[2] + z) * extents[3] + w];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 4>>
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

__global__ auto matmul_kernel(CudaTensorView<const f32, 2> a, CudaTensorView<const f32, 2> b,
                              CudaTensorView<f32, 2> out) -> void;

}; // namespace vika

#ifdef VIKA_IMPLEMENTATION
namespace vika
{
auto to_string(cudaError_t e) -> std::string
{
    return cudaGetErrorString(e);
}

__global__ auto matmul_kernel(CudaTensorView<const f32, 2> a, CudaTensorView<const f32, 2> b,
                              CudaTensorView<f32, 2> out) -> void
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
#endif

// TODO (ecrt):
// - CpuTensor
// - Requirements on T and Rank
// - Tiled matmul
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
