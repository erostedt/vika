#pragma once

#include <array>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <initializer_list>
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

inline auto to_string(cudaError_t e) -> std::string
{
    return cudaGetErrorString(e);
}

inline auto is_error(cudaError_t err) -> bool
{
    return err != cudaSuccess;
}

template <typename T>
struct Ok
{
    T value;
};

template <typename E>
struct Error
{
    E value;
};

template <typename T>
auto ok(T value) -> Ok<std::decay_t<T>>
{
    return Ok<std::decay_t<T>>{std::move(value)};
}

template <typename E>
auto error(E value) -> Error<std::decay_t<E>>
{
    return Error<std::decay_t<E>>{std::move(value)};
}

template <typename T, typename E>
class Result
{
  public:
    using value_type = T;
    using error_type = E;

    static auto ok(T value) -> Result
    {
        return Result(std::move(value));
    }

    static auto error(E error) -> Result
    {
        return Result(std::move(error));
    }

    Result(Ok<T> ok) : storage(std::move(ok.value))
    {
    }

    Result(Error<E> error) : storage(std::move(error.value))
    {
    }

    auto is_ok() const -> bool
    {
        return std::holds_alternative<T>(storage);
    }

    auto is_error() const -> bool
    {
        return std::holds_alternative<E>(storage);
    }

    auto unwrap() & -> T &
    {
        CHECK_MSG(is_ok(), "called unwrap() on Error Result");
        return std::get<T>(storage);
    }

    auto unwrap() const & -> const T &
    {
        CHECK_MSG(is_ok(), "called unwrap() on Error Result");
        return std::get<T>(storage);
    }

    auto unwrap() && -> T &&
    {
        CHECK_MSG(is_ok(), "called unwrap() on Error Result");
        return std::move(std::get<T>(storage));
    }

    auto unwrap_error() & -> E &
    {
        CHECK_MSG(is_error(), "called unwrap_error() on Ok Result");
        return std::get<E>(storage);
    }

    auto unwrap_error() const & -> const E &
    {
        CHECK_MSG(is_error(), "called unwrap_error() on Ok Result");
        return std::get<E>(storage);
    }

    auto unwrap_error() && -> E &&
    {
        CHECK_MSG(is_error(), "called unwrap_error() on Ok Result");
        return std::move(std::get<E>(storage));
    }

    auto unwrap_or(T fallback) const -> T
    {
        if (is_ok())
        {
            return std::get<T>(storage);
        }
        return std::move(fallback);
    }

  private:
    explicit Result(T value) : storage(std::move(value))
    {
    }
    explicit Result(E error) : storage(std::move(error))
    {
    }

    std::variant<T, E> storage;
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

    static auto empty(const Extents &extents) -> Self
    {
        return HostTensor(std::vector<T>(HostTensor<T, Rank>::size(extents)), extents);
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
struct DeviceDeleter
{
    auto operator()(T *ptr)
    {
        cudaFree(ptr);
    }
};

template <typename T, usize Rank>
struct DeviceTensorView;

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
class DeviceOwningTensor
{
    using Self = DeviceOwningTensor<T, Rank>;
    using Extents = std::array<usize, Rank>;

  public:
    static auto empty(const Extents &extents) -> Result<Self, cudaError_t>
    {
        T *ptr = nullptr;
        const auto err = cudaMalloc(&ptr, vika::byte_count<T>(extents));
        if (err)
        {
            return error(err);
        }
        return ok(Self(ptr, extents));
    }

    static auto from(const std::vector<T> &data, const Extents &extents) -> Result<Self, cudaError_t>
    {
        auto tensor = empty(extents);
        if (tensor.is_error())
        {
            return tensor;
        }
        const auto err =
            cudaMemcpy(tensor.unwrap().data(), data.data(), tensor.unwrap().byte_count(), cudaMemcpyHostToDevice);

        if (is_error(err))
        {
            return error(err);
        }
        return tensor;
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 1>>
    static auto from(const std::vector<T> &data) -> Result<Self, cudaError_t>
    {
        return from(data, {data.size()});
    }

    static auto empty_like(const Self &other) -> Result<Self, cudaError_t>
    {
        return empty(other.extents());
    }

    auto element_count() const -> usize
    {
        return vika::element_count<Rank>(_extents);
    }

    auto byte_count() const -> usize
    {
        return vika::byte_count<T>(_extents);
    }

    auto extents() const -> const Extents &
    {
        return _extents;
    }

    auto data() const -> const T *
    {
        return _data.get();
    }

    auto data() -> T *
    {
        return _data.get();
    }

    auto view() -> DeviceTensorView<T, Rank>
    {
        DeviceTensorView<T, Rank> tensor_view = {.data = _data.get()};
        for (usize i = 0; i < _extents.size(); ++i)
        {
            tensor_view.extents[i] = _extents[i];
        }

        return tensor_view;
    }

    auto const_view() -> DeviceTensorView<const T, Rank>
    {
        DeviceTensorView<const T, Rank> tensor_view = {.data = _data.get()};
        for (usize i = 0; i < _extents.size(); ++i)
        {
            tensor_view.extents[i] = _extents[i];
        }

        return tensor_view;
    }

  private:
    DeviceOwningTensor(T *data, const Extents &extents) : _data(data), _extents(extents)
    {
    }

  private:
    std::unique_ptr<T[], DeviceDeleter<T>> _data;
    Extents _extents;
};

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto copy(const DeviceOwningTensor<T, Rank> &src, HostTensor<T, Rank> &dst) -> cudaError_t
{
    CHECK_MSG(src.extents() == dst.extents(), "element_mismatch");
    return cudaMemcpy(dst.data(), src.data(), src.byte_count(), cudaMemcpyDeviceToHost);
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto copy(const HostTensor<T, Rank> &src, DeviceOwningTensor<T, Rank> &dst) -> cudaError_t
{
    CHECK_MSG(src.extents() == dst.extents(), "element_mismatch");
    return cudaMemcpy(dst.data(), src.data(), dst.byte_count(), cudaMemcpyHostToDevice);
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto upload(const HostTensor<T, Rank> &src) -> Result<DeviceOwningTensor<T, Rank>, cudaError_t>
{
    auto dst = DeviceOwningTensor<T, Rank>::empty(src.extents());
    if (dst.is_error())
    {
        return error(dst);
    }

    const auto err = copy(src, dst.unwrap());
    if (is_error(err))
    {
        return error(err);
    }
    return ok(dst);
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto download(const DeviceOwningTensor<T, Rank> &src) -> Result<HostTensor<T, Rank>, cudaError_t>
{
    auto dst = HostTensor<T, Rank>::empty(src.extents());
    const auto err = copy(src, dst);
    if (is_error(err))
    {
        return error(err);
    }
    return ok(dst);
}

template <typename T, usize Rank>
struct DeviceTensorView
{
    T *data;
    usize extents[Rank];

    __host__ __device__ inline usize element_count()
    {
        usize count = 1;
        for (usize i = 0; i < Rank; ++i)
        {
            count *= extents[i];
        }
        return count;
    }

    __host__ __device__ inline T &operator[](usize i)
    {
        return data[i];
    }

    __host__ __device__ inline const T &operator[](usize i) const
    {
        return data[i];
    }

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

using DeviceOwningTensor1f = DeviceOwningTensor<f32, 1>;
using DeviceOwningTensor2f = DeviceOwningTensor<f32, 2>;
using DeviceOwningTensor3f = DeviceOwningTensor<f32, 3>;
using DeviceOwningTensor4f = DeviceOwningTensor<f32, 4>;

template <usize Rank>
using DeviceTensorViewf = DeviceTensorView<f32, Rank>;
using DeviceTensorView1f = DeviceTensorViewf<1>;
using DeviceTensorView2f = DeviceTensorViewf<2>;
using DeviceTensorView3f = DeviceTensorViewf<3>;
using DeviceTensorView4f = DeviceTensorViewf<4>;

template <usize Rank>
using DeviceTensorConstViewf = DeviceTensorView<const f32, Rank>;
using DeviceTensorConstView1f = DeviceTensorConstViewf<1>;
using DeviceTensorConstView2f = DeviceTensorConstViewf<2>;
using DeviceTensorConstView3f = DeviceTensorConstViewf<3>;
using DeviceTensorConstView4f = DeviceTensorConstViewf<4>;

__global__ auto matmul_kernel(DeviceTensorConstView2f a, DeviceTensorConstView2f b, DeviceTensorView2f out) -> void;

__host__ __device__ inline auto sigmoid(f32 x) -> f32
{
    return 1.0f / (1.0f + std::exp(-x));
}

template <usize Rank>
__global__ auto sigmoid_kernel(DeviceTensorConstViewf<Rank> a, DeviceTensorViewf<Rank> out) -> void
{
    const usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.element_count())
    {
        out[i] = sigmoid(a[i]);
    }
}

}; // namespace vika

#ifdef VIKA_IMPLEMENTATION
namespace vika
{

__global__ auto matmul_kernel(DeviceTensorConstView2f a, DeviceTensorConstView2f b, DeviceTensorView2f out) -> void
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
