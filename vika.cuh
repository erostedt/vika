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
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

using i32 = int32_t;
using u32 = uint32_t;
using f32 = float;
using usize = size_t;

namespace vika
{

#define CHECK_MSG(expr, msg)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(expr))                                                                                                   \
        {                                                                                                              \
            fprintf(stderr, "Check failed: %s\n", msg);                                                                \
            assert(expr);                                                                                              \
        }                                                                                                              \
    } while (0)

#define unwrap_or_return(expr)                                                                                         \
    ({                                                                                                                 \
        auto _res = (expr);                                                                                            \
        if (_res.is_error())                                                                                           \
        {                                                                                                              \
            return error(_res.unwrap_error());                                                                         \
        }                                                                                                              \
        std::move(_res.unwrap());                                                                                      \
    })

inline auto to_string(cudaError_t e) -> std::string
{
    return cudaGetErrorString(e);
}

inline auto is_error(cudaError_t err) -> bool
{
    return err != cudaSuccess;
}

template <usize Rank>
inline auto to_extents(const usize data[Rank]) -> std::array<usize, Rank>
{
    std::array<usize, Rank> extents{};
    for (usize i = 0; i < Rank; ++i)
    {
        extents[i] = data[i];
    }
    return extents;
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

#define return_on_cuda_error(cudacall)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto _err = (cudacall);                                                                                  \
        if (is_error(_err))                                                                                            \
        {                                                                                                              \
            return error(_err);                                                                                        \
        }                                                                                                              \
    } while (0)

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

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
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

    template <usize Dimension, usize R = Rank, typename = std::enable_if_t<(Dimension < Rank)>>
    auto extent() const -> usize
    {
        return _extents[Dimension];
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
        return DeviceTensorView<T, Rank>(_data.get(), _extents);
    }

    auto const_view() const -> DeviceTensorView<const T, Rank>
    {
        return DeviceTensorView<const T, Rank>(_data.get(), _extents);
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
auto copy(const DeviceTensorView<const T, Rank> &src, HostTensor<T, Rank> &dst) -> cudaError_t
{
    std::array<usize, Rank> extents;
    std::copy(src.extents, src.extents + Rank, extents.begin());
    CHECK_MSG((dst.extents() == extents), "element_mismatch");
    return cudaMemcpy(dst.data(), src.data, src.byte_count(), cudaMemcpyDeviceToHost);
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
auto download(const DeviceTensorView<const T, Rank> &src) -> Result<HostTensor<T, Rank>, cudaError_t>
{
    auto dst = HostTensor<T, Rank>::empty(to_extents<Rank>(src.extents));
    const auto err = copy(src, dst);
    if (is_error(err))
    {
        return error(err);
    }
    return ok(dst);
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto download(const DeviceOwningTensor<T, Rank> &src) -> Result<HostTensor<T, Rank>, cudaError_t>
{
    return download(src.const_view());
}

template <typename T, usize Rank, typename>
struct DeviceTensorView
{
    DeviceTensorView(T *data_, const std::array<usize, Rank> &extents_) : data(data_)
    {
        std::copy(std::begin(extents_), std::end(extents_), extents);
        std::exclusive_scan(std::rbegin(extents_), std::rend(extents_), std::rbegin(strides), 1, std::multiplies<>{});
    }

    T *data = nullptr;
    usize extents[Rank] = {};
    usize strides[Rank] = {};

    __host__ __device__ inline usize element_count() const
    {
        usize count = 1;
        for (usize i = 0; i < Rank; ++i)
        {
            count *= extents[i];
        }
        return count;
    }

    __host__ __device__ inline usize byte_count() const
    {
        return element_count() * sizeof(T);
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
        return data[x * strides[0] + y * strides[1]];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 2>>
    __host__ __device__ inline const T &operator()(usize x, usize y) const
    {
        return data[x * strides[0] + y * strides[1]];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 3>>
    __host__ __device__ inline T &operator()(usize x, usize y, usize z)
    {
        return data[x * strides[0] + y * strides[1] + z * strides[2]];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 3>>
    __host__ __device__ inline const T &operator()(usize x, usize y, usize z) const
    {

        return data[x * strides[0] + y * strides[1] + z * strides[2]];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 4>>
    __host__ __device__ inline T &operator()(usize x, usize y, usize z, usize w)
    {
        return data[x * strides[0] + y * strides[1] + z * strides[2] + w * strides[3]];
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 4>>
    __host__ __device__ inline const T &operator()(usize x, usize y, usize z, usize w) const
    {
        return data[x * strides[0] + y * strides[1] + z * strides[2] + w * strides[3]];
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

inline auto transposed(const DeviceTensorConstView2f &view) -> DeviceTensorConstView2f
{
    auto transposed_view = view;
    std::swap(transposed_view.strides[0], transposed_view.strides[1]);
    std::swap(transposed_view.extents[0], transposed_view.extents[1]);
    return transposed_view;
}

__global__ auto matmul_kernel(DeviceTensorConstView2f a, DeviceTensorConstView2f b, DeviceTensorView2f out) -> void;

__global__ auto dense_forward(DeviceTensorConstView2f inputs, DeviceTensorConstView2f weights,
                              DeviceTensorConstView1f biases, DeviceTensorView2f out) -> void;

__global__ auto dense_backward(DeviceTensorConstView2f d_outputs, DeviceTensorConstView2f weights,
                               DeviceTensorView2f d_inputs) -> void;

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

__global__ auto add_bias(DeviceTensorConstView2f matrix, DeviceTensorConstView1f biases, DeviceTensorView2f out)
    -> void;

__global__ auto sum_rows(DeviceTensorConstView2f matrix, DeviceTensorView1f out) -> void;

template <typename T>
struct KernelJob
{
    auto wait() -> Result<T, cudaError_t>
    {
        const auto err = cudaStreamSynchronize(stream);
        if (is_error(err))
        {
            return error(err);
        }
        return ok(value);
    }
    T value;
    cudaStream_t stream;
};

struct DenseLayer
{
    auto static with_weights(usize batch_size, DeviceOwningTensor2f weights, DeviceOwningTensor1f biases)
        -> Result<DenseLayer, cudaError_t>
    {
        const auto feature_count = weights.extent<0>();
        const auto neuron_count = weights.extent<1>();

        auto outputs = unwrap_or_return(DeviceOwningTensor2f::empty({batch_size, neuron_count}));
        auto d_inputs = unwrap_or_return(DeviceOwningTensor2f::empty({batch_size, feature_count}));
        auto d_outputs = unwrap_or_return(DeviceOwningTensor2f::empty({batch_size, neuron_count}));
        auto d_weights = unwrap_or_return(DeviceOwningTensor2f::empty_like(weights));
        auto d_biases = unwrap_or_return(DeviceOwningTensor1f::empty_like(biases));
        cudaStream_t stream;
        return_on_cuda_error(cudaStreamCreate(&stream));
        return ok<DenseLayer>({
            .outputs = std::move(outputs),
            .weights = std::move(weights),
            .biases = std::move(biases),
            .d_inputs = std::move(d_inputs),
            .d_weights = std::move(d_weights),
            .d_biases = std::move(d_biases),
            .stream = stream,
        });
    }

    auto forward(const DeviceTensorConstView2f &inputs) -> KernelJob<DeviceTensorConstView2f>
    {
        const u32 M = outputs.extent<0>();
        const u32 N = outputs.extent<1>();
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        matmul_kernel<<<grid, block, 0, stream>>>(inputs, weights.const_view(), outputs.view());
        add_bias<<<grid, block, 0, stream>>>(inputs, biases.const_view(), outputs.view());
        return KernelJob<DeviceTensorConstView2f>{outputs.const_view(), stream};
    }

    auto backward(const DeviceTensorConstView2f &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>
    {
        const u32 M = d_inputs.extent<0>();
        const u32 N = d_inputs.extent<1>();
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        matmul_kernel<<<grid, block, 0, stream>>>(upstream_gradient, transposed(weights.const_view()), d_inputs.view());
        return KernelJob<DeviceTensorConstView2f>{d_inputs.const_view(), stream};
    }

    auto weight_gradients(const DeviceTensorConstView2f &inputs, const DeviceTensorConstView2f &upstream_gradient)
        -> KernelJob<std::tuple<DeviceTensorConstView2f, DeviceTensorConstView1f>>
    {
        const u32 M = d_inputs.extent<0>();
        const u32 N = d_inputs.extent<1>();
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        // NOTE: Run in separate streams?
        matmul_kernel<<<grid, block, 0, stream>>>(transposed(inputs), upstream_gradient, d_weights.view());

        const auto block_dim = dim3(256);
        const auto grid_dim = dim3(upstream_gradient.extents[0] + block_dim.x - 1 / block_dim.x);
        sum_rows<<<block_dim, grid_dim, 0, stream>>>(transposed(upstream_gradient), d_biases.view());
        return KernelJob<std::tuple<DeviceTensorConstView2f, DeviceTensorConstView1f>>{
            std::make_tuple(d_weights.const_view(), d_biases.const_view()), stream};
    }
    auto update() -> void;

    DeviceOwningTensor2f outputs;
    DeviceOwningTensor2f weights;
    DeviceOwningTensor1f biases;

    DeviceOwningTensor2f d_inputs;
    DeviceOwningTensor2f d_weights;
    DeviceOwningTensor1f d_biases;
    cudaStream_t stream;
};

}; // namespace vika

#ifdef VIKA_IMPLEMENTATION

namespace vika
{

__global__ auto sum_rows(DeviceTensorConstView2f matrix, DeviceTensorView1f out) -> void
{
    // NOTE: Reduce in blocks?
    const usize row = blockIdx.x * blockDim.x + threadIdx.x;
    const usize row_count = matrix.extents[0];

    if (row >= row_count)
    {
        return;
    }

    const usize col_count = matrix.extents[1];
    f32 sum = 0.0f;
    for (usize col = 0; col < col_count; ++col)
    {
        sum += matrix(row, col);
    }
    out[row] = sum;
}

__global__ auto add_bias(DeviceTensorConstView2f matrix, DeviceTensorConstView1f biases, DeviceTensorView2f out) -> void
{
    const usize sample_index = blockIdx.y * blockDim.y + threadIdx.y;
    const usize col = blockIdx.x * blockDim.x + threadIdx.x;

    const usize sample_count = matrix.extents[0];
    const usize bias_count = biases.extents[0];

    if (sample_index >= sample_count || col >= bias_count)
    {
        return;
    }

    out(sample_index, col) += biases[col];
}

__global__ auto matmul_kernel(DeviceTensorConstView2f a, DeviceTensorConstView2f b, DeviceTensorView2f out) -> void
{
    // NOTE: Multiply in tiles?
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
// - Sigmoid forward
// - Sigmoid backward
// - Sigmoid Layer
//
// - Flatten Forward
// - Flatten Backward
// - Flatten Layer
//
// - Dense weight update
//
// - Conv Forward
// - Conv Backward
// - Conv weight gradients
// - Conv weight update
// - Conv Layer
//
// - Maxpool Forward
// - Maxpool Backward
// - Maxpool Layer
//
// - Softmax Forward
// - Softmax Backward
// - Softmax Layer
//
// - CategoricalCrossEntropy Forward
// - CategoricalCrossEntropy Backward
// - CategoricalCrossEntropy Layer
//
// - Adam optimizer
// - Link layers
// - Pick device
// - Events/Async/Streams?
