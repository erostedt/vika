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
#include <utility>
#include <variant>
#include <vector>

using i32 = int32_t;
using u32 = uint32_t;
using f32 = float;
using usize = size_t;

#define panic(fmt, ...)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        fprintf(stderr, "Panicked: " fmt "\n", ##__VA_ARGS__);                                                         \
        exit(1);                                                                                                       \
    } while (0)

#define panic_if(expr, fmt, ...)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (expr)                                                                                                      \
        {                                                                                                              \
            panic(fmt, ##__VA_ARGS__);                                                                                 \
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

#define return_on_cuda_error(cudacall)                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto _err = (cudacall);                                                                                  \
        if (is_error(_err))                                                                                            \
        {                                                                                                              \
            return error(DeviceError(_err));                                                                           \
        }                                                                                                              \
    } while (0)

namespace vika
{

struct Void
{
};

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
        panic_if(is_error(), "called unwrap() on Error Result");
        return std::get<T>(storage);
    }

    auto unwrap() const & -> const T &
    {
        panic_if(is_error(), "called unwrap() on Error Result");
        return std::get<T>(storage);
    }

    auto unwrap() && -> T &&
    {
        panic_if(is_error(), "called unwrap() on Error Result");
        return std::move(std::get<T>(storage));
    }

    auto unwrap_error() & -> E &
    {
        panic_if(is_ok(), "called unwrap_error() on Ok Result");
        return std::get<E>(storage);
    }

    auto unwrap_error() const & -> const E &
    {
        panic_if(is_ok(), "called unwrap_error() on Ok Result");
        return std::get<E>(storage);
    }

    auto unwrap_error() && -> E &&
    {
        panic_if(is_ok(), "called unwrap_error() on Ok Result");
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

inline auto is_error(cudaError_t err) -> bool
{
    return err != cudaSuccess;
}

class DeviceError
{
  public:
    DeviceError(cudaError_t err) : _code(err)
    {
        panic_if(!is_error(err), "%s %s", "Not an error type: ", cudaGetErrorName(_code));
    }

    auto name() -> std::string
    {
        return cudaGetErrorName(_code);
    }

    auto string() -> std::string
    {
        return cudaGetErrorString(_code);
    }

    [[noreturn]] auto crash() -> void
    {
        panic("Crashed due to: [%s] %s", cudaGetErrorName(_code), cudaGetErrorString(_code));
    }

  private:
    cudaError_t _code;
};

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

template <usize Rank>
inline auto extents_equal(const usize e1[Rank], const usize e2[Rank]) -> bool
{
    return std::equal(e1, e1 + Rank, e2, e2 + Rank);
}

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
    static auto empty(const Extents &extents) -> Result<Self, DeviceError>
    {
        T *ptr = nullptr;
        const auto err = cudaMalloc(&ptr, vika::byte_count<T>(extents));
        if (err)
        {
            return error(DeviceError(err));
        }
        return ok(Self(ptr, extents));
    }

    static auto from(const std::vector<T> &data, const Extents &extents) -> Result<Self, DeviceError>
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
            return error(DeviceError(err));
        }
        return tensor;
    }

    template <usize R = Rank, typename = std::enable_if_t<R == 1>>
    static auto from(const std::vector<T> &data) -> Result<Self, DeviceError>
    {
        return from(data, {data.size()});
    }

    static auto empty_like(const Self &other) -> Result<Self, DeviceError>
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
auto copy(DeviceOwningTensor<T, Rank> src, HostTensor<T, Rank> &dst) -> Result<Void, DeviceError>
{
    panic_if(src.extents() != dst.extents(), "element_mismatch");
    const auto err = cudaMemcpy(dst.data(), src.data(), src.byte_count(), cudaMemcpyDeviceToHost);
    if (is_error(err))
    {
        return error(DeviceError(err));
    }
    return ok(Void{});
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto copy(const DeviceTensorView<const T, Rank> &src, HostTensor<T, Rank> &dst) -> Result<Void, DeviceError>
{
    const auto extents = to_extents<Rank>(src.extents);
    panic_if(dst.extents() != extents, "element_mismatch");
    const auto err = cudaMemcpy(dst.data(), src.data, src.byte_count(), cudaMemcpyDeviceToHost);
    if (is_error(err))
    {
        return error(DeviceError(err));
    }
    return ok(Void{});
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto copy(const HostTensor<T, Rank> &src, DeviceOwningTensor<T, Rank> &dst) -> Result<Void, DeviceError>
{
    panic_if(src.extents() != dst.extents(), "element_mismatch");
    const auto err = cudaMemcpy(dst.data(), src.data(), dst.byte_count(), cudaMemcpyHostToDevice);
    if (is_error(err))
    {
        return error(DeviceError(err));
    }
    return ok(Void{});
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto upload(const HostTensor<T, Rank> &src) -> Result<DeviceOwningTensor<T, Rank>, DeviceError>
{
    auto dst = DeviceOwningTensor<T, Rank>::empty(src.extents());
    if (dst.is_error())
    {
        return error(dst.unwrap_error());
    }

    const auto err = copy(src, dst.unwrap());
    if (err.is_error())
    {
        return error(err.unwrap_error());
    }
    return dst;
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto download(const DeviceTensorView<const T, Rank> &src) -> Result<HostTensor<T, Rank>, DeviceError>
{
    auto dst = HostTensor<T, Rank>::empty(to_extents<Rank>(src.extents));
    const auto err = copy(src, dst);
    if (err.is_error())
    {
        return error(err.unwrap_error());
    }
    return ok(dst);
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
auto download(const DeviceOwningTensor<T, Rank> &src) -> Result<HostTensor<T, Rank>, DeviceError>
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

template <usize Rank>
using DeviceOwningTensorf = DeviceOwningTensor<f32, Rank>;
using DeviceOwningTensor1f = DeviceOwningTensorf<1>;
using DeviceOwningTensor2f = DeviceOwningTensorf<2>;
using DeviceOwningTensor3f = DeviceOwningTensorf<3>;
using DeviceOwningTensor4f = DeviceOwningTensorf<4>;

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

// filters: [kH, kW, C_in, C_out], inputs: [N, H, W, C_in], out: [N, out_H, out_W, C_out]
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C_out), block: (bx, by, 1)
__global__ auto conv_forward(DeviceTensorConstView4f inputs, DeviceTensorConstView4f filters,
                             DeviceTensorConstView1f biases, DeviceTensorView4f out, usize stride, usize padding)
    -> void;

// filters: [kH, kW, C_in, C_out], inputs: [N, H, W, C_in], out: [N, out_H, out_W, C_out]
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C_out), block: (bx, by, 1)
__global__ auto conv_forward(DeviceTensorConstView4f inputs, DeviceTensorConstView4f filters,
                             DeviceTensorConstView1f biases, DeviceTensorView4f out, usize stride, usize padding)
    -> void;

// filters: [kH, kW, C_in, C_out], inputs: [N, H, W, C_in], out: [N, out_H, out_W, C_out]
__global__ auto conv_forward(DeviceTensorConstView4f inputs, DeviceTensorConstView4f filters,
                             DeviceTensorConstView1f biases, DeviceTensorView4f out, usize stride, usize padding)
    -> void;

__host__ __device__ inline auto sigmoid(f32 x) -> f32
{
    return 1.0f / (1.0f + std::exp(-x));
}

template <usize Rank>
__global__ auto sigmoid_forward(DeviceTensorConstViewf<Rank> a, DeviceTensorViewf<Rank> out) -> void
{
    const usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.element_count())
    {
        out[i] = sigmoid(a[i]);
    }
}

template <usize Rank>
__global__ auto sigmoid_backward(DeviceTensorConstViewf<Rank> a, DeviceTensorConstViewf<Rank> upstream_gradient,
                                 DeviceTensorViewf<Rank> out) -> void
{
    const usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.element_count())
    {

        out[i] = a[i] * (1.0 - a[i]) * upstream_gradient[i];
    }
}

__global__ auto add_bias(DeviceTensorConstView2f matrix, DeviceTensorConstView1f biases, DeviceTensorView2f out)
    -> void;

__global__ auto sum_rows(DeviceTensorConstView2f matrix, DeviceTensorView1f out) -> void;

template <typename T>
struct KernelJob
{
    auto wait() -> Result<T, DeviceError>
    {
        const auto err = cudaStreamSynchronize(stream);
        if (is_error(err))
        {
            return error(DeviceError(err));
        }
        return ok(value);
    }
    T value;
    cudaStream_t stream;
};

struct AdamParameters
{
    f32 learning_rate = 1e-1f;
    f32 beta1 = 0.9f;
    f32 beta2 = 0.999f;
    f32 epsilon = 1e-8f;
};

template <usize Rank>
__global__ auto adam_update(const AdamParameters parameters, f32 t, DeviceTensorConstViewf<Rank> d_weights,
                            DeviceTensorViewf<Rank> weights, DeviceTensorViewf<Rank> m_weights,
                            DeviceTensorViewf<Rank> v_weights) -> void;

struct DenseLayer
{
    auto static with_weights(usize batch_size, DeviceOwningTensor2f weights, DeviceOwningTensor1f biases)
        -> Result<DenseLayer, DeviceError>
    {
        const auto feature_count = weights.extent<0>();
        const auto neuron_count = weights.extent<1>();

        auto outputs = unwrap_or_return(DeviceOwningTensor2f::empty({batch_size, neuron_count}));

        auto d_inputs = unwrap_or_return(DeviceOwningTensor2f::empty({batch_size, feature_count}));
        auto d_outputs = unwrap_or_return(DeviceOwningTensor2f::empty({batch_size, neuron_count}));
        auto d_weights = unwrap_or_return(DeviceOwningTensor2f::empty_like(weights));
        auto d_biases = unwrap_or_return(DeviceOwningTensor1f::empty_like(biases));

        auto m_weights = unwrap_or_return(DeviceOwningTensor2f::empty_like(weights));
        auto v_weights = unwrap_or_return(DeviceOwningTensor2f::empty_like(weights));

        auto m_biases = unwrap_or_return(DeviceOwningTensor1f::empty_like(biases));
        auto v_biases = unwrap_or_return(DeviceOwningTensor1f::empty_like(biases));

        cudaStream_t stream;
        return_on_cuda_error(cudaStreamCreate(&stream));
        return ok<DenseLayer>({
            .outputs = std::move(outputs),
            .weights = std::move(weights),
            .biases = std::move(biases),
            .d_inputs = std::move(d_inputs),
            .d_weights = std::move(d_weights),
            .d_biases = std::move(d_biases),
            .m_weights = std::move(m_weights),
            .v_weights = std::move(v_weights),
            .m_biases = std::move(m_biases),
            .v_biases = std::move(v_biases),
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
        const auto grid_dim = dim3((upstream_gradient.extents[1] + block_dim.x - 1) / block_dim.x);
        sum_rows<<<block_dim, grid_dim, 0, stream>>>(transposed(upstream_gradient), d_biases.view());
        return KernelJob<std::tuple<DeviceTensorConstView2f, DeviceTensorConstView1f>>{
            std::make_tuple(d_weights.const_view(), d_biases.const_view()), stream};
    }

    auto update(DeviceTensorConstView2f d_weights, DeviceTensorConstView1f d_biases, const AdamParameters &parameters,
                usize t) -> KernelJob<std::monostate>
    {
        const auto weight_count = d_weights.element_count();
        const auto bias_count = d_biases.element_count();
        usize threads = 256;
        usize weight_blocks = (weight_count + threads - 1) / threads;
        usize bias_blocks = (bias_count + threads - 1) / threads;
        adam_update<2><<<weight_blocks, threads, 0, stream>>>(parameters, (f32)t, d_weights, weights.view(),
                                                              m_weights.view(), v_weights.view());
        adam_update<1><<<bias_blocks, threads, 0, stream>>>(parameters, (f32)t, d_biases, biases.view(),
                                                            m_biases.view(), v_biases.view());
        return KernelJob<std::monostate>{std::monostate{}, stream};
    }

    DeviceOwningTensor2f outputs;
    DeviceOwningTensor2f weights;
    DeviceOwningTensor1f biases;

    DeviceOwningTensor2f d_inputs;
    DeviceOwningTensor2f d_weights;
    DeviceOwningTensor1f d_biases;

    DeviceOwningTensor2f m_weights;
    DeviceOwningTensor2f v_weights;
    DeviceOwningTensor1f m_biases;
    DeviceOwningTensor1f v_biases;

    cudaStream_t stream;
};

struct SigmoidLayer
{
    static auto with_extents(const std::array<usize, 2> &extents) -> Result<SigmoidLayer, DeviceError>
    {
        auto outputs = unwrap_or_return(DeviceOwningTensor2f::empty(extents));
        auto d_inputs = unwrap_or_return(DeviceOwningTensor2f::empty(extents));

        cudaStream_t stream;
        return_on_cuda_error(cudaStreamCreate(&stream));
        return ok(SigmoidLayer{.outputs = std::move(outputs), .d_inputs = std::move(d_inputs), .stream = stream});
    }

    auto forward(const DeviceTensorConstView2f &inputs) -> KernelJob<DeviceTensorConstView2f>
    {
        panic_if(to_extents<2>(inputs.extents) != outputs.extents(), "MISMATCH");
        usize threads = 256;
        usize blocks = (inputs.element_count() + threads - 1) / threads;

        sigmoid_forward<2><<<blocks, threads, 0, stream>>>(inputs, outputs.view());
        return KernelJob<DeviceTensorConstView2f>{outputs.const_view(), stream};
    }

    auto backward(const DeviceTensorConstView2f &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>
    {
        panic_if(to_extents<2>(upstream_gradient.extents) != d_inputs.extents(), "MISMATCH");
        panic_if(d_inputs.extents() != outputs.extents(), "MISMATCH");
        usize threads = 256;
        usize blocks = (upstream_gradient.element_count() + threads - 1) / threads;

        sigmoid_backward<2><<<blocks, threads, 0, stream>>>(outputs.const_view(), upstream_gradient, d_inputs.view());
        return KernelJob<DeviceTensorConstView2f>{d_inputs.const_view(), stream};
    }

    DeviceOwningTensor2f outputs;
    DeviceOwningTensor2f d_inputs;

    cudaStream_t stream;
};

template <usize Rank, typename = std::enable_if_t<(Rank >= 2)>>
struct Flatten2DLayer
{
    static auto with_extents(const std::array<usize, Rank> &extents) -> Flatten2DLayer<Rank>
    {
        return {extents};
    }

    inline auto forward(DeviceTensorConstViewf<Rank> inputs) const -> DeviceTensorConstView2f
    {
        panic_if(to_extents<Rank>(inputs.extents) != extents, "INVALID EXTENTS");

        const auto batch = inputs.extents[0];
        const usize features =
            std::accumulate(inputs.extents + 1, inputs.extents + Rank, 1ul, std::multiplies<usize>{});
        return DeviceTensorConstView2f(inputs.data, {batch, features});
    }

    inline auto backward(DeviceTensorConstViewf<Rank> upstream_gradient) const -> DeviceTensorConstViewf<Rank>
    {
        panic_if(element_count(to_extents<Rank>(upstream_gradient.extents)) != element_count(extents),
                 "INVALID EXTENTS");
        return DeviceTensorConstViewf<Rank>(upstream_gradient.data, extents);
    }

    std::array<usize, Rank> extents;
};

struct Conv2DLayer
{
    // filters: [kH, kW, C_in, C_out]
    static auto with_weights(usize batch_size, usize input_height, usize input_width, DeviceOwningTensor4f filters,
                             DeviceOwningTensor1f biases, usize stride, usize padding)
        -> Result<Conv2DLayer, DeviceError>
    {
        const usize kH    = filters.extent<0>();
        const usize kW    = filters.extent<1>();
        const usize C_out = filters.extent<3>();

        const usize out_H = (input_height + 2 * padding - kH) / stride + 1;
        const usize out_W = (input_width + 2 * padding - kW) / stride + 1;

        auto outputs = unwrap_or_return(DeviceOwningTensor4f::empty({batch_size, out_H, out_W, C_out}));

        cudaStream_t stream;
        return_on_cuda_error(cudaStreamCreate(&stream));
        return ok(Conv2DLayer{
            .outputs = std::move(outputs),
            .filters = std::move(filters),
            .biases = std::move(biases),
            .stride = stride,
            .padding = padding,
            .stream = stream,
        });
    }

    auto forward(const DeviceTensorConstView4f &inputs) -> KernelJob<DeviceTensorConstView4f>
    {
        const usize N = outputs.extent<0>();
        const usize H_out = outputs.extent<1>();
        const usize W_out = outputs.extent<2>();
        const usize C_out = outputs.extent<3>();

        dim3 block(16, 16, 1);
        dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C_out);

        conv_forward<<<grid, block, 0, stream>>>(inputs, filters.const_view(), biases.const_view(), outputs.view(),
                                                 stride, padding);
        return KernelJob<DeviceTensorConstView4f>{outputs.const_view(), stream};
    }

    DeviceOwningTensor4f outputs;
    DeviceOwningTensor4f filters;
    DeviceOwningTensor1f biases;

    usize stride;
    usize padding;

    cudaStream_t stream;
};

template <usize Rank>
__global__ auto adam_update(const AdamParameters parameters, f32 t, DeviceTensorConstViewf<Rank> d_weights,
                            DeviceTensorViewf<Rank> weights, DeviceTensorViewf<Rank> m_weights,
                            DeviceTensorViewf<Rank> v_weights) -> void
{
    usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= weights.element_count())
    {
        return;
    }

    f32 m_hat_scale = 1.0 / (1.0 - std::pow(parameters.beta1, t));
    f32 v_hat_scale = 1.0 / (1.0 - std::pow(parameters.beta2, t));

    f32 g = d_weights[i];
    f32 m = m_weights[i];
    f32 v = v_weights[i];

    m = parameters.beta1 * m + (1.0 - parameters.beta1) * g;
    v = parameters.beta2 * v + (1.0 - parameters.beta2) * (g * g);

    m_weights[i] = m;
    v_weights[i] = v;

    f32 m_hat = m * m_hat_scale;
    f32 v_hat = v * v_hat_scale;

    if (v_hat < 0)
    {
        v_hat = 0.0f;
    }

    weights[i] -= parameters.learning_rate * m_hat / (std::sqrt(v_hat) + parameters.epsilon);
}

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

__global__ auto conv_forward(DeviceTensorConstView4f inputs, DeviceTensorConstView4f filters,
                             DeviceTensorConstView1f biases, DeviceTensorView4f out, usize stride, usize padding)
    -> void
{
    const usize ow = blockIdx.x * blockDim.x + threadIdx.x;
    const usize oh = blockIdx.y * blockDim.y + threadIdx.y;
    const usize oc = blockIdx.z % out.extents[3];
    const usize n = blockIdx.z / out.extents[3];

    if (ow >= out.extents[2] || oh >= out.extents[1] || n >= out.extents[0])
    {
        return;
    }

    const usize kH   = filters.extents[0];
    const usize kW   = filters.extents[1];
    const usize C_in = filters.extents[2];
    const usize H_in = inputs.extents[1];
    const usize W_in = inputs.extents[2];

    f32 sum = biases[oc];
    for (usize kh = 0; kh < kH; ++kh)
    {
        for (usize kw = 0; kw < kW; ++kw)
        {
            // TODO: think of a better name for ih_unpadded/iw_unpadded
            const usize ih_unpadded = oh * stride + kh;
            const usize iw_unpadded = ow * stride + kw;
            if (ih_unpadded >= padding && ih_unpadded - padding < H_in && iw_unpadded >= padding &&
                iw_unpadded - padding < W_in)
            {
                const usize ih = ih_unpadded - padding;
                const usize iw = iw_unpadded - padding;
                for (usize ic = 0; ic < C_in; ++ic)
                {
                    sum += inputs(n, ih, iw, ic) * filters(kh, kw, ic, oc);
                }
            }
        }
    }
    out(n, oh, ow, oc) = sum;
}

}; // namespace vika
#endif

// TODO (ecrt):
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
// - Link layers
// - Pick device?
// - Sequential model
// - Non-sequential builder api
// - XOR
// - mnist
// - unet
