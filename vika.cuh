#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <numeric>
#include <queue>
#include <string>
#include <tuple>
#include <type_traits>
#include <optional>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using i32 = int32_t;
using u32 = uint32_t;
using f32 = float;
using usize = size_t;

#define VIKA_MAX_RANK 6

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

// =============================================================================
// Generic Utilities
// =============================================================================

template <typename T, usize Capacity>
class FixedVector
{
    using Self = FixedVector<T, Capacity>;

  public:
    FixedVector() : m_size(0)
    {
    }

    FixedVector(std::initializer_list<T> init) : m_size(0)
    {
        for (const auto &v : init)
        {
            push_back(v);
        }
    }

    auto operator[](usize idx) -> T &
    {
        return m_data[idx];
    }

    auto operator[](usize idx) const -> const T &
    {
        return m_data[idx];
    }

    auto at(usize idx) -> T &
    {
        check_bounds(idx);
        return m_data[idx];
    }

    auto at(usize idx) const -> const T &
    {
        check_bounds(idx);
        return m_data[idx];
    }

    auto front() -> T &
    {
        return m_data[0];
    }

    auto front() const -> const T &
    {
        return m_data[0];
    }

    auto back() -> T &
    {
        return m_data[m_size - 1];
    }

    auto back() const -> const T &
    {
        return m_data[m_size - 1];
    }

    auto data() -> T *
    {
        return m_data;
    }

    auto data() const -> const T *
    {
        return m_data;
    }

    auto size() const -> usize
    {
        return m_size;
    }

    static constexpr auto capacity() -> usize
    {
        return Capacity;
    }

    auto empty() const -> bool
    {
        return m_size == 0;
    }

    auto full() const -> bool
    {
        return m_size == Capacity;
    }

    auto push_back(const T &value) -> void
    {
        check_not_full();
        m_data[m_size++] = value;
    }

    auto push_back(T &&value) -> void
    {
        check_not_full();
        m_data[m_size++] = std::move(value);
    }

    template <typename... Args>
    auto emplace_back(Args &&...args) -> T &
    {
        check_not_full();
        T *slot = &m_data[m_size++];
        new (slot) T(std::forward<Args>(args)...);
        return *slot;
    }

    auto pop_back() -> void
    {
        panic_if(m_size == 0, "Panicked: FixedVector::pop_back on empty vector\n");
        --m_size;
    }

    auto clear() -> void
    {
        m_size = 0;
    }

    auto begin() -> T *
    {
        return m_data;
    }

    auto end() -> T *
    {
        return m_data + m_size;
    }

    auto begin() const -> const T *
    {
        return m_data;
    }

    auto end() const -> const T *
    {
        return m_data + m_size;
    }

    auto cbegin() const -> const T *
    {
        return m_data;
    }

    auto cend() const -> const T *
    {
        return m_data + m_size;
    }

    auto rbegin() -> std::reverse_iterator<T *>
    {
        return std::reverse_iterator<T *>(end());
    }

    auto rend() -> std::reverse_iterator<T *>
    {
        return std::reverse_iterator<T *>(begin());
    }

    auto rbegin() const -> std::reverse_iterator<const T *>
    {
        return std::reverse_iterator<const T *>(end());
    }

    auto rend() const -> std::reverse_iterator<const T *>
    {
        return std::reverse_iterator<const T *>(begin());
    }

    auto operator==(const Self &other) const -> bool
    {
        return std::equal(cbegin(), cend(), other.cbegin(), other.cend());
    }

    auto operator!=(const Self &other) const -> bool
    {
        return !(*this == other);
    }

  private:
    T m_data[Capacity];
    usize m_size;

    auto check_bounds(usize idx) const -> void
    {
        panic_if(idx >= m_size, "FixedVector::at index %zu out of bounds (size=%zu)\n", idx, m_size);
    }

    auto check_not_full() const -> void
    {
        panic_if(m_size >= Capacity, "FixedVector is at capacity (%zu)\n", Capacity);
    }
};

using Extents = FixedVector<usize, VIKA_MAX_RANK>;

template <typename Node>
using AdjecencyGraph = std::unordered_map<Node, std::vector<Node>>;

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

template <typename Node>
auto topological_sort(const AdjecencyGraph<Node> &adj) -> Result<std::vector<Node>, std::string>
{
    std::unordered_map<Node, i32> indegree{};

    for (const auto &[u, neighbors] : adj)
    {
        if (!indegree.count(u))
        {
            indegree[u] = 0;
        }
        for (const Node &v : neighbors)
        {
            ++indegree[v];
        }
    }

    std::queue<Node> q{};
    for (const auto &[node, deg] : indegree)
    {
        if (deg == 0)
        {
            q.push(node);
        }
    }

    std::vector<Node> order;
    order.reserve(indegree.size());
    while (!q.empty())
    {
        Node u = std::move(q.front());
        q.pop();
        order.push_back(u);
        const auto it = adj.find(u);
        if (it == adj.end())
        {
            continue;
        }

        for (const Node &v : it->second)
        {
            --indegree[v];
            if (indegree[v] == 0)
            {
                q.push(v);
            }
        }
    }

    if (order.size() == indegree.size())
    {
        return ok(order);
    }
    return error(std::string("Cycle detected"));
}

// =============================================================================
// CUDA Error Handling
// =============================================================================

auto is_error(cudaError_t err) -> bool;

class DeviceError
{
  public:
    DeviceError(cudaError_t err);
    auto name() -> std::string;
    auto string() -> std::string;
    [[noreturn]] auto crash() -> void;

  private:
    cudaError_t _code;
};

// =============================================================================
// Tensor Helpers
// =============================================================================

auto to_extents(const usize *data, usize rank) -> Extents;
auto element_count(const Extents &extents) -> usize;

template <typename T>
inline auto byte_count(const Extents &extents) -> usize
{
    return element_count(extents) * sizeof(T);
}

// =============================================================================
// Host Tensors
// =============================================================================

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class HostTensor
{
    using Self = HostTensor<T>;
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

    auto operator()(usize r, usize c) -> T &
    {
        return _data[r * _extents[1] + c];
    }

    auto operator()(usize r, usize c) const -> const T &
    {
        return _data[r * _extents[1] + c];
    }

    auto operator()(usize n, usize h, usize w, usize c) -> T &
    {
        const auto height = _extents[1];
        const auto width = _extents[2];
        const auto channels = _extents[3];
        return _data[((n * height + h) * width + w) * channels + c];
    }

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

    auto extent(usize dimension) const -> usize
    {
        return _extents.at(dimension);
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
        return HostTensor(std::vector<T>(Self::size(extents), T{}), extents);
    }

    static auto empty(const Extents &extents) -> Self
    {
        return HostTensor(std::vector<T>(Self::size(extents)), extents);
    }

    static auto zero(usize element_count) -> Self
    {
        return Self::zero(Extents{element_count});
    }

    template <typename OtherType>
    static auto zero_like(const HostTensor<OtherType> &tensor) -> Self
    {
        return Self::zero(tensor.extents());
    }

    static auto from(std::initializer_list<T> data, const Extents &extents) -> Self
    {
        return copy_from(data, extents);
    }

    static auto copy_from(std::vector<T> data, const Extents &extents) -> Self
    {
        panic_if(std::size(data) != Self::size(extents), "Size mismatch");
        return HostTensor(std::move(data), extents);
    }

  private:
    template <typename OtherT, typename>
    friend class HostTensor;

    HostTensor(std::vector<T> &&data, const Extents &extents) : _data(std::move(data)), _extents(extents)
    {
        panic_if(_extents.empty(), "Empty extents!");
        panic_if(size(_extents) == 0, "Extent was 0");
        panic_if(_data.size() != size(_extents), "data/extents size mismatch");
    }

  private:
    std::vector<T> _data{};
    const Extents _extents{};
};

using HostTensorf = HostTensor<f32>;
using HostTensor1f = HostTensorf;
using HostTensor2f = HostTensorf;
using HostTensor3f = HostTensorf;
using HostTensor4f = HostTensorf;
using HostTensoru = HostTensor<u32>;
using HostTensor4u = HostTensor<u32>;
using Vectorf = HostTensor1f;
using Vectoru = HostTensor<u32>;
using Matrixf = HostTensor2f;

// =============================================================================
// Device Tensors
// =============================================================================

template <typename T>
struct DeviceDeleter
{
    auto operator()(T *ptr)
    {
        cudaFree(ptr);
    }
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
struct DeviceTensorView;

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
class DeviceOwningTensor
{
    using Self = DeviceOwningTensor<T>;

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

    static auto from(const std::vector<T> &data) -> Result<Self, DeviceError>
    {
        return from(data, {data.size()});
    }

    static auto empty_like(const Self &other) -> Result<Self, DeviceError>
    {
        return empty(other.extents());
    }

    static auto zero_like(const Self &other) -> Result<Self, DeviceError>
    {
        auto tensor = empty_like(other);
        if (tensor.is_error())
        {
            return tensor;
        }
        const auto err = cudaMemset(tensor.unwrap().data(), 0, tensor.unwrap().byte_count());
        if (is_error(err))
        {
            return error(DeviceError(err));
        }
        return tensor;
    }

    auto element_count() const -> usize
    {
        return vika::element_count(_extents);
    }

    auto byte_count() const -> usize
    {
        return vika::byte_count<T>(_extents);
    }

    auto extent(usize dimension) const -> usize
    {
        return _extents.at(dimension);
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

    auto view() -> DeviceTensorView<T>
    {
        return DeviceTensorView<T>(_data.get(), _extents, _extents.size());
    }

    auto const_view() const -> DeviceTensorView<const T>
    {
        return DeviceTensorView<const T>(_data.get(), _extents, _extents.size());
    }

  private:
    DeviceOwningTensor(T *data, const Extents &extents) : _data(data), _extents(extents)
    {
    }

  private:
    std::unique_ptr<T[], DeviceDeleter<T>> _data;
    Extents _extents;
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto copy(DeviceOwningTensor<T> src, HostTensor<T> &dst) -> Result<Void, DeviceError>
{
    panic_if(src.extents() != dst.extents(), "element_mismatch");
    const auto err = cudaMemcpy(dst.data(), src.data(), src.byte_count(), cudaMemcpyDeviceToHost);
    if (is_error(err))
    {
        return error(DeviceError(err));
    }
    return ok(Void{});
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto copy(const DeviceTensorView<const T> &src, HostTensor<T> &dst) -> Result<Void, DeviceError>
{
    const auto extents = to_extents(src.extents, src.rank);
    panic_if(dst.extents() != extents, "element_mismatch");
    const auto err = cudaMemcpy(dst.data(), src.data, src.byte_count(), cudaMemcpyDeviceToHost);
    if (is_error(err))
    {
        return error(DeviceError(err));
    }
    return ok(Void{});
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto copy(const HostTensor<T> &src, DeviceOwningTensor<T> &dst) -> Result<Void, DeviceError>
{
    panic_if(src.extents() != dst.extents(), "element_mismatch");
    const auto err = cudaMemcpy(dst.data(), src.data(), dst.byte_count(), cudaMemcpyHostToDevice);
    if (is_error(err))
    {
        return error(DeviceError(err));
    }
    return ok(Void{});
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto upload(const HostTensor<T> &src) -> Result<DeviceOwningTensor<T>, DeviceError>
{
    auto dst = DeviceOwningTensor<T>::empty(src.extents());
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

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto download(const DeviceTensorView<const T> &src) -> Result<HostTensor<T>, DeviceError>
{
    auto dst = HostTensor<T>::empty(to_extents(src.extents, src.rank));
    const auto err = copy(src, dst);
    if (err.is_error())
    {
        return error(err.unwrap_error());
    }
    return ok(dst);
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto download(const DeviceOwningTensor<T> &src) -> Result<HostTensor<T>, DeviceError>
{
    return download(src.const_view());
}

template <typename T, typename>
struct DeviceTensorView
{
    DeviceTensorView(T *data_, const Extents &extents_, usize rank_) : data(data_)
    {
        panic_if(rank_ >= VIKA_MAX_RANK, "Rank %zu larger or equal to VIKA_MAX_RANK %d", rank_, VIKA_MAX_RANK);
        std::copy(std::begin(extents_), std::end(extents_), extents);
        std::exclusive_scan(std::rbegin(extents_), std::rend(extents_), std::make_reverse_iterator(strides + rank_),
                            usize{1}, std::multiplies<usize>{});
        rank = rank_;
    }

    T *data = nullptr;
    usize extents[VIKA_MAX_RANK] = {};
    usize strides[VIKA_MAX_RANK] = {};
    usize rank = 0;

    __host__ __device__ inline usize element_count() const
    {
        usize count = 1;
        for (usize i = 0; i < rank; ++i)
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

    __host__ __device__ inline T &operator()(usize x)
    {
        return data[x];
    }

    __host__ __device__ inline const T &operator()(usize x) const
    {
        return data[x];
    }

    __host__ __device__ inline T &operator()(usize x, usize y)
    {
        return data[x * strides[0] + y * strides[1]];
    }

    __host__ __device__ inline const T &operator()(usize x, usize y) const
    {
        return data[x * strides[0] + y * strides[1]];
    }

    __host__ __device__ inline T &operator()(usize x, usize y, usize z)
    {
        return data[x * strides[0] + y * strides[1] + z * strides[2]];
    }

    __host__ __device__ inline const T &operator()(usize x, usize y, usize z) const
    {

        return data[x * strides[0] + y * strides[1] + z * strides[2]];
    }

    __host__ __device__ inline T &operator()(usize x, usize y, usize z, usize w)
    {
        return data[x * strides[0] + y * strides[1] + z * strides[2] + w * strides[3]];
    }

    __host__ __device__ inline const T &operator()(usize x, usize y, usize z, usize w) const
    {
        return data[x * strides[0] + y * strides[1] + z * strides[2] + w * strides[3]];
    }
};

using DeviceOwningTensorf = DeviceOwningTensor<f32>;
using DeviceOwningTensor1f = DeviceOwningTensorf;
using DeviceOwningTensor2f = DeviceOwningTensorf;
using DeviceOwningTensor3f = DeviceOwningTensorf;
using DeviceOwningTensor4f = DeviceOwningTensorf;
using DeviceOwningTensor4u = DeviceOwningTensor<u32>;

using DeviceTensorViewf = DeviceTensorView<f32>;
using DeviceTensorView1f = DeviceTensorViewf;
using DeviceTensorView2f = DeviceTensorViewf;
using DeviceTensorView3f = DeviceTensorViewf;
using DeviceTensorView4f = DeviceTensorViewf;
using DeviceTensorView4u = DeviceTensorView<u32>;

using DeviceTensorConstViewf = DeviceTensorView<const f32>;
using DeviceTensorConstView1f = DeviceTensorConstViewf;
using DeviceTensorConstView2f = DeviceTensorConstViewf;
using DeviceTensorConstView3f = DeviceTensorConstViewf;
using DeviceTensorConstView4f = DeviceTensorConstViewf;
using DeviceTensorConstView4u = DeviceTensorView<const u32>;

auto transposed(const DeviceTensorConstView2f &view) -> DeviceTensorConstView2f;

// =============================================================================
// Kernel Infrastructure
// =============================================================================

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

struct AdamState
{
    DeviceOwningTensorf m_weights;
    DeviceOwningTensorf v_weights;
    DeviceOwningTensorf m_biases;
    DeviceOwningTensorf v_biases;

    static auto create(const DeviceOwningTensorf &weights, const DeviceOwningTensorf &biases)
        -> Result<AdamState, DeviceError>;
};

__global__ auto matmul_kernel(DeviceTensorConstView2f a, DeviceTensorConstView2f b, DeviceTensorView2f out) -> void;

// filters: [kH, kW, C_in, C_out], inputs: [N, H, W, C_in], out: [N, out_H, out_W, C_out]
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C_out), block: (bx, by, 1)
__global__ auto conv_forward(DeviceTensorConstView4f inputs, DeviceTensorConstView4f filters,
                             DeviceTensorConstView1f biases, DeviceTensorView4f out, usize stride, usize padding)
    -> void;

// filters: [kH, kW, C_in, C_out], upstream: [N, out_H, out_W, C_out], d_inputs: [N, H, W, C_in]
// grid: (ceil(W_in/bx), ceil(H_in/by), N*C_in), block: (bx, by, 1)
__global__ auto conv_backward(DeviceTensorConstView4f upstream, DeviceTensorConstView4f filters,
                              DeviceTensorView4f d_inputs, usize stride, usize padding) -> void;

// inputs: [N, H, W, C_in], upstream: [N, out_H, out_W, C_out], d_filters: [kH, kW, C_in, C_out]
// 1D grid over all filter elements
__global__ auto conv_weight_gradients(DeviceTensorConstView4f inputs, DeviceTensorConstView4f upstream,
                                      DeviceTensorView4f d_filters, usize stride, usize padding) -> void;

// upstream: [N, out_H, out_W, C_out], d_biases: [C_out], 1D grid over C_out
__global__ auto conv_bias_gradients(DeviceTensorConstView4f upstream, DeviceTensorView1f d_biases) -> void;

// inputs: [N, H, W, C], out: [N, out_H, out_W, C], argmax: [N, out_H, out_W, C]
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C), block: (bx, by, 1)
__global__ auto maxpool_forward(DeviceTensorConstView4f inputs, DeviceTensorView4f out, DeviceTensorView4u argmax,
                                usize pool_h, usize pool_w, usize stride) -> void;

// upstream: [N, out_H, out_W, C], argmax: [N, out_H, out_W, C], d_inputs: [N, H, W, C]
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C), block: (bx, by, 1)
__global__ auto maxpool_backward(DeviceTensorConstView4f upstream, DeviceTensorConstView4u argmax,
                                 DeviceTensorView4f d_inputs) -> void;

__global__ auto uniform_tensor_kernel(DeviceTensorViewf tensor, u32 seed) -> void;
__global__ auto xavier_tensor_kernel(DeviceTensorViewf tensor, u32 seed, f32 limit) -> void;
__global__ auto sigmoid_forward(DeviceTensorConstViewf a, DeviceTensorViewf out) -> void;
__global__ auto sigmoid_backward(DeviceTensorConstViewf a, DeviceTensorConstViewf upstream_gradient,
                                 DeviceTensorViewf out) -> void;

__global__ auto adam_update(const AdamParameters parameters, f32 t, DeviceTensorConstViewf d_weights,
                            DeviceTensorViewf weights, DeviceTensorViewf m_weights, DeviceTensorViewf v_weights)
    -> void;

__global__ auto add_bias(DeviceTensorConstView2f matrix, DeviceTensorConstView1f biases, DeviceTensorView2f out)
    -> void;

__global__ auto sum_rows(DeviceTensorConstView2f matrix, DeviceTensorView1f out) -> void;

__global__ auto mse_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets, DeviceTensorView1f out)
    -> void;
__global__ auto mse_gradient_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets,
                                    DeviceTensorViewf out) -> void;

__host__ __device__ auto sigmoid(f32 x) -> f32;

auto uniform_tensor(DeviceTensorViewf tensor, u32 seed) -> KernelJob<Void>;
auto xavier_tensor(DeviceTensorViewf tensor, u32 seed, usize fan_in, usize fan_out) -> KernelJob<Void>;

// =============================================================================
// Layers
// =============================================================================

struct DenseLayer
{
    static auto with_weights(usize batch_size, DeviceOwningTensor2f weights, DeviceOwningTensor1f biases)
        -> Result<DenseLayer, DeviceError>;

    static auto randomized(usize batch_size, usize input_features, usize neuron_count, u32 seed)
        -> Result<DenseLayer, DeviceError>;

    auto forward(const DeviceTensorConstView2f &inputs) -> KernelJob<DeviceTensorConstView2f>;

    auto backward(const DeviceTensorConstView2f &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>;

    auto weight_gradients(const DeviceTensorConstView2f &inputs, const DeviceTensorConstView2f &upstream_gradient)
        -> KernelJob<std::tuple<DeviceTensorConstView2f, DeviceTensorConstView1f>>;

    auto update(DeviceTensorConstView2f d_weights, DeviceTensorConstView1f d_biases, AdamState &state,
                const AdamParameters &params, usize t) -> KernelJob<Void>;

    DeviceOwningTensor2f outputs;
    DeviceOwningTensor2f weights;
    DeviceOwningTensor1f biases;

    DeviceOwningTensor2f d_inputs;
    DeviceOwningTensor2f d_weights;
    DeviceOwningTensor1f d_biases;

    cudaStream_t stream;
};

struct SigmoidLayer
{
    static auto with_extents(const Extents &extents) -> Result<SigmoidLayer, DeviceError>;

    auto forward(const DeviceTensorConstViewf &inputs) -> KernelJob<DeviceTensorConstView2f>;

    auto backward(const DeviceTensorConstViewf &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>;

    DeviceOwningTensor2f outputs;
    DeviceOwningTensor2f d_inputs;

    cudaStream_t stream;
};

struct Flatten2DLayer
{
    // TODO: CHECK RANK SIZE
    static auto with_extents(const Extents &extents) -> Flatten2DLayer;

    auto forward(DeviceTensorConstViewf inputs) const -> KernelJob<DeviceTensorConstView2f>;

    auto backward(DeviceTensorConstView2f upstream_gradient) const -> KernelJob<DeviceTensorConstViewf>;

    Extents extents;
};

struct Conv2DLayer
{
    // filters: [kH, kW, C_in, C_out]
    static auto with_weights(usize batch_size, usize input_height, usize input_width, DeviceOwningTensor4f filters,
                             DeviceOwningTensor1f biases, usize stride, usize padding)
        -> Result<Conv2DLayer, DeviceError>;

    static auto randomized(usize batch_size, usize input_height, usize input_width, usize kH, usize kW, usize C_in,
                           usize C_out, usize stride, usize padding, u32 seed) -> Result<Conv2DLayer, DeviceError>;

    auto forward(const DeviceTensorConstView4f &inputs) -> KernelJob<DeviceTensorConstView4f>;

    auto backward(const DeviceTensorConstView4f &upstream) -> KernelJob<DeviceTensorConstView4f>;

    auto weight_gradients(const DeviceTensorConstView4f &inputs, const DeviceTensorConstView4f &upstream)
        -> KernelJob<std::tuple<DeviceTensorConstView4f, DeviceTensorConstView1f>>;

    auto update(const DeviceTensorConstView4f &d_filters_, const DeviceTensorConstView1f &d_biases_,
                AdamState &state, const AdamParameters &params, usize t) -> KernelJob<Void>;

    DeviceOwningTensor4f outputs;
    DeviceOwningTensor4f d_inputs;
    DeviceOwningTensor4f filters;
    DeviceOwningTensor1f biases;
    DeviceOwningTensor4f d_filters;
    DeviceOwningTensor1f d_biases;

    usize stride;
    usize padding;

    cudaStream_t stream;
};

struct MaxPool2DLayer
{
    static auto with_extents(usize batch_size, usize input_height, usize input_width, usize channels, usize pool_h,
                             usize pool_w, usize stride) -> Result<MaxPool2DLayer, DeviceError>;

    auto forward(const DeviceTensorConstView4f &inputs) -> KernelJob<DeviceTensorConstView4f>;

    auto backward(const DeviceTensorConstView4f &upstream) -> KernelJob<DeviceTensorConstView4f>;

    DeviceOwningTensor4f outputs;
    DeviceOwningTensor4u argmax;
    DeviceOwningTensor4f d_inputs;

    usize pool_h;
    usize pool_w;
    usize stride;

    cudaStream_t stream;
};

struct MSELoss
{
    static auto with_extents(const Extents &extents) -> Result<MSELoss, DeviceError>;

    auto forward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
        -> KernelJob<DeviceTensorConstView1f>;

    auto backward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
        -> KernelJob<DeviceTensorConstViewf>;

    DeviceOwningTensor1f loss;
    DeviceOwningTensorf d_inputs;

    cudaStream_t stream;
};

// =============================================================================
// Computation Graph
// =============================================================================

struct NodeId
{
    usize value;
};

struct InputSpec
{
};

struct SigmoidSpec
{
};

struct FlattenSpec
{
};

struct DenseSpec
{
    usize output_features;
    u32 seed;
};

struct Conv2DSpec
{
    usize kernel_height, kernel_width, channels_out, stride, padding;
    u32 seed;
};

struct MaxPool2DSpec
{
    usize pool_height, pool_width, stride;
};

using LayerSpec = std::variant<InputSpec, DenseSpec, Conv2DSpec, SigmoidSpec, MaxPool2DSpec, FlattenSpec>;

struct Node
{
    LayerSpec spec;
    Extents output_extents;
    std::vector<NodeId> inputs;
};

struct InputLayer
{
    auto forward(DeviceTensorConstViewf input) const -> KernelJob<DeviceTensorConstViewf>;
    auto backward(DeviceTensorConstViewf upstream) const -> KernelJob<DeviceTensorConstViewf>;
};

using LayerKind = std::variant<InputLayer, DenseLayer, SigmoidLayer, Conv2DLayer, MaxPool2DLayer, Flatten2DLayer>;

struct Layer
{
    LayerKind kind;
    bool is_frozen = false;

    static auto trainable(LayerKind kind) -> Layer { return {std::move(kind), false}; }
    static auto frozen(LayerKind kind) -> Layer { return {std::move(kind), true}; }

    auto freeze() -> void { is_frozen = true; }
    auto unfreeze() -> void { is_frozen = false; }
};

struct AdamOptimizer;

struct Model
{
    usize batch_size;
    std::vector<Layer> layers;
    std::vector<std::vector<NodeId>> layer_inputs;
    std::vector<NodeId> execution_order;
    NodeId input_node;
    NodeId output_node;

    std::unordered_map<usize, KernelJob<DeviceTensorConstViewf>> forward_jobs;
    std::unordered_map<usize, KernelJob<DeviceTensorConstViewf>> backward_jobs;

    auto forward(DeviceTensorConstViewf input) -> Result<DeviceTensorConstViewf, DeviceError>;
    auto backward(DeviceTensorConstViewf loss_grad) -> Result<Void, DeviceError>;
    auto step(AdamOptimizer &optimizer, usize t) -> Result<Void, DeviceError>;
};

struct AdamOptimizer
{
    AdamParameters params;
    std::unordered_map<usize, AdamState> states;

    static auto from_model(const Model &model, AdamParameters params) -> Result<AdamOptimizer, DeviceError>;
};

struct ComputationGraph
{
    usize batch_size;
    std::vector<Node> nodes;

    auto input(Extents spatial_extents) -> NodeId;
    auto dense(NodeId input, usize output_features, u32 seed) -> Result<NodeId, std::string>;
    auto sigmoid(NodeId input) -> Result<NodeId, std::string>;
    auto flatten(NodeId input) -> Result<NodeId, std::string>;
    auto conv2d(NodeId input, usize kernel_height, usize kernel_width, usize channels_out, usize stride, usize padding,
                u32 seed) -> Result<NodeId, std::string>;
    auto maxpool2d(NodeId input, usize pool_height, usize pool_width, usize stride) -> Result<NodeId, std::string>;
    auto compile(NodeId output) -> Result<Model, std::string>;
};

auto make_layer(const LayerSpec &spec, usize batch_size, const Extents &pred_extents) -> Result<Layer, std::string>;

auto update_layer(LayerKind &kind, DeviceTensorConstViewf forward_input, DeviceTensorConstViewf upstream,
                  AdamState &state, const AdamParameters &params, usize t) -> Result<Void, DeviceError>;


}; // namespace vika

#ifdef VIKA_IMPLEMENTATION

namespace vika
{

// =============================================================================
// CUDA Error Handling
// =============================================================================

auto is_error(cudaError_t err) -> bool
{
    return err != cudaSuccess;
}

DeviceError::DeviceError(cudaError_t err) : _code(err)
{
    panic_if(!is_error(err), "%s %s", "Not an error type: ", cudaGetErrorName(_code));
}

auto DeviceError::name() -> std::string
{
    return cudaGetErrorName(_code);
}

auto DeviceError::string() -> std::string
{
    return cudaGetErrorString(_code);
}

auto DeviceError::crash() -> void
{
    panic("Crashed due to: [%s] %s", cudaGetErrorName(_code), cudaGetErrorString(_code));
}

// =============================================================================
// Tensor Helpers
// =============================================================================

auto to_extents(const usize *data, usize rank) -> Extents
{
    panic_if(rank >= Extents::capacity(), "Rank %zu larger than VIKA_MAX_RANK %d", rank, VIKA_MAX_RANK);
    Extents extents{};
    for (usize i = 0; i < rank; ++i)
    {
        extents.push_back(data[i]);
    }
    return extents;
}

auto element_count(const Extents &extents) -> usize
{
    using namespace std;
    return accumulate(begin(extents), end(extents), 1ul, std::multiplies<>{});
}

// =============================================================================
// Device Tensors
// =============================================================================

auto transposed(const DeviceTensorConstView2f &view) -> DeviceTensorConstView2f
{
    auto transposed_view = view;
    std::swap(transposed_view.strides[0], transposed_view.strides[1]);
    std::swap(transposed_view.extents[0], transposed_view.extents[1]);
    return transposed_view;
}

// =============================================================================
// Kernel Infrastructure
// =============================================================================

__host__ __device__ auto sigmoid(f32 x) -> f32
{
    return 1.0f / (1.0f + std::exp(-x));
}

__device__ inline auto pcg_hash(u32 input) -> u32
{
    u32 state = input * 747796405u + 2891336453u;
    u32 word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ inline auto hash_to_float(u32 hash) -> f32
{
    return __uint_as_float((hash >> 9u) | 0x3f800000u) - 1.0f;
}

__device__ inline auto uniform_f32(u32 seed) -> f32
{
    return hash_to_float(pcg_hash(seed));
}

auto uniform_tensor(DeviceTensorViewf tensor, u32 seed) -> KernelJob<Void>
{
    const usize n = tensor.element_count();
    const usize threads = 256;
    uniform_tensor_kernel<<<(n + threads - 1) / threads, threads>>>(tensor, seed);
    return KernelJob<Void>{Void{}, 0};
}

auto xavier_tensor(DeviceTensorViewf tensor, u32 seed, usize fan_in, usize fan_out) -> KernelJob<Void>
{
    const f32 limit = std::sqrt(6.0f / (f32)(fan_in + fan_out));
    const usize n = tensor.element_count();
    const usize threads = 256;
    xavier_tensor_kernel<<<(n + threads - 1) / threads, threads>>>(tensor, seed, limit);
    return KernelJob<Void>{Void{}, 0};
}

// =============================================================================
// Layers
// =============================================================================

auto DenseLayer::with_weights(usize batch_size, DeviceOwningTensor2f weights, DeviceOwningTensor1f biases)
    -> Result<DenseLayer, DeviceError>
{
    const auto feature_count = weights.extent(0);
    const auto neuron_count = weights.extent(1);

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

auto DenseLayer::randomized(usize batch_size, usize input_features, usize neuron_count, u32 seed)
    -> Result<DenseLayer, DeviceError>
{
    auto weights = unwrap_or_return(DeviceOwningTensor2f::empty({input_features, neuron_count}));
    auto biases = unwrap_or_return(DeviceOwningTensor1f::from(std::vector<f32>(neuron_count, 0.0f)));

    unwrap_or_return(xavier_tensor(weights.view(), seed, input_features, neuron_count).wait());

    return with_weights(batch_size, std::move(weights), std::move(biases));
}

auto DenseLayer::forward(const DeviceTensorConstView2f &inputs) -> KernelJob<DeviceTensorConstView2f>
{
    const u32 M = outputs.extent(0);
    const u32 N = outputs.extent(1);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block, 0, stream>>>(inputs, weights.const_view(), outputs.view());
    add_bias<<<grid, block, 0, stream>>>(inputs, biases.const_view(), outputs.view());
    return KernelJob<DeviceTensorConstView2f>{outputs.const_view(), stream};
}

auto DenseLayer::backward(const DeviceTensorConstView2f &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>
{
    const u32 M = d_inputs.extent(0);
    const u32 N = d_inputs.extent(1);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    matmul_kernel<<<grid, block, 0, stream>>>(upstream_gradient, transposed(weights.const_view()), d_inputs.view());
    return KernelJob<DeviceTensorConstView2f>{d_inputs.const_view(), stream};
}

auto DenseLayer::weight_gradients(const DeviceTensorConstView2f &inputs,
                                  const DeviceTensorConstView2f &upstream_gradient)
    -> KernelJob<std::tuple<DeviceTensorConstView2f, DeviceTensorConstView1f>>
{
    const u32 M = d_inputs.extent(0);
    const u32 N = d_inputs.extent(1);
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

auto DenseLayer::update(DeviceTensorConstView2f d_weights, DeviceTensorConstView1f d_biases, AdamState &state,
                        const AdamParameters &params, usize t) -> KernelJob<Void>
{
    const usize threads = 256;
    const auto weight_count = d_weights.element_count();
    const auto bias_count = d_biases.element_count();
    adam_update<<<(weight_count + threads - 1) / threads, threads, 0, stream>>>(
        params, (f32)t, d_weights, weights.view(), state.m_weights.view(), state.v_weights.view());
    adam_update<<<(bias_count + threads - 1) / threads, threads, 0, stream>>>(
        params, (f32)t, d_biases, biases.view(), state.m_biases.view(), state.v_biases.view());
    return KernelJob<Void>{Void{}, stream};
}

auto SigmoidLayer::with_extents(const Extents &extents) -> Result<SigmoidLayer, DeviceError>
{
    auto outputs = unwrap_or_return(DeviceOwningTensor2f::empty(extents));
    auto d_inputs = unwrap_or_return(DeviceOwningTensor2f::empty(extents));

    cudaStream_t stream;
    return_on_cuda_error(cudaStreamCreate(&stream));
    return ok(SigmoidLayer{.outputs = std::move(outputs), .d_inputs = std::move(d_inputs), .stream = stream});
}

auto SigmoidLayer::forward(const DeviceTensorConstViewf &inputs) -> KernelJob<DeviceTensorConstView2f>
{
    panic_if(to_extents(inputs.extents, inputs.rank) != outputs.extents(), "MISMATCH");
    usize threads = 256;
    usize blocks = (inputs.element_count() + threads - 1) / threads;

    sigmoid_forward<<<blocks, threads, 0, stream>>>(inputs, outputs.view());
    return KernelJob<DeviceTensorConstView2f>{outputs.const_view(), stream};
}

auto SigmoidLayer::backward(const DeviceTensorConstViewf &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>
{
    panic_if(to_extents(upstream_gradient.extents, upstream_gradient.rank) != d_inputs.extents(), "MISMATCH");
    panic_if(d_inputs.extents() != outputs.extents(), "MISMATCH");
    usize threads = 256;
    usize blocks = (upstream_gradient.element_count() + threads - 1) / threads;

    sigmoid_backward<<<blocks, threads, 0, stream>>>(outputs.const_view(), upstream_gradient, d_inputs.view());
    return KernelJob<DeviceTensorConstView2f>{d_inputs.const_view(), stream};
}

auto Flatten2DLayer::with_extents(const Extents &extents) -> Flatten2DLayer
{
    return {extents};
}

auto Flatten2DLayer::forward(DeviceTensorConstViewf inputs) const -> KernelJob<DeviceTensorConstView2f>
{
    panic_if(to_extents(inputs.extents, inputs.rank) != extents, "INVALID EXTENTS");

    const auto batch = inputs.extents[0];
    const usize features =
        std::accumulate(inputs.extents + 1, inputs.extents + extents.size(), 1ul, std::multiplies<usize>{});
    return KernelJob<DeviceTensorConstView2f>{DeviceTensorConstView2f(inputs.data, {batch, features}, 2), 0};
}

auto InputLayer::forward(DeviceTensorConstViewf input) const -> KernelJob<DeviceTensorConstViewf>
{
    return KernelJob<DeviceTensorConstViewf>{input, 0};
}

auto InputLayer::backward(DeviceTensorConstViewf upstream) const -> KernelJob<DeviceTensorConstViewf>
{
    return KernelJob<DeviceTensorConstViewf>{upstream, 0};
}

auto Flatten2DLayer::backward(DeviceTensorConstView2f upstream_gradient) const -> KernelJob<DeviceTensorConstViewf>
{
    panic_if(element_count(to_extents(upstream_gradient.extents, upstream_gradient.rank)) != element_count(extents),
             "INVALID EXTENTS");
    return KernelJob<DeviceTensorConstViewf>{DeviceTensorConstViewf(upstream_gradient.data, extents, extents.size()), 0};
}

auto Conv2DLayer::with_weights(usize batch_size, usize input_height, usize input_width, DeviceOwningTensor4f filters,
                               DeviceOwningTensor1f biases, usize stride, usize padding)
    -> Result<Conv2DLayer, DeviceError>
{
    const usize kH = filters.extent(0);
    const usize kW = filters.extent(1);
    const usize C_out = filters.extent(3);

    const usize C_in = filters.extent(2);
    const usize out_H = (input_height + 2 * padding - kH) / stride + 1;
    const usize out_W = (input_width + 2 * padding - kW) / stride + 1;

    auto outputs = unwrap_or_return(DeviceOwningTensor4f::empty({batch_size, out_H, out_W, C_out}));
    auto d_inputs = unwrap_or_return(DeviceOwningTensor4f::empty({batch_size, input_height, input_width, C_in}));
    auto d_filters = unwrap_or_return(DeviceOwningTensor4f::empty_like(filters));
    auto d_biases = unwrap_or_return(DeviceOwningTensor1f::empty_like(biases));

    cudaStream_t stream;
    return_on_cuda_error(cudaStreamCreate(&stream));
    return ok(Conv2DLayer{
        .outputs = std::move(outputs),
        .d_inputs = std::move(d_inputs),
        .filters = std::move(filters),
        .biases = std::move(biases),
        .d_filters = std::move(d_filters),
        .d_biases = std::move(d_biases),
        .stride = stride,
        .padding = padding,
        .stream = stream,
    });
}

auto Conv2DLayer::randomized(usize batch_size, usize input_height, usize input_width, usize kH, usize kW, usize C_in,
                             usize C_out, usize stride, usize padding, u32 seed) -> Result<Conv2DLayer, DeviceError>
{
    auto filters = unwrap_or_return(DeviceOwningTensor4f::empty({kH, kW, C_in, C_out}));
    auto biases = unwrap_or_return(DeviceOwningTensor1f::from(std::vector<f32>(C_out, 0.0f)));

    unwrap_or_return(xavier_tensor(filters.view(), seed, kH * kW * C_in, kH * kW * C_out).wait());

    return with_weights(batch_size, input_height, input_width, std::move(filters), std::move(biases), stride, padding);
}

auto Conv2DLayer::forward(const DeviceTensorConstView4f &inputs) -> KernelJob<DeviceTensorConstView4f>
{
    const usize N = outputs.extent(0);
    const usize H_out = outputs.extent(1);
    const usize W_out = outputs.extent(2);
    const usize C_out = outputs.extent(3);

    dim3 block(16, 16, 1);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C_out);

    conv_forward<<<grid, block, 0, stream>>>(inputs, filters.const_view(), biases.const_view(), outputs.view(), stride,
                                             padding);
    return KernelJob<DeviceTensorConstView4f>{outputs.const_view(), stream};
}

auto Conv2DLayer::backward(const DeviceTensorConstView4f &upstream) -> KernelJob<DeviceTensorConstView4f>
{
    const usize N = d_inputs.extent(0);
    const usize H_in = d_inputs.extent(1);
    const usize W_in = d_inputs.extent(2);
    const usize C_in = d_inputs.extent(3);

    dim3 block(16, 16, 1);
    dim3 grid((W_in + block.x - 1) / block.x, (H_in + block.y - 1) / block.y, N * C_in);

    conv_backward<<<grid, block, 0, stream>>>(upstream, filters.const_view(), d_inputs.view(), stride, padding);
    return KernelJob<DeviceTensorConstView4f>{d_inputs.const_view(), stream};
}

auto Conv2DLayer::weight_gradients(const DeviceTensorConstView4f &inputs, const DeviceTensorConstView4f &upstream)
    -> KernelJob<std::tuple<DeviceTensorConstView4f, DeviceTensorConstView1f>>
{
    const usize filter_count = d_filters.element_count();
    const usize C_out = d_biases.element_count();
    const usize threads = 256;

    conv_weight_gradients<<<(filter_count + threads - 1) / threads, threads, 0, stream>>>(
        inputs, upstream, d_filters.view(), stride, padding);
    conv_bias_gradients<<<(C_out + threads - 1) / threads, threads, 0, stream>>>(upstream, d_biases.view());

    return KernelJob<std::tuple<DeviceTensorConstView4f, DeviceTensorConstView1f>>{
        std::make_tuple(d_filters.const_view(), d_biases.const_view()), stream};
}

auto Conv2DLayer::update(const DeviceTensorConstView4f &d_filters_, const DeviceTensorConstView1f &d_biases_,
                         AdamState &state, const AdamParameters &params, usize t) -> KernelJob<Void>
{
    const usize threads = 256;
    const usize filter_count = filters.element_count();
    const usize bias_count = biases.element_count();

    adam_update<<<(filter_count + threads - 1) / threads, threads, 0, stream>>>(
        params, (f32)t, d_filters_, filters.view(), state.m_weights.view(), state.v_weights.view());
    adam_update<<<(bias_count + threads - 1) / threads, threads, 0, stream>>>(
        params, (f32)t, d_biases_, biases.view(), state.m_biases.view(), state.v_biases.view());

    return KernelJob<Void>{Void{}, stream};
}

auto MaxPool2DLayer::with_extents(usize batch_size, usize input_height, usize input_width, usize channels, usize pool_h,
                                  usize pool_w, usize stride) -> Result<MaxPool2DLayer, DeviceError>
{
    const usize out_H = (input_height - pool_h) / stride + 1;
    const usize out_W = (input_width - pool_w) / stride + 1;

    auto outputs = unwrap_or_return(DeviceOwningTensor4f::empty({batch_size, out_H, out_W, channels}));
    auto argmax = unwrap_or_return((DeviceOwningTensor4u::empty({batch_size, out_H, out_W, channels})));
    auto d_inputs = unwrap_or_return(DeviceOwningTensor4f::empty({batch_size, input_height, input_width, channels}));

    cudaStream_t stream;
    return_on_cuda_error(cudaStreamCreate(&stream));
    return ok(MaxPool2DLayer{
        .outputs = std::move(outputs),
        .argmax = std::move(argmax),
        .d_inputs = std::move(d_inputs),
        .pool_h = pool_h,
        .pool_w = pool_w,
        .stride = stride,
        .stream = stream,
    });
}

auto MaxPool2DLayer::forward(const DeviceTensorConstView4f &inputs) -> KernelJob<DeviceTensorConstView4f>
{
    const usize N = outputs.extent(0);
    const usize H_out = outputs.extent(1);
    const usize W_out = outputs.extent(2);
    const usize C = outputs.extent(3);

    dim3 block(16, 16, 1);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C);

    maxpool_forward<<<grid, block, 0, stream>>>(inputs, outputs.view(), argmax.view(), pool_h, pool_w, stride);
    return KernelJob<DeviceTensorConstView4f>{outputs.const_view(), stream};
}

auto MaxPool2DLayer::backward(const DeviceTensorConstView4f &upstream) -> KernelJob<DeviceTensorConstView4f>
{
    const usize H_out = upstream.extents[1];
    const usize W_out = upstream.extents[2];
    const usize N = d_inputs.extent(0);
    const usize C = d_inputs.extent(3);

    cudaMemsetAsync(d_inputs.data(), 0, d_inputs.byte_count(), stream);

    dim3 block(16, 16, 1);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, N * C);

    maxpool_backward<<<grid, block, 0, stream>>>(upstream, argmax.const_view(), d_inputs.view());
    return KernelJob<DeviceTensorConstView4f>{d_inputs.const_view(), stream};
}

auto MSELoss::with_extents(const Extents &extents) -> Result<MSELoss, DeviceError>
{
    auto loss = unwrap_or_return(DeviceOwningTensor1f::empty({1}));
    auto d_inputs = unwrap_or_return(DeviceOwningTensorf::empty(extents));

    cudaStream_t stream;
    return_on_cuda_error(cudaStreamCreate(&stream));
    return ok(MSELoss{.loss = std::move(loss), .d_inputs = std::move(d_inputs), .stream = stream});
}

auto MSELoss::forward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
    -> KernelJob<DeviceTensorConstView1f>
{
    cudaMemsetAsync(loss.data(), 0, sizeof(f32), stream);

    const usize n = predictions.element_count();
    const usize threads = 256;
    mse_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(predictions, targets, loss.view());
    return KernelJob<DeviceTensorConstView1f>{loss.const_view(), stream};
}

auto MSELoss::backward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
    -> KernelJob<DeviceTensorConstViewf>
{
    const usize n = predictions.element_count();
    const usize threads = 256;
    mse_gradient_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(predictions, targets, d_inputs.view());
    return KernelJob<DeviceTensorConstViewf>{d_inputs.const_view(), stream};
}

// =============================================================================
// Computation Graph
// =============================================================================

auto ComputationGraph::input(Extents spatial_extents) -> NodeId
{
    Extents output_extents{};
    output_extents.push_back(batch_size);
    for (const auto dim : spatial_extents)
    {
        output_extents.push_back(dim);
    }
    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = InputSpec{},
        .output_extents = output_extents,
        .inputs = {},
    });
    return id;
}

auto ComputationGraph::dense(NodeId input, usize output_features, u32 seed) -> Result<NodeId, std::string>
{
    if (input.value >= nodes.size())
    {
        return error(std::string("dense: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() != 2)
    {
        return error(std::string("dense: input must be rank 2 [N, features]"));
    }

    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = DenseSpec{output_features, seed},
        .output_extents = {in_extents[0], output_features},
        .inputs = {input},
    });
    return ok(id);
}

auto ComputationGraph::sigmoid(NodeId input) -> Result<NodeId, std::string>
{
    if (input.value >= nodes.size())
    {
        return error(std::string("sigmoid: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = SigmoidSpec{},
        .output_extents = in_extents,
        .inputs = {input},
    });
    return ok(id);
}

auto ComputationGraph::flatten(NodeId input) -> Result<NodeId, std::string>
{
    if (input.value >= nodes.size())
    {
        return error(std::string("flatten: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() < 2)
    {
        return error(std::string("flatten: input must be at least rank 2"));
    }

    usize features = 1;
    for (usize i = 1; i < in_extents.size(); ++i)
    {
        features *= in_extents[i];
    }

    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = FlattenSpec{},
        .output_extents = {in_extents[0], features},
        .inputs = {input},
    });
    return ok(id);
}

auto ComputationGraph::conv2d(NodeId input, usize kernel_height, usize kernel_width, usize channels_out, usize stride,
                              usize padding, u32 seed) -> Result<NodeId, std::string>
{
    if (input.value >= nodes.size())
    {
        return error(std::string("conv2d: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() != 4)
    {
        return error(std::string("conv2d: input must be rank 4 [N, H, W, C]"));
    }

    const auto H = in_extents[1];
    const auto W = in_extents[2];

    if (H + 2 * padding < kernel_height)
    {
        return error(std::string("conv2d: kernel height exceeds padded input height"));
    }
    if (W + 2 * padding < kernel_width)
    {
        return error(std::string("conv2d: kernel width exceeds padded input width"));
    }

    const auto out_H = (H + 2 * padding - kernel_height) / stride + 1;
    const auto out_W = (W + 2 * padding - kernel_width) / stride + 1;

    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = Conv2DSpec{kernel_height, kernel_width, channels_out, stride, padding, seed},
        .output_extents = {in_extents[0], out_H, out_W, channels_out},
        .inputs = {input},
    });
    return ok(id);
}

auto ComputationGraph::maxpool2d(NodeId input, usize pool_height, usize pool_width, usize stride)
    -> Result<NodeId, std::string>
{
    if (input.value >= nodes.size())
    {
        return error(std::string("maxpool2d: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() != 4)
    {
        return error(std::string("maxpool2d: input must be rank 4 [N, H, W, C]"));
    }

    const auto H = in_extents[1];
    const auto W = in_extents[2];

    if (H < pool_height)
    {
        return error(std::string("maxpool2d: pool height exceeds input height"));
    }
    if (W < pool_width)
    {
        return error(std::string("maxpool2d: pool width exceeds input width"));
    }

    const auto out_H = (H - pool_height) / stride + 1;
    const auto out_W = (W - pool_width) / stride + 1;

    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = MaxPool2DSpec{pool_height, pool_width, stride},
        .output_extents = {in_extents[0], out_H, out_W, in_extents[3]},
        .inputs = {input},
    });
    return ok(id);
}

auto make_layer(const LayerSpec &spec, usize batch_size, const Extents &pred_extents) -> Result<Layer, std::string>
{
    return std::visit(
        [&](const auto &s) -> Result<Layer, std::string> {
            using T = std::decay_t<decltype(s)>;
            if constexpr (std::is_same_v<T, InputSpec>)
            {
                return ok(Layer::trainable(LayerKind{InputLayer{}}));
            }
            else if constexpr (std::is_same_v<T, DenseSpec>)
            {
                auto result = DenseLayer::randomized(batch_size, pred_extents[1], s.output_features, s.seed);
                if (result.is_error())
                {
                    return error(result.unwrap_error().string());
                }
                return ok(Layer::trainable(LayerKind{std::move(result.unwrap())}));
            }
            else if constexpr (std::is_same_v<T, SigmoidSpec>)
            {
                auto result = SigmoidLayer::with_extents(pred_extents);
                if (result.is_error())
                {
                    return error(result.unwrap_error().string());
                }
                return ok(Layer::trainable(LayerKind{std::move(result.unwrap())}));
            }
            else if constexpr (std::is_same_v<T, FlattenSpec>)
            {
                return ok(Layer::trainable(LayerKind{Flatten2DLayer::with_extents(pred_extents)}));
            }
            else if constexpr (std::is_same_v<T, Conv2DSpec>)
            {
                auto result = Conv2DLayer::randomized(batch_size, pred_extents[1], pred_extents[2], s.kernel_height,
                                                      s.kernel_width, pred_extents[3], s.channels_out, s.stride,
                                                      s.padding, s.seed);
                if (result.is_error())
                {
                    return error(result.unwrap_error().string());
                }
                return ok(Layer::trainable(LayerKind{std::move(result.unwrap())}));
            }
            else if constexpr (std::is_same_v<T, MaxPool2DSpec>)
            {
                auto result = MaxPool2DLayer::with_extents(batch_size, pred_extents[1], pred_extents[2],
                                                           pred_extents[3], s.pool_height, s.pool_width, s.stride);
                if (result.is_error())
                {
                    return error(result.unwrap_error().string());
                }
                return ok(Layer::trainable(LayerKind{std::move(result.unwrap())}));
            }
            else
            {
                static_assert(sizeof(T) == 0, "unhandled LayerSpec type in make_layer");
                return error(std::string("unreachable"));
            }
        },
        spec);
}

auto ComputationGraph::compile(NodeId output) -> Result<Model, std::string>
{
    if (output.value >= nodes.size())
    {
        return error(std::string("compile: invalid output NodeId"));
    }

    NodeId input_node{0};
    usize input_count = 0;
    for (usize i = 0; i < nodes.size(); ++i)
    {
        if (std::holds_alternative<InputSpec>(nodes[i].spec))
        {
            input_node = NodeId{i};
            ++input_count;
        }
    }
    if (input_count != 1)
    {
        return error(std::string("compile: graph must have exactly one input node"));
    }

    AdjecencyGraph<usize> adj{};
    for (usize i = 0; i < nodes.size(); ++i)
    {
        if (adj.find(i) == adj.end())
        {
            adj[i] = {};
        }
        for (const auto &pred : nodes[i].inputs)
        {
            adj[pred.value].push_back(i);
        }
    }

    auto sort_result = topological_sort(adj);
    if (sort_result.is_error())
    {
        return error(sort_result.unwrap_error());
    }

    const auto &topo_order = sort_result.unwrap();

    std::vector<Layer> layers;
    std::vector<std::vector<NodeId>> layer_inputs_result;
    layers.reserve(nodes.size());
    layer_inputs_result.reserve(nodes.size());

    for (const auto idx : topo_order)
    {
        const auto &node = nodes[idx];
        layer_inputs_result.push_back(node.inputs);

        const Extents pred_extents = node.inputs.empty() ? Extents{} : nodes[node.inputs[0].value].output_extents;

        auto layer_result = make_layer(node.spec, batch_size, pred_extents);
        if (layer_result.is_error())
        {
            return error(layer_result.unwrap_error());
        }
        layers.push_back(std::move(layer_result.unwrap()));
    }

    std::vector<NodeId> execution_order{};
    execution_order.reserve(nodes.size());
    for (const auto idx : topo_order)
    {
        execution_order.push_back(NodeId{idx});
    }

    return ok(Model{
        .batch_size = batch_size,
        .layers = std::move(layers),
        .layer_inputs = std::move(layer_inputs_result),
        .execution_order = std::move(execution_order),
        .input_node = input_node,
        .output_node = output,
    });
}

auto Model::forward(DeviceTensorConstViewf input) -> Result<DeviceTensorConstViewf, DeviceError>
{
    forward_jobs.clear();
    forward_jobs.reserve(layers.size());

    for (const auto node_id : execution_order)
    {
        const auto &preds = layer_inputs[node_id.value];
        panic_if(preds.size() > 1, "forward: multi-input nodes not yet supported");

        const auto pred_output = preds.empty() ? input : [&]() -> DeviceTensorConstViewf {
            auto result = forward_jobs.at(preds[0].value).wait();
            panic_if(result.is_error(), "forward: predecessor stream sync failed");
            return result.unwrap();
        }();

        auto job = std::visit(
            [&pred_output](auto &layer) -> KernelJob<DeviceTensorConstViewf> { return layer.forward(pred_output); },
            layers[node_id.value].kind);

        forward_jobs.emplace(node_id.value, std::move(job));
    }

    auto result = forward_jobs.at(output_node.value).wait();
    if (result.is_error())
    {
        return error(result.unwrap_error());
    }
    return ok(result.unwrap());
}

auto Model::backward(DeviceTensorConstViewf loss_grad) -> Result<Void, DeviceError>
{
    backward_jobs.clear();
    backward_jobs.reserve(layers.size());
    backward_jobs.emplace(output_node.value, KernelJob<DeviceTensorConstViewf>{loss_grad, 0});

    for (auto it = execution_order.rbegin(); it != execution_order.rend(); ++it)
    {
        const auto node_id = *it;
        const auto &preds = layer_inputs[node_id.value];
        if (preds.empty())
        {
            continue;
        }

        auto upstream_result = backward_jobs.at(node_id.value).wait();
        if (upstream_result.is_error())
        {
            return error(upstream_result.unwrap_error());
        }
        const auto upstream = upstream_result.unwrap();

        auto d_input_job = std::visit(
            [&upstream](auto &layer) -> KernelJob<DeviceTensorConstViewf> { return layer.backward(upstream); },
            layers[node_id.value].kind);

        panic_if(preds.size() > 1, "backward: multi-input nodes not yet supported");
        backward_jobs.emplace(preds[0].value, std::move(d_input_job));
    }

    return ok(Void{});
}

auto update_layer(LayerKind &kind, DeviceTensorConstViewf forward_input, DeviceTensorConstViewf upstream,
                  AdamState &state, const AdamParameters &params, usize t) -> Result<Void, DeviceError>
{
    return std::visit(
        [&](auto &l) -> Result<Void, DeviceError> {
            using T = std::decay_t<decltype(l)>;
            if constexpr (std::is_same_v<T, DenseLayer>)
            {
                auto wg_result = l.weight_gradients(forward_input, upstream).wait();
                if (wg_result.is_error()) { return error(wg_result.unwrap_error()); }
                auto [d_w, d_b] = wg_result.unwrap();
                auto up_result = l.update(d_w, d_b, state, params, t).wait();
                if (up_result.is_error()) { return error(up_result.unwrap_error()); }
            }
            else if constexpr (std::is_same_v<T, Conv2DLayer>)
            {
                auto wg_result = l.weight_gradients(forward_input, upstream).wait();
                if (wg_result.is_error()) { return error(wg_result.unwrap_error()); }
                auto [d_f, d_b] = wg_result.unwrap();
                auto up_result = l.update(d_f, d_b, state, params, t).wait();
                if (up_result.is_error()) { return error(up_result.unwrap_error()); }
            }
            return ok(Void{});
        },
        kind);
}

auto AdamState::create(const DeviceOwningTensorf &weights, const DeviceOwningTensorf &biases)
    -> Result<AdamState, DeviceError>
{
    auto m_w = unwrap_or_return(DeviceOwningTensorf::zero_like(weights));
    auto v_w = unwrap_or_return(DeviceOwningTensorf::zero_like(weights));
    auto m_b = unwrap_or_return(DeviceOwningTensorf::zero_like(biases));
    auto v_b = unwrap_or_return(DeviceOwningTensorf::zero_like(biases));
    return ok(AdamState{std::move(m_w), std::move(v_w), std::move(m_b), std::move(v_b)});
}

auto AdamOptimizer::from_model(const Model &model, AdamParameters params) -> Result<AdamOptimizer, DeviceError>
{
    AdamOptimizer optimizer{params, {}};

    for (const auto node_id : model.execution_order)
    {
        const auto &layer = model.layers[node_id.value];
        auto maybe_state = std::visit(
            [](const auto &l) -> std::optional<Result<AdamState, DeviceError>> {
                using T = std::decay_t<decltype(l)>;
                if constexpr (std::is_same_v<T, DenseLayer>)
                {
                    return AdamState::create(l.weights, l.biases);
                }
                else if constexpr (std::is_same_v<T, Conv2DLayer>)
                {
                    return AdamState::create(l.filters, l.biases);
                }
                else
                {
                    return std::nullopt;
                }
            },
            layer.kind);

        if (!maybe_state.has_value()) { continue; }
        if (maybe_state->is_error()) { return error(maybe_state->unwrap_error()); }
        optimizer.states.emplace(node_id.value, std::move(maybe_state->unwrap()));
    }

    return ok(std::move(optimizer));
}

auto Model::step(AdamOptimizer &optimizer, usize t) -> Result<Void, DeviceError>
{
    for (const auto node_id : execution_order)
    {
        auto &layer = layers[node_id.value];
        if (layer.is_frozen) { continue; }

        const auto &preds = layer_inputs[node_id.value];
        if (preds.empty()) { continue; }

        auto it = optimizer.states.find(node_id.value);
        if (it == optimizer.states.end()) { continue; }

        const auto forward_input = forward_jobs.at(preds[0].value).value;
        const auto upstream = backward_jobs.at(node_id.value).value;

        auto result = update_layer(layer.kind, forward_input, upstream, it->second, optimizer.params, t);
        if (result.is_error()) { return error(result.unwrap_error()); }
    }

    return ok(Void{});
}

// =============================================================================
// Kernels
// =============================================================================

__global__ auto uniform_tensor_kernel(DeviceTensorViewf tensor, u32 seed) -> void
{
    const usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tensor.element_count())
    {
        return;
    }
    tensor[i] = uniform_f32((u32)i ^ seed);
}

__global__ auto xavier_tensor_kernel(DeviceTensorViewf tensor, u32 seed, f32 limit) -> void
{
    const usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tensor.element_count())
    {
        return;
    }
    tensor[i] = (uniform_f32((u32)i ^ seed) * 2.0f - 1.0f) * limit;
}

__global__ auto mse_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets, DeviceTensorView1f out)
    -> void
{
    const usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= predictions.element_count())
    {
        return;
    }
    const f32 diff = predictions[i] - targets[i];
    atomicAdd(&out[0], diff * diff / (f32)predictions.element_count());
}

__global__ auto mse_gradient_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets,
                                    DeviceTensorViewf out) -> void
{
    const usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= predictions.element_count())
    {
        return;
    }
    out[i] = 2.0f * (predictions[i] - targets[i]) / (f32)predictions.element_count();
}

__global__ auto adam_update(const AdamParameters parameters, f32 t, DeviceTensorConstViewf d_weights,
                            DeviceTensorViewf weights, DeviceTensorViewf m_weights, DeviceTensorViewf v_weights) -> void
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

__global__ auto sigmoid_forward(DeviceTensorConstViewf a, DeviceTensorViewf out) -> void
{
    const usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.element_count())
    {
        out[i] = sigmoid(a[i]);
    }
}

__global__ auto sigmoid_backward(DeviceTensorConstViewf a, DeviceTensorConstViewf upstream_gradient,
                                 DeviceTensorViewf out) -> void
{
    const usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < a.element_count())
    {

        out[i] = a[i] * (1.0 - a[i]) * upstream_gradient[i];
    }
}

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

    const usize kH = filters.extents[0];
    const usize kW = filters.extents[1];
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

__global__ auto conv_backward(DeviceTensorConstView4f upstream, DeviceTensorConstView4f filters,
                              DeviceTensorView4f d_inputs, usize stride, usize padding) -> void
{
    const usize w = blockIdx.x * blockDim.x + threadIdx.x;
    const usize h = blockIdx.y * blockDim.y + threadIdx.y;
    const usize ic = blockIdx.z % d_inputs.extents[3];
    const usize n = blockIdx.z / d_inputs.extents[3];

    if (w >= d_inputs.extents[2] || h >= d_inputs.extents[1] || n >= d_inputs.extents[0])
    {
        return;
    }

    const usize kH = filters.extents[0];
    const usize kW = filters.extents[1];
    const usize C_out = filters.extents[3];
    const usize H_out = upstream.extents[1];
    const usize W_out = upstream.extents[2];

    f32 sum = 0.0f;
    for (usize kh = 0; kh < kH; ++kh)
    {
        for (usize kw = 0; kw < kW; ++kw)
        {
            // The output position that consumed inputs[n, h, w] via filter[kh, kw] satisfies:
            // oh * stride = h + padding - kh  (must be a non-negative multiple of stride)
            // ow * stride = w + padding - kw
            if (h + padding < kh || w + padding < kw)
            {
                continue;
            }
            const usize oh_unpadded = h + padding - kh;
            const usize ow_unpadded = w + padding - kw;
            if (oh_unpadded % stride != 0 || ow_unpadded % stride != 0)
            {
                continue;
            }
            const usize oh = oh_unpadded / stride;
            const usize ow = ow_unpadded / stride;
            if (oh >= H_out || ow >= W_out)
            {
                continue;
            }
            for (usize oc = 0; oc < C_out; ++oc)
            {
                sum += upstream(n, oh, ow, oc) * filters(kh, kw, ic, oc);
            }
        }
    }
    d_inputs(n, h, w, ic) = sum;
}

__global__ auto conv_weight_gradients(DeviceTensorConstView4f inputs, DeviceTensorConstView4f upstream,
                                      DeviceTensorView4f d_filters, usize stride, usize padding) -> void
{
    const usize idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_filters.element_count())
    {
        return;
    }

    const usize C_out = d_filters.extents[3];
    const usize C_in = d_filters.extents[2];
    const usize kW = d_filters.extents[1];

    const usize oc = idx % C_out;
    const usize ic = (idx / C_out) % C_in;
    const usize kw = (idx / (C_out * C_in)) % kW;
    const usize kh = idx / (C_out * C_in * kW);

    const usize N = inputs.extents[0];
    const usize H_in = inputs.extents[1];
    const usize W_in = inputs.extents[2];
    const usize H_out = upstream.extents[1];
    const usize W_out = upstream.extents[2];

    f32 sum = 0.0f;
    for (usize n = 0; n < N; ++n)
    {
        for (usize oh = 0; oh < H_out; ++oh)
        {
            const usize ih_unpadded = oh * stride + kh;
            if (ih_unpadded < padding || ih_unpadded - padding >= H_in)
            {
                continue;
            }
            const usize ih = ih_unpadded - padding;
            for (usize ow = 0; ow < W_out; ++ow)
            {
                const usize iw_unpadded = ow * stride + kw;
                if (iw_unpadded < padding || iw_unpadded - padding >= W_in)
                {
                    continue;
                }
                const usize iw = iw_unpadded - padding;
                sum += inputs(n, ih, iw, ic) * upstream(n, oh, ow, oc);
            }
        }
    }
    d_filters(kh, kw, ic, oc) = sum;
}

__global__ auto conv_bias_gradients(DeviceTensorConstView4f upstream, DeviceTensorView1f d_biases) -> void
{
    const usize oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= d_biases.extents[0])
    {
        return;
    }

    const usize N = upstream.extents[0];
    const usize H_out = upstream.extents[1];
    const usize W_out = upstream.extents[2];

    f32 sum = 0.0f;
    for (usize n = 0; n < N; ++n)
    {
        for (usize oh = 0; oh < H_out; ++oh)
        {
            for (usize ow = 0; ow < W_out; ++ow)
            {
                sum += upstream(n, oh, ow, oc);
            }
        }
    }
    d_biases[oc] = sum;
}

__global__ auto maxpool_forward(DeviceTensorConstView4f inputs, DeviceTensorView4f out, DeviceTensorView4u argmax,
                                usize pool_h, usize pool_w, usize stride) -> void
{
    const usize ow = blockIdx.x * blockDim.x + threadIdx.x;
    const usize oh = blockIdx.y * blockDim.y + threadIdx.y;
    const usize c = blockIdx.z % out.extents[3];
    const usize n = blockIdx.z / out.extents[3];

    if (ow >= out.extents[2] || oh >= out.extents[1] || n >= out.extents[0])
    {
        return;
    }

    const usize H_in = inputs.extents[1];
    const usize W_in = inputs.extents[2];

    f32 max_val = -INFINITY;
    u32 max_idx = 0;

    for (usize kh = 0; kh < pool_h; ++kh)
    {
        const usize ih = oh * stride + kh;
        if (ih >= H_in)
        {
            continue;
        }
        for (usize kw = 0; kw < pool_w; ++kw)
        {
            const usize iw = ow * stride + kw;
            if (iw >= W_in)
            {
                continue;
            }
            const f32 val = inputs(n, ih, iw, c);
            if (val > max_val)
            {
                max_val = val;
                max_idx = (u32)(ih * W_in + iw);
            }
        }
    }
    out(n, oh, ow, c) = max_val;
    argmax(n, oh, ow, c) = max_idx;
}

__global__ auto maxpool_backward(DeviceTensorConstView4f upstream, DeviceTensorConstView4u argmax,
                                 DeviceTensorView4f d_inputs) -> void
{
    const usize ow = blockIdx.x * blockDim.x + threadIdx.x;
    const usize oh = blockIdx.y * blockDim.y + threadIdx.y;
    const usize c = blockIdx.z % upstream.extents[3];
    const usize n = blockIdx.z / upstream.extents[3];

    if (ow >= upstream.extents[2] || oh >= upstream.extents[1] || n >= upstream.extents[0])
    {
        return;
    }

    const usize W_in = d_inputs.extents[2];
    const u32 idx = argmax(n, oh, ow, c);
    const usize ih = idx / W_in;
    const usize iw = idx % W_in;

    atomicAdd(&d_inputs(n, ih, iw, c), upstream(n, oh, ow, c));
}

}; // namespace vika
#endif

// TODO (ecrt):
// - Softmax Forward
// - Softmax Backward
// - Softmax Layer
//
// - CategoricalCrossEntropy Forward
// - CategoricalCrossEntropy Backward
// - CategoricalCrossEntropy Layer
//
// - Upsampling Forward
// - Upsampling Backward
// - Upsampling Layer
//
// - Pick device?
// - Sequential model
// - Non-sequential builder api
// - unet
