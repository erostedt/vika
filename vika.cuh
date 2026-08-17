#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#ifndef VIKA_MAX_RANK
#define VIKA_MAX_RANK 6
#endif

#ifndef VIKA_MAX_ERROR_MESSAGE
#define VIKA_MAX_ERROR_MESSAGE 192
#endif

#define VIKA_PANIC(fmt, ...)                                                                                           \
    do                                                                                                                 \
    {                                                                                                                  \
        fprintf(stderr, "Panicked: " fmt "\n", ##__VA_ARGS__);                                                         \
        exit(1);                                                                                                       \
    } while (0)

#define VIKA_PANIC_IF(expr, fmt, ...)                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        if (expr)                                                                                                      \
        {                                                                                                              \
            VIKA_PANIC(fmt, ##__VA_ARGS__);                                                                            \
        }                                                                                                              \
    } while (0)

// Works for any enclosing return type with an implicit Err<Error> constructor - both
// Result<T, Error> and KernelJob<T> qualify, so this covers ordinary Result-returning
// functions as well as forward()/backward()-style functions returning a KernelJob. No T
// argument needed: `return error(...)` converts via the return statement itself, to
// whatever the enclosing function actually declared.
#define UNWRAP_OR_RETURN(expr)                                                                                         \
    ({                                                                                                                 \
        auto _res = (expr);                                                                                            \
        if (_res.is_error())                                                                                           \
        {                                                                                                              \
            return error(_res.unwrap_error());                                                                         \
        }                                                                                                              \
        std::move(_res.unwrap());                                                                                      \
    })

#define VIKA_RETURN_ON_CUDA_ERROR(cudacall)                                                                            \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto _err = (cudacall);                                                                                  \
        if (is_error(_err))                                                                                            \
        {                                                                                                              \
            return error(VIKA_DEVICE_ERROR(_err));                                                                     \
        }                                                                                                              \
    } while (0)

namespace vika
{

using i32 = int32_t;
using u32 = uint32_t;
using f32 = float;
using usize = size_t;

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
        VIKA_PANIC_IF(m_size == 0, "Panicked: FixedVector::pop_back on empty vector\n");
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
        VIKA_PANIC_IF(idx >= m_size, "FixedVector::at index %zu out of bounds (size=%zu)\n", idx, m_size);
    }

    auto check_not_full() const -> void
    {
        VIKA_PANIC_IF(m_size >= Capacity, "FixedVector is at capacity (%zu)\n", Capacity);
    }
};

using Extents = FixedVector<usize, VIKA_MAX_RANK>;

template <typename Node>
using AdjecencyGraph = std::unordered_map<Node, std::vector<Node>>;

template <typename Node>
using MinHeap = std::priority_queue<Node, std::vector<Node>, std::greater<Node>>;

struct Void
{
};

template <typename T>
struct Ok
{
    T value;
};

// Tag type paired with Ok<T>, produced by error() below. Distinct from the Error class
// in the Error Handling section, which is the actual error payload.
template <typename E>
struct Err
{
    E value;
};

template <typename T>
auto ok(T value) -> Ok<std::decay_t<T>>
{
    return Ok<std::decay_t<T>>{std::move(value)};
}

template <typename E>
auto error(E value) -> Err<std::decay_t<E>>
{
    return Err<std::decay_t<E>>{std::move(value)};
}

template <typename T, typename E>
class [[nodiscard]] Result
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

    Result(Err<E> error) : storage(std::move(error.value))
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
        VIKA_PANIC_IF(is_error(), "called unwrap() on Err Result");
        return std::get<T>(storage);
    }

    auto unwrap() const & -> const T &
    {
        VIKA_PANIC_IF(is_error(), "called unwrap() on Err Result");
        return std::get<T>(storage);
    }

    // Returns by value: an rvalue Result is usually a temporary, so handing back a
    // reference into its storage would dangle as soon as the full expression ends.
    auto unwrap() && -> T
    {
        VIKA_PANIC_IF(is_error(), "called unwrap() on Err Result");
        return std::move(std::get<T>(storage));
    }

    auto unwrap_error() & -> E &
    {
        VIKA_PANIC_IF(is_ok(), "called unwrap_error() on Ok Result");
        return std::get<E>(storage);
    }

    auto unwrap_error() const & -> const E &
    {
        VIKA_PANIC_IF(is_ok(), "called unwrap_error() on Ok Result");
        return std::get<E>(storage);
    }

    auto unwrap_error() && -> E
    {
        VIKA_PANIC_IF(is_ok(), "called unwrap_error() on Ok Result");
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

    // Transforms the value, forwarding an error untouched:
    //   Result<A, E>::map(A -> B) -> Result<B, E>
    // The callback must return a value; use Void for "nothing meaningful".
    // Consuming, because T is usually move-only here (tensors, layers, models).
    template <typename F>
    auto map(F &&f) && -> Result<std::invoke_result_t<F, T &&>, E>
    {
        using U = std::invoke_result_t<F, T &&>;
        if (is_error())
        {
            return Result<U, E>(Err<E>{std::move(std::get<E>(storage))});
        }
        return Result<U, E>(Ok<U>{std::forward<F>(f)(std::move(std::get<T>(storage)))});
    }

    // Chains a step that can itself fail, flattening the two Results into one:
    //   Result<A, E>::and_then(A -> Result<B, E>) -> Result<B, E>
    template <typename F>
    auto and_then(F &&f) && -> std::invoke_result_t<F, T &&>
    {
        using R = std::invoke_result_t<F, T &&>;
        static_assert(std::is_same_v<typename R::error_type, E>,
                      "and_then callback must return a Result with the same error type");

        if (is_error())
        {
            return R(Err<E>{std::move(std::get<E>(storage))});
        }
        return std::forward<F>(f)(std::move(std::get<T>(storage)));
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

// =============================================================================
// Error Handling
// =============================================================================

auto is_error(cudaError_t err) -> bool;

enum class ErrorKind
{
    Device,      // a CUDA runtime call or kernel launch failed
    Shape,       // extents, rank, or layout disagreement
    Graph,       // malformed computation graph
    Unsupported, // a valid request vika cannot serve yet
};

auto error_kind_name(ErrorKind kind) -> const char *;

// A single flat error type for every fallible vika operation. Deliberately allocation
// free and trivially copyable: it travels through Result and KernelJob by value, and one
// of the failures it reports is device allocation running out of memory.
class Error
{
  public:
    // Prefer the VIKA_*_ERROR macros below, which capture the originating file and line.
    static auto make(ErrorKind kind, const char *file, i32 line, const char *fmt, ...) -> Error;
    static auto from_cuda(cudaError_t code, const char *file, i32 line) -> Error;

    auto kind() const -> ErrorKind
    {
        return _kind;
    }

    // cudaSuccess unless kind() == ErrorKind::Device
    auto code() const -> cudaError_t
    {
        return _code;
    }

    auto message() const -> const char *
    {
        return _message;
    }

    auto file() const -> const char *
    {
        return _file;
    }

    auto line() const -> i32
    {
        return _line;
    }

    auto describe() const -> std::string;
    [[noreturn]] auto crash() const -> void;

  private:
    ErrorKind _kind = ErrorKind::Device;
    cudaError_t _code = cudaSuccess;
    const char *_file = "";
    i32 _line = 0;
    char _message[VIKA_MAX_ERROR_MESSAGE] = {};
};

#define VIKA_ERROR(kind, ...) ::vika::Error::make((kind), __FILE__, __LINE__, __VA_ARGS__)
#define VIKA_SHAPE_ERROR(...) VIKA_ERROR(::vika::ErrorKind::Shape, __VA_ARGS__)
#define VIKA_GRAPH_ERROR(...) VIKA_ERROR(::vika::ErrorKind::Graph, __VA_ARGS__)
#define VIKA_UNSUPPORTED_ERROR(...) VIKA_ERROR(::vika::ErrorKind::Unsupported, __VA_ARGS__)
#define VIKA_DEVICE_ERROR(code) ::vika::Error::from_cuda((code), __FILE__, __LINE__)

template <typename Node>
auto topological_sort(const AdjecencyGraph<Node> &adj) -> Result<std::vector<Node>, Error>
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

    // A min-heap rather than a plain FIFO queue: seeding/ties are otherwise resolved by
    // unordered_map's bucket order, which is a deterministic function of the hash table but not
    // of anything a caller controls (declaration order, NodeId value, ...). Always advancing the
    // smallest available node gives one canonical order regardless of where ties occur.
    MinHeap<Node> heap{};
    for (const auto &[node, deg] : indegree)
    {
        if (deg == 0)
        {
            heap.push(node);
        }
    }

    std::vector<Node> order;
    order.reserve(indegree.size());
    while (!heap.empty())
    {
        Node u = std::move(heap.top());
        heap.pop();
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
                heap.push(v);
            }
        }
    }

    if (order.size() == indegree.size())
    {
        return ok(order);
    }
    return error(VIKA_GRAPH_ERROR("Cycle detected"));
}

// =============================================================================
// Tensor Helpers
// =============================================================================

auto element_count(const Extents &extents) -> usize;

template <typename T>
inline auto byte_count(const Extents &extents) -> usize
{
    return element_count(extents) * sizeof(T);
}

// Overflow-reporting counterparts, for the paths that turn extents into an allocation
// size. Wrapping there is worse than failing: the product can land on a small value (or
// exactly zero), cudaMalloc happily succeeds, and every later kernel writes out of
// bounds with nothing reporting an error. The unchecked versions above stay for extents
// that were already validated when their tensor was created.
auto checked_element_count(const Extents &extents) -> Result<usize, Error>;

template <typename T>
inline auto checked_byte_count(const Extents &extents) -> Result<usize, Error>
{
    auto count = checked_element_count(extents);
    if (count.is_error())
    {
        return error(count.unwrap_error());
    }

    const usize elements = count.unwrap();
    if (elements > std::numeric_limits<usize>::max() / sizeof(T))
    {
        return error(VIKA_SHAPE_ERROR("byte count overflows usize: %zu elements of %zu bytes", elements, sizeof(T)));
    }
    return ok(elements * sizeof(T));
}

// Output extent of one spatial dimension for a sliding window. Convolution and pooling
// share this geometry; pooling is simply the padding == 0 case.
// Requires stride > 0 and input + 2 * padding >= window.
inline constexpr auto window_output_extent(usize input, usize window, usize stride, usize padding) -> usize
{
    return (input + 2 * padding - window) / stride + 1;
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

    // Element count described by extents, rejecting the three ways extents can be
    // unusable: empty, containing a zero, or overflowing usize when multiplied out.
    // Every factory below funnels through this, so the private constructor can assume
    // both the extents and the data length are sound.
    static auto checked_size(const Extents &extents) -> Result<usize, Error>
    {
        if (extents.empty())
        {
            return error(VIKA_SHAPE_ERROR("host tensor extents are empty"));
        }

        const usize count = UNWRAP_OR_RETURN(checked_element_count(extents));
        if (count == 0)
        {
            return error(VIKA_SHAPE_ERROR("host tensor extents contain a zero extent"));
        }
        return ok(count);
    }

    static auto zero(const Extents &extents) -> Result<Self, Error>
    {
        const usize count = UNWRAP_OR_RETURN(Self::checked_size(extents));
        return ok(HostTensor(std::vector<T>(count, T{}), extents));
    }

    static auto empty(const Extents &extents) -> Result<Self, Error>
    {
        const usize count = UNWRAP_OR_RETURN(Self::checked_size(extents));
        return ok(HostTensor(std::vector<T>(count), extents));
    }

    static auto zero(usize element_count) -> Result<Self, Error>
    {
        return Self::zero(Extents{element_count});
    }

    template <typename OtherType>
    static auto zero_like(const HostTensor<OtherType> &tensor) -> Result<Self, Error>
    {
        return Self::zero(tensor.extents());
    }

    static auto from(std::initializer_list<T> data, const Extents &extents) -> Result<Self, Error>
    {
        return copy_from(data, extents);
    }

    static auto copy_from(std::vector<T> data, const Extents &extents) -> Result<Self, Error>
    {
        const usize count = UNWRAP_OR_RETURN(Self::checked_size(extents));
        if (std::size(data) != count)
        {
            return error(VIKA_SHAPE_ERROR("host tensor data holds %zu elements but extents describe %zu",
                                          std::size(data), count));
        }
        return ok(HostTensor(std::move(data), extents));
    }

  private:
    template <typename OtherT, typename>
    friend class HostTensor;

    HostTensor(std::vector<T> &&data, const Extents &extents) : _data(std::move(data)), _extents(extents)
    {
        VIKA_PANIC_IF(_extents.empty(), "Empty extents!");
        VIKA_PANIC_IF(size(_extents) == 0, "Extent was 0");
        VIKA_PANIC_IF(_data.size() != size(_extents), "data/extents size mismatch");
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
    static auto empty(const Extents &extents) -> Result<Self, Error>
    {
        const usize bytes = UNWRAP_OR_RETURN(vika::checked_byte_count<T>(extents));

        T *ptr = nullptr;
        const auto err = cudaMalloc(&ptr, bytes);
        if (err)
        {
            return error(VIKA_DEVICE_ERROR(err));
        }
        return ok(Self(ptr, extents));
    }

    static auto from(const std::vector<T> &data, const Extents &extents) -> Result<Self, Error>
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
            return error(VIKA_DEVICE_ERROR(err));
        }
        return tensor;
    }

    static auto from(const std::vector<T> &data) -> Result<Self, Error>
    {
        return from(data, {data.size()});
    }

    static auto empty_like(const Self &other) -> Result<Self, Error>
    {
        return empty(other.extents());
    }

    static auto zero(const Extents &extents) -> Result<Self, Error>
    {
        auto tensor = empty(extents);
        if (tensor.is_error())
        {
            return tensor;
        }
        const auto err = cudaMemset(tensor.unwrap().data(), 0, tensor.unwrap().byte_count());
        if (is_error(err))
        {
            return error(VIKA_DEVICE_ERROR(err));
        }
        return tensor;
    }

    static auto zero_like(const Self &other) -> Result<Self, Error>
    {
        return zero(other.extents());
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
        return DeviceTensorView<T>(_data.get(), _extents);
    }

    auto const_view() const -> DeviceTensorView<const T>
    {
        return DeviceTensorView<const T>(_data.get(), _extents);
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
auto copy(const DeviceOwningTensor<T> &src, HostTensor<T> &dst) -> Result<Void, Error>
{
    if (src.extents() != dst.extents())
    {
        return error(VIKA_SHAPE_ERROR("copy device -> host: source holds %zu elements, destination holds %zu",
                                      src.element_count(), dst.size()));
    }

    const auto err = cudaMemcpy(dst.data(), src.data(), src.byte_count(), cudaMemcpyDeviceToHost);
    if (is_error(err))
    {
        return error(VIKA_DEVICE_ERROR(err));
    }
    return ok(Void{});
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto copy(const DeviceTensorView<const T> &src, HostTensor<T> &dst) -> Result<Void, Error>
{
    if (dst.extents() != src.to_extents())
    {
        return error(VIKA_SHAPE_ERROR("copy view -> host: source holds %zu elements, destination holds %zu",
                                      src.element_count(), dst.size()));
    }

    const auto err = cudaMemcpy(dst.data(), src.data, src.byte_count(), cudaMemcpyDeviceToHost);
    if (is_error(err))
    {
        return error(VIKA_DEVICE_ERROR(err));
    }
    return ok(Void{});
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto copy(const HostTensor<T> &src, DeviceOwningTensor<T> &dst) -> Result<Void, Error>
{
    if (src.extents() != dst.extents())
    {
        return error(VIKA_SHAPE_ERROR("copy host -> device: source holds %zu elements, destination holds %zu",
                                      src.size(), dst.element_count()));
    }

    const auto err = cudaMemcpy(dst.data(), src.data(), dst.byte_count(), cudaMemcpyHostToDevice);
    if (is_error(err))
    {
        return error(VIKA_DEVICE_ERROR(err));
    }
    return ok(Void{});
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto upload(const HostTensor<T> &src) -> Result<DeviceOwningTensor<T>, Error>
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
auto download(const DeviceTensorView<const T> &src) -> Result<HostTensor<T>, Error>
{
    auto dst = UNWRAP_OR_RETURN(HostTensor<T>::empty(src.to_extents()));
    const auto err = copy(src, dst);
    if (err.is_error())
    {
        return error(err.unwrap_error());
    }
    return ok(std::move(dst));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto download(const DeviceOwningTensor<T> &src) -> Result<HostTensor<T>, Error>
{
    return download(src.const_view());
}

template <typename T, typename>
struct DeviceTensorView
{
    // Rank is taken from extents_ rather than passed separately. Extents cannot hold more
    // than VIKA_MAX_RANK entries, so an over-rank view is unrepresentable instead of being
    // a runtime failure.
    DeviceTensorView(T *data_, const Extents &extents_) : data(data_), rank(extents_.size())
    {
        std::copy(std::begin(extents_), std::end(extents_), extents);
        std::exclusive_scan(std::rbegin(extents_), std::rend(extents_), std::make_reverse_iterator(strides + rank),
                            usize{1}, std::multiplies<usize>{});
    }

    T *data = nullptr;
    usize extents[VIKA_MAX_RANK] = {};
    usize strides[VIKA_MAX_RANK] = {};
    usize rank = 0;

    // rank is bounded by construction, so this cannot overflow Extents.
    auto to_extents() const -> Extents
    {
        Extents result{};
        for (usize i = 0; i < rank; ++i)
        {
            result.push_back(extents[i]);
        }
        return result;
    }

    // Slices the leading (batch) dimension down to the first n entries. Row-major storage means
    // those are already a contiguous prefix of the same buffer, so this is just a reinterpreted
    // extent, never a copy or allocation. A Result rather than a panic: the view is a
    // self-contained type and shouldn't assume its caller already validated n, since a caller's
    // actual batch size exceeding what a tensor was allocated for is an ordinary, expected-to-be-
    // handled condition, not an internal invariant violation.
    auto first_n(usize n) const -> Result<DeviceTensorView, Error>
    {
        if (n > extents[0])
        {
            return error(VIKA_SHAPE_ERROR("first_n: requested %zu rows but tensor only has %zu", n, extents[0]));
        }
        auto sliced_extents = to_extents();
        sliced_extents[0] = n;
        return ok(DeviceTensorView(data, sliced_extents));
    }

    auto const_view() const -> DeviceTensorView<const T>
    {
        return DeviceTensorView<const T>(data, to_extents());
    }

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
// Stream
// =============================================================================

struct StreamDeleter
{
    auto operator()(std::remove_pointer_t<cudaStream_t> *stream)
    {
        cudaStreamDestroy(stream);
    }
};

// Owns a CUDA stream. Move-only via unique_ptr, same idiom as DeviceOwningTensor's DeviceDeleter:
// a moved-from Stream holds a null pointer, which cudaStreamDestroy never sees since the deleter
// only runs on a non-null owned handle. Created non-blocking so it never implicitly synchronizes
// with the legacy default stream (stream 0) the way a plain cudaStreamCreate() stream would.
class Stream
{
    using Self = Stream;

  public:
    static auto create() -> Result<Self, Error>
    {
        cudaStream_t stream = nullptr;
        const auto err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (is_error(err))
        {
            return error(VIKA_DEVICE_ERROR(err));
        }
        return ok(Self(stream));
    }

    auto handle() const -> cudaStream_t
    {
        return _stream.get();
    }

  private:
    explicit Stream(cudaStream_t stream) : _stream(stream)
    {
    }

    std::unique_ptr<std::remove_pointer_t<cudaStream_t>, StreamDeleter> _stream;
};

// =============================================================================
// Kernel Infrastructure
// =============================================================================

template <typename T>
struct [[nodiscard]] KernelJob
{
    // Implicit on purpose, mirroring Result's Err<E> constructor: it lets a function
    // returning KernelJob<T> write `return error(...)` (or UNWRAP_OR_RETURN) without
    // naming T again — the return statement's own conversion to the declared return
    // type supplies it. Takes KernelJob out of aggregate territory, which is fine:
    // nothing outside launched()/failed()/ready() ever brace-initialised one.
    KernelJob(Err<Error> err) : result(std::move(err)), stream(nullptr)
    {
    }

    // Work that was enqueued on a stream.
    static auto launched(T value, cudaStream_t stream) -> KernelJob
    {
        return KernelJob(std::move(value), stream);
    }

    // A request rejected before anything was enqueued, so there is no stream to wait on.
    // Holding a Result rather than a value plus an error flag means there is no
    // placeholder value to accidentally read: callers cannot reach a T without first
    // confronting the error, and result.is_error() is inspectable without synchronising.
    static auto failed(Error err) -> KernelJob
    {
        return KernelJob(error(std::move(err)));
    }

    // A value that was never asynchronous to begin with (a passthrough view, a host-supplied
    // gradient seeding a backward pass, ...): no kernel ran, so there is no stream to wait on
    // either. Named after std::future's "ready future" — a future whose value is already
    // available needs no wait. wait() recognises the null stream and skips synchronisation
    // instead of synchronising the real legacy default stream for work that was never queued.
    static auto ready(T value) -> KernelJob
    {
        return KernelJob(std::move(value), nullptr);
    }

    auto wait() -> Result<T, Error>
    {
        if (result.is_error())
        {
            return error(result.unwrap_error());
        }

        // No stream means no launch happened (failed()/ready()); nothing to synchronise on.
        if (stream != nullptr)
        {
            // Qualified: the unqualified name would resolve to this class's is_error().
            const auto err = cudaStreamSynchronize(stream);
            if (vika::is_error(err))
            {
                return error(VIKA_DEVICE_ERROR(err));
            }
        }
        return ok(result.unwrap());
    }

    auto is_error() const -> bool
    {
        return result.is_error();
    }

    auto is_ok() const -> bool
    {
        return result.is_ok();
    }

    Result<T, Error> result;
    cudaStream_t stream;

  private:
    KernelJob(T value, cudaStream_t stream_) : result(ok(std::move(value))), stream(stream_)
    {
    }
};

// Launches `kernel` and immediately reads back cudaGetLastError(), since
// cudaStreamSynchronize (what KernelJob::wait() checks) does not surface launch-configuration
// failures such as cudaErrorInvalidConfiguration. On success the launch's `output` view rides
// through as the job's value; on failure the job carries the error instead, so a bad launch
// can never be read as if it had produced real data.
template <typename Kernel, typename Output, typename... Args>
auto launch_kernel(Kernel kernel, Output output, dim3 grid, dim3 block, cudaStream_t stream, Args... args)
    -> KernelJob<Output>
{
    kernel<<<grid, block, 0, stream>>>(args...);
    const auto err = cudaGetLastError();
    if (is_error(err))
    {
        return KernelJob<Output>::failed(VIKA_DEVICE_ERROR(err));
    }
    return KernelJob<Output>::launched(output, stream);
}

// Every job is waited on regardless of an earlier one's outcome: short-circuiting on the first
// failure would leave later jobs' async work unsynchronised, racing with whatever the caller
// does next. Each job's own Result is preserved rather than collapsed into one aggregate
// success/failure, so the caller decides how to interpret a batch of independent outcomes
// (fail on the first one that matters to it, collect every failure, count successes, ...)
// instead of this function committing to one policy for everyone.
template <typename T>
auto wait_on(std::vector<KernelJob<T>> &jobs) -> std::vector<Result<T, Error>>
{
    std::vector<Result<T, Error>> results;
    results.reserve(jobs.size());
    for (auto &job : jobs)
    {
        results.push_back(job.wait());
    }
    return results;
}

struct AdamParameters
{
    f32 learning_rate = 1e-1f;
    f32 beta1 = 0.9f;
    f32 beta2 = 0.999f;
    f32 epsilon = 1e-8f;
};

// One trainable tensor: value is mutated in place by the optimizer, grad is read-only (the
// optimizer step never writes gradients, only reads them, regardless of which overload of
// parameters() produced this). Layers expose their full parameter list uniformly through
// parameters() regardless of how many tensors they own, instead of every caller needing to know
// each layer type's specific field names (weights/biases, filters/biases, ...).
struct Parameter
{
    DeviceTensorViewf value;
    DeviceTensorConstViewf grad;
};

// Read-only counterpart of Parameter, returned by the const overload of parameters() - a const
// layer cannot hand out a mutable view of its own value, so callers that only need to inspect
// parameters (e.g. to discover shapes) use this instead. Must stay in lockstep with Parameter's
// order/count for a given layer, since AdamState entries are positional.
struct ConstParameter
{
    DeviceTensorConstViewf value;
    DeviceTensorConstViewf grad;
};

// Per parameter, not per layer: a layer with N parameters gets N independent AdamStates
// (see Layer::parameters()), so adding a layer type with a different parameter count needs no
// changes here.
struct AdamState
{
    DeviceOwningTensorf m;
    DeviceOwningTensorf v;

    static auto create(const Extents &extents) -> Result<AdamState, Error>;
};

// Launches one adam_update per parameter on the given stream and returns every job unresolved
// (mirroring forward()/backward()/weight_gradients()): the caller decides when to wait, so
// independent parameters - and independent layers, each on their own stream - can all be
// in flight before anything blocks.
auto update_parameters(std::vector<Parameter> &parameters, std::vector<AdamState> &states, cudaStream_t stream,
                       const AdamParameters &params, usize t) -> std::vector<KernelJob<Void>>;

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

__global__ auto adam_update(const AdamParameters parameters, f32 m_hat_scale, f32 v_hat_scale,
                            DeviceTensorConstViewf d_weights, DeviceTensorViewf weights, DeviceTensorViewf m_weights,
                            DeviceTensorViewf v_weights) -> void;

__global__ auto add_bias(DeviceTensorConstView1f biases, DeviceTensorView2f out) -> void;

__global__ auto sum_rows(DeviceTensorConstView2f matrix, DeviceTensorView1f out) -> void;

__global__ auto mse_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets, DeviceTensorView1f out)
    -> void;
__global__ auto mse_gradient_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets,
                                    DeviceTensorViewf out) -> void;

__host__ __device__ auto sigmoid(f32 x) -> f32;

auto uniform_tensor(DeviceTensorViewf tensor, u32 seed, cudaStream_t stream) -> KernelJob<Void>;
auto xavier_tensor(DeviceTensorViewf tensor, u32 seed, usize fan_in, usize fan_out, cudaStream_t stream)
    -> KernelJob<Void>;

// =============================================================================
// Layers
// =============================================================================

struct DenseLayer
{
    static auto with_weights(usize batch_size, DeviceOwningTensor2f weights, DeviceOwningTensor1f biases)
        -> Result<DenseLayer, Error>;

    static auto randomized(usize batch_size, usize input_features, usize neuron_count, u32 seed)
        -> Result<DenseLayer, Error>;

    auto forward(const DeviceTensorConstView2f &inputs) -> KernelJob<DeviceTensorConstView2f>;

    auto backward(const DeviceTensorConstView2f &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>;

    auto weight_gradients(const DeviceTensorConstView2f &inputs, const DeviceTensorConstView2f &upstream_gradient)
        -> KernelJob<std::tuple<DeviceTensorConstView2f, DeviceTensorConstView1f>>;

    auto parameters() -> std::vector<Parameter>;
    auto parameters() const -> std::vector<ConstParameter>;

    auto update(std::vector<AdamState> &states, const AdamParameters &params, usize t) -> std::vector<KernelJob<Void>>;

    DeviceOwningTensor2f outputs;
    DeviceOwningTensor2f weights;
    DeviceOwningTensor1f biases;

    DeviceOwningTensor2f d_inputs;
    DeviceOwningTensor2f d_weights;
    DeviceOwningTensor1f d_biases;

    Stream stream;
};

struct SigmoidLayer
{
    static auto with_extents(const Extents &extents) -> Result<SigmoidLayer, Error>;

    auto forward(const DeviceTensorConstViewf &inputs) -> KernelJob<DeviceTensorConstView2f>;

    auto backward(const DeviceTensorConstViewf &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>;

    DeviceOwningTensor2f outputs;
    DeviceOwningTensor2f d_inputs;

    Stream stream;
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
                             DeviceOwningTensor1f biases, usize stride, usize padding) -> Result<Conv2DLayer, Error>;

    static auto randomized(usize batch_size, usize input_height, usize input_width, usize kH, usize kW, usize C_in,
                           usize C_out, usize stride, usize padding, u32 seed) -> Result<Conv2DLayer, Error>;

    auto forward(const DeviceTensorConstView4f &inputs) -> KernelJob<DeviceTensorConstView4f>;

    auto backward(const DeviceTensorConstView4f &upstream) -> KernelJob<DeviceTensorConstView4f>;

    auto weight_gradients(const DeviceTensorConstView4f &inputs, const DeviceTensorConstView4f &upstream)
        -> KernelJob<std::tuple<DeviceTensorConstView4f, DeviceTensorConstView1f>>;

    auto parameters() -> std::vector<Parameter>;
    auto parameters() const -> std::vector<ConstParameter>;

    auto update(std::vector<AdamState> &states, const AdamParameters &params, usize t) -> std::vector<KernelJob<Void>>;

    DeviceOwningTensor4f outputs;
    DeviceOwningTensor4f d_inputs;
    DeviceOwningTensor4f filters;
    DeviceOwningTensor1f biases;
    DeviceOwningTensor4f d_filters;
    DeviceOwningTensor1f d_biases;

    usize stride;
    usize padding;

    Stream stream;
};

struct MaxPool2DLayer
{
    static auto with_extents(usize batch_size, usize input_height, usize input_width, usize channels, usize pool_h,
                             usize pool_w, usize stride) -> Result<MaxPool2DLayer, Error>;

    auto forward(const DeviceTensorConstView4f &inputs) -> KernelJob<DeviceTensorConstView4f>;

    auto backward(const DeviceTensorConstView4f &upstream) -> KernelJob<DeviceTensorConstView4f>;

    DeviceOwningTensor4f outputs;
    DeviceOwningTensor4u argmax;
    DeviceOwningTensor4f d_inputs;

    usize pool_h;
    usize pool_w;
    usize stride;

    Stream stream;
};

struct MSELoss
{
    static auto with_extents(const Extents &extents) -> Result<MSELoss, Error>;

    auto forward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
        -> KernelJob<DeviceTensorConstView1f>;

    auto backward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
        -> KernelJob<DeviceTensorConstViewf>;

    DeviceOwningTensor1f loss;
    DeviceOwningTensorf d_inputs;

    Stream stream;
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

    static auto trainable(LayerKind kind) -> Layer
    {
        return {std::move(kind), false};
    }
    static auto frozen(LayerKind kind) -> Layer
    {
        return {std::move(kind), true};
    }

    auto freeze() -> void
    {
        is_frozen = true;
    }
    auto unfreeze() -> void
    {
        is_frozen = false;
    }
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

    // Indexed directly by NodeId::value, which is already a dense 0..layers.size()-1 index
    // (same space layers/layer_inputs use) rather than an opaque hash key. Slots are optional
    // because KernelJob<T> cannot be default-constructed (see KernelJob<T>'s Result<T, Error>,
    // which requires DeviceTensorConstViewf to be default-constructible, and it deliberately
    // is not) and because not every slot is ever written: the input node never gets a
    // backward_jobs entry, since nothing needs its gradient.
    std::vector<std::optional<KernelJob<DeviceTensorConstViewf>>> forward_jobs;
    std::vector<std::optional<KernelJob<DeviceTensorConstViewf>>> backward_jobs;

    auto forward(DeviceTensorConstViewf input) -> Result<DeviceTensorConstViewf, Error>;
    auto backward(DeviceTensorConstViewf loss_grad) -> Result<Void, Error>;
    auto step(AdamOptimizer &optimizer, usize t) -> Result<Void, Error>;
};

struct AdamOptimizer
{
    AdamParameters params;
    // One AdamState per parameter, not per layer - see AdamState.
    std::unordered_map<usize, std::vector<AdamState>> states;

    static auto from_model(const Model &model, AdamParameters params) -> Result<AdamOptimizer, Error>;
};

struct ComputationGraph
{
    usize batch_size;
    std::vector<Node> nodes;
    u32 seed = 0;

    auto input(Extents spatial_extents) -> NodeId;
    auto dense(NodeId input, usize output_features, std::optional<u32> requested_seed = std::nullopt)
        -> Result<NodeId, Error>;
    auto sigmoid(NodeId input) -> Result<NodeId, Error>;
    auto flatten(NodeId input) -> Result<NodeId, Error>;
    auto conv2d(NodeId input, usize kernel_height, usize kernel_width, usize channels_out, usize stride, usize padding,
                std::optional<u32> requested_seed = std::nullopt) -> Result<NodeId, Error>;
    auto maxpool2d(NodeId input, usize pool_height, usize pool_width, usize stride) -> Result<NodeId, Error>;
    auto compile(NodeId output) -> Result<Model, Error>;

  private:
    // Rolls the graph's own seed forward and returns the new value, so each layer that omits an
    // explicit seed gets a distinct, deterministic one derived from this graph's seed alone.
    auto next_seed() -> u32;
};

auto make_layer(const LayerSpec &spec, usize batch_size, const Extents &pred_extents) -> Result<Layer, Error>;

auto update_layer(LayerKind &kind, DeviceTensorConstViewf forward_input, DeviceTensorConstViewf upstream,
                  std::vector<AdamState> &states, const AdamParameters &params, usize t) -> Result<Void, Error>;

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

auto error_kind_name(ErrorKind kind) -> const char *
{
    switch (kind)
    {
    case ErrorKind::Device:
        return "device";
    case ErrorKind::Shape:
        return "shape";
    case ErrorKind::Graph:
        return "graph";
    case ErrorKind::Unsupported:
        return "unsupported";
    }
    return "unknown";
}

auto Error::make(ErrorKind kind, const char *file, i32 line, const char *fmt, ...) -> Error
{
    Error err{};
    err._kind = kind;
    err._file = file;
    err._line = line;

    va_list args;
    va_start(args, fmt);
    vsnprintf(err._message, sizeof(err._message), fmt, args);
    va_end(args);
    return err;
}

auto Error::from_cuda(cudaError_t code, const char *file, i32 line) -> Error
{
    VIKA_PANIC_IF(!is_error(code), "Error::from_cuda called with cudaSuccess");

    Error err{};
    err._kind = ErrorKind::Device;
    err._code = code;
    err._file = file;
    err._line = line;
    snprintf(err._message, sizeof(err._message), "[%s] %s", cudaGetErrorName(code), cudaGetErrorString(code));
    return err;
}

auto Error::describe() const -> std::string
{
    return std::string(_file) + ":" + std::to_string(_line) + ": " + error_kind_name(_kind) + ": " + _message;
}

auto Error::crash() const -> void
{
    VIKA_PANIC("%s", describe().c_str());
}

// =============================================================================
// Tensor Helpers
// =============================================================================

auto element_count(const Extents &extents) -> usize
{
    using namespace std;
    return accumulate(begin(extents), end(extents), 1ul, std::multiplies<>{});
}

// Every dimension but the leading one. A layer whose workspace is sized for a batch capacity
// rather than one fixed batch size compares two extents on this rather than in full, since the
// leading (batch) dimension is expected to vary from call to call.
auto trailing_extents(const Extents &extents) -> Extents
{
    Extents result{};
    for (usize i = 1; i < extents.size(); ++i)
    {
        result.push_back(extents[i]);
    }
    return result;
}

auto checked_element_count(const Extents &extents) -> Result<usize, Error>
{
    usize count = 1;
    for (usize i = 0; i < extents.size(); ++i)
    {
        const usize extent = extents[i];
        if (extent != 0 && count > std::numeric_limits<usize>::max() / extent)
        {
            return error(VIKA_SHAPE_ERROR("element count overflows usize at dimension %zu (extent %zu)", i, extent));
        }
        count *= extent;
    }
    return ok(count);
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

__host__ __device__ inline auto pcg_hash(u32 input) -> u32
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

auto uniform_tensor(DeviceTensorViewf tensor, u32 seed, cudaStream_t stream) -> KernelJob<Void>
{
    const usize n = tensor.element_count();
    const usize threads = 256;
    return launch_kernel(uniform_tensor_kernel, Void{}, dim3((n + threads - 1) / threads), dim3(threads), stream,
                         tensor, seed);
}

auto xavier_tensor(DeviceTensorViewf tensor, u32 seed, usize fan_in, usize fan_out, cudaStream_t stream)
    -> KernelJob<Void>
{
    const f32 limit = std::sqrt(6.0f / (f32)(fan_in + fan_out));
    const usize n = tensor.element_count();
    const usize threads = 256;
    return launch_kernel(xavier_tensor_kernel, Void{}, dim3((n + threads - 1) / threads), dim3(threads), stream, tensor,
                         seed, limit);
}

// Adam bias-correction scales. These depend only on the step count, so computing them
// once on the host beats recomputing pow() in every thread.
inline auto adam_bias_correction(const AdamParameters &params, usize t) -> std::pair<f32, f32>
{
    const auto correct = [t](f32 beta) -> f32 { return (f32)(1.0 / (1.0 - std::pow((double)beta, (double)t))); };
    return {correct(params.beta1), correct(params.beta2)};
}

auto update_parameters(std::vector<Parameter> &parameters, std::vector<AdamState> &states, cudaStream_t stream,
                       const AdamParameters &params, usize t) -> std::vector<KernelJob<Void>>
{
    const auto [m_hat_scale, v_hat_scale] = adam_bias_correction(params, t);
    const usize threads = 256;

    std::vector<KernelJob<Void>> jobs;
    jobs.reserve(parameters.size());
    for (usize i = 0; i < parameters.size(); ++i)
    {
        const auto count = parameters[i].value.element_count();
        jobs.push_back(launch_kernel(adam_update, Void{}, dim3((count + threads - 1) / threads), dim3(threads), stream,
                                     params, m_hat_scale, v_hat_scale, parameters[i].grad, parameters[i].value,
                                     states[i].m.view(), states[i].v.view()));
    }
    return jobs;
}

// =============================================================================
// Layers
// =============================================================================

auto DenseLayer::with_weights(usize batch_size, DeviceOwningTensor2f weights, DeviceOwningTensor1f biases)
    -> Result<DenseLayer, Error>
{
    const auto feature_count = weights.extent(0);
    const auto neuron_count = weights.extent(1);

    auto outputs = UNWRAP_OR_RETURN(DeviceOwningTensor2f::empty({batch_size, neuron_count}));

    auto d_inputs = UNWRAP_OR_RETURN(DeviceOwningTensor2f::empty({batch_size, feature_count}));
    auto d_weights = UNWRAP_OR_RETURN(DeviceOwningTensor2f::empty_like(weights));
    auto d_biases = UNWRAP_OR_RETURN(DeviceOwningTensor1f::empty_like(biases));

    auto stream = UNWRAP_OR_RETURN(Stream::create());
    return ok<DenseLayer>({
        .outputs = std::move(outputs),
        .weights = std::move(weights),
        .biases = std::move(biases),
        .d_inputs = std::move(d_inputs),
        .d_weights = std::move(d_weights),
        .d_biases = std::move(d_biases),
        .stream = std::move(stream),
    });
}

auto DenseLayer::randomized(usize batch_size, usize input_features, usize neuron_count, u32 seed)
    -> Result<DenseLayer, Error>
{
    auto weights = UNWRAP_OR_RETURN(DeviceOwningTensor2f::empty({input_features, neuron_count}));
    auto biases = UNWRAP_OR_RETURN(DeviceOwningTensor1f::from(std::vector<f32>(neuron_count, 0.0f)));

    // Built first so xavier_tensor can run on the layer's own stream instead of standing up a
    // second, throwaway one just for initialization.
    auto layer = UNWRAP_OR_RETURN(with_weights(batch_size, std::move(weights), std::move(biases)));
    UNWRAP_OR_RETURN(
        xavier_tensor(layer.weights.view(), seed, input_features, neuron_count, layer.stream.handle()).wait());
    return ok(std::move(layer));
}

auto DenseLayer::forward(const DeviceTensorConstView2f &inputs) -> KernelJob<DeviceTensorConstView2f>
{
    if (inputs.extents[1] != weights.extent(0))
    {
        return KernelJob<DeviceTensorConstView2f>::failed(VIKA_SHAPE_ERROR(
            "dense forward: input has %zu features, layer expects %zu", inputs.extents[1], weights.extent(0)));
    }

    const usize k = inputs.extents[0];
    auto sliced_outputs = UNWRAP_OR_RETURN(outputs.view().first_n(k));
    auto sliced_outputs_const = sliced_outputs.const_view();

    const u32 M = (u32)k;
    const u32 N = outputs.extent(1);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    auto matmul_job = launch_kernel(matmul_kernel, sliced_outputs_const, grid, block, stream.handle(), inputs,
                                    weights.const_view(), sliced_outputs);
    if (matmul_job.is_error())
    {
        return matmul_job;
    }
    return launch_kernel(add_bias, sliced_outputs_const, grid, block, stream.handle(), biases.const_view(),
                         sliced_outputs);
}

auto DenseLayer::backward(const DeviceTensorConstView2f &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>
{
    if (upstream_gradient.extents[1] != weights.extent(1))
    {
        return KernelJob<DeviceTensorConstView2f>::failed(
            VIKA_SHAPE_ERROR("dense backward: upstream has %zu features, layer expects %zu",
                             upstream_gradient.extents[1], weights.extent(1)));
    }

    const usize k = upstream_gradient.extents[0];
    auto sliced_d_inputs = UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const u32 M = (u32)k;
    const u32 N = d_inputs.extent(1);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    return launch_kernel(matmul_kernel, sliced_d_inputs.const_view(), grid, block, stream.handle(), upstream_gradient,
                         transposed(weights.const_view()), sliced_d_inputs);
}

auto DenseLayer::weight_gradients(const DeviceTensorConstView2f &inputs,
                                  const DeviceTensorConstView2f &upstream_gradient)
    -> KernelJob<std::tuple<DeviceTensorConstView2f, DeviceTensorConstView1f>>
{
    const u32 M = d_inputs.extent(0);
    const u32 N = d_inputs.extent(1);
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    const auto gradients = std::make_tuple(d_weights.const_view(), d_biases.const_view());

    // NOTE: Run in separate streams?
    auto matmul_job = launch_kernel(matmul_kernel, gradients, grid, block, stream.handle(), transposed(inputs),
                                    upstream_gradient, d_weights.view());
    if (matmul_job.is_error())
    {
        return matmul_job;
    }

    const auto block_dim = dim3(256);
    const auto grid_dim = dim3((upstream_gradient.extents[1] + block_dim.x - 1) / block_dim.x);
    return launch_kernel(sum_rows, gradients, grid_dim, block_dim, stream.handle(), transposed(upstream_gradient),
                         d_biases.view());
}

auto DenseLayer::parameters() -> std::vector<Parameter>
{
    return {
        {weights.view(), d_weights.const_view()},
        {biases.view(), d_biases.const_view()},
    };
}

// Must stay in lockstep with the non-const overload above: same order, same count.
auto DenseLayer::parameters() const -> std::vector<ConstParameter>
{
    return {
        {weights.const_view(), d_weights.const_view()},
        {biases.const_view(), d_biases.const_view()},
    };
}

auto DenseLayer::update(std::vector<AdamState> &states, const AdamParameters &params, usize t)
    -> std::vector<KernelJob<Void>>
{
    auto params_list = parameters();
    return update_parameters(params_list, states, stream.handle(), params, t);
}

auto SigmoidLayer::with_extents(const Extents &extents) -> Result<SigmoidLayer, Error>
{
    auto outputs = UNWRAP_OR_RETURN(DeviceOwningTensor2f::empty(extents));
    auto d_inputs = UNWRAP_OR_RETURN(DeviceOwningTensor2f::empty(extents));

    auto stream = UNWRAP_OR_RETURN(Stream::create());
    return ok(
        SigmoidLayer{.outputs = std::move(outputs), .d_inputs = std::move(d_inputs), .stream = std::move(stream)});
}

auto SigmoidLayer::forward(const DeviceTensorConstViewf &inputs) -> KernelJob<DeviceTensorConstView2f>
{
    const auto input_extents = inputs.to_extents();
    if (trailing_extents(input_extents) != trailing_extents(outputs.extents()))
    {
        return KernelJob<DeviceTensorConstView2f>::failed(VIKA_SHAPE_ERROR(
            "sigmoid forward: input is rank %zu with %zu elements, layer expects rank %zu with %zu",
            input_extents.size(), element_count(input_extents), outputs.extents().size(), outputs.element_count()));
    }

    const usize k = input_extents[0];
    auto sliced_outputs = UNWRAP_OR_RETURN(outputs.view().first_n(k));

    usize threads = 256;
    usize blocks = (inputs.element_count() + threads - 1) / threads;

    return launch_kernel(sigmoid_forward, sliced_outputs.const_view(), dim3(blocks), dim3(threads), stream.handle(),
                         inputs, sliced_outputs);
}

auto SigmoidLayer::backward(const DeviceTensorConstViewf &upstream_gradient) -> KernelJob<DeviceTensorConstView2f>
{
    const auto upstream_extents = upstream_gradient.to_extents();
    if (trailing_extents(upstream_extents) != trailing_extents(d_inputs.extents()))
    {
        return KernelJob<DeviceTensorConstView2f>::failed(VIKA_SHAPE_ERROR(
            "sigmoid backward: upstream is rank %zu with %zu elements, layer expects rank %zu with %zu",
            upstream_extents.size(), element_count(upstream_extents), d_inputs.extents().size(),
            d_inputs.element_count()));
    }

    // Invariant: with_extents allocates both from the same extents.
    if (d_inputs.extents() != outputs.extents())
    {
        return KernelJob<DeviceTensorConstView2f>::failed(
            VIKA_SHAPE_ERROR("sigmoid backward: gradient holds %zu elements but outputs hold %zu",
                             d_inputs.element_count(), outputs.element_count()));
    }

    const usize k = upstream_extents[0];
    auto sliced_d_inputs = UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    usize threads = 256;
    usize blocks = (upstream_gradient.element_count() + threads - 1) / threads;

    return launch_kernel(sigmoid_backward, sliced_d_inputs.const_view(), dim3(blocks), dim3(threads), stream.handle(),
                         outputs.const_view().first_n(k).unwrap(), upstream_gradient, sliced_d_inputs);
}

auto Flatten2DLayer::with_extents(const Extents &extents) -> Flatten2DLayer
{
    return {extents};
}

auto Flatten2DLayer::forward(DeviceTensorConstViewf inputs) const -> KernelJob<DeviceTensorConstView2f>
{
    const auto input_extents = inputs.to_extents();
    if (input_extents != extents)
    {
        return KernelJob<DeviceTensorConstView2f>::failed(VIKA_SHAPE_ERROR(
            "flatten forward: input is rank %zu with %zu elements, layer expects rank %zu with %zu",
            input_extents.size(), element_count(input_extents), extents.size(), element_count(extents)));
    }

    const auto batch = inputs.extents[0];
    const usize features =
        std::accumulate(inputs.extents + 1, inputs.extents + extents.size(), 1ul, std::multiplies<usize>{});
    return KernelJob<DeviceTensorConstView2f>::ready(DeviceTensorConstView2f(inputs.data, {batch, features}));
}

auto InputLayer::forward(DeviceTensorConstViewf input) const -> KernelJob<DeviceTensorConstViewf>
{
    return KernelJob<DeviceTensorConstViewf>::ready(input);
}

auto InputLayer::backward(DeviceTensorConstViewf upstream) const -> KernelJob<DeviceTensorConstViewf>
{
    return KernelJob<DeviceTensorConstViewf>::ready(upstream);
}

auto Flatten2DLayer::backward(DeviceTensorConstView2f upstream_gradient) const -> KernelJob<DeviceTensorConstViewf>
{
    const auto upstream_extents = upstream_gradient.to_extents();
    if (element_count(upstream_extents) != element_count(extents))
    {
        return KernelJob<DeviceTensorConstViewf>::failed(
            VIKA_SHAPE_ERROR("flatten backward: upstream holds %zu elements, layer extents hold %zu",
                             element_count(upstream_extents), element_count(extents)));
    }

    return KernelJob<DeviceTensorConstViewf>::ready(DeviceTensorConstViewf(upstream_gradient.data, extents));
}

auto Conv2DLayer::with_weights(usize batch_size, usize input_height, usize input_width, DeviceOwningTensor4f filters,
                               DeviceOwningTensor1f biases, usize stride, usize padding) -> Result<Conv2DLayer, Error>
{
    const usize kH = filters.extent(0);
    const usize kW = filters.extent(1);
    const usize C_out = filters.extent(3);

    const usize C_in = filters.extent(2);
    const usize out_H = window_output_extent(input_height, kH, stride, padding);
    const usize out_W = window_output_extent(input_width, kW, stride, padding);

    auto outputs = UNWRAP_OR_RETURN(DeviceOwningTensor4f::empty({batch_size, out_H, out_W, C_out}));
    auto d_inputs = UNWRAP_OR_RETURN(DeviceOwningTensor4f::empty({batch_size, input_height, input_width, C_in}));
    auto d_filters = UNWRAP_OR_RETURN(DeviceOwningTensor4f::empty_like(filters));
    auto d_biases = UNWRAP_OR_RETURN(DeviceOwningTensor1f::empty_like(biases));

    auto stream = UNWRAP_OR_RETURN(Stream::create());
    return ok(Conv2DLayer{
        .outputs = std::move(outputs),
        .d_inputs = std::move(d_inputs),
        .filters = std::move(filters),
        .biases = std::move(biases),
        .d_filters = std::move(d_filters),
        .d_biases = std::move(d_biases),
        .stride = stride,
        .padding = padding,
        .stream = std::move(stream),
    });
}

auto Conv2DLayer::randomized(usize batch_size, usize input_height, usize input_width, usize kH, usize kW, usize C_in,
                             usize C_out, usize stride, usize padding, u32 seed) -> Result<Conv2DLayer, Error>
{
    auto filters = UNWRAP_OR_RETURN(DeviceOwningTensor4f::empty({kH, kW, C_in, C_out}));
    auto biases = UNWRAP_OR_RETURN(DeviceOwningTensor1f::from(std::vector<f32>(C_out, 0.0f)));

    // Built first so xavier_tensor can run on the layer's own stream instead of standing up a
    // second, throwaway one just for initialization.
    auto layer = UNWRAP_OR_RETURN(
        with_weights(batch_size, input_height, input_width, std::move(filters), std::move(biases), stride, padding));
    UNWRAP_OR_RETURN(
        xavier_tensor(layer.filters.view(), seed, kH * kW * C_in, kH * kW * C_out, layer.stream.handle()).wait());
    return ok(std::move(layer));
}

auto Conv2DLayer::forward(const DeviceTensorConstView4f &inputs) -> KernelJob<DeviceTensorConstView4f>
{
    if (trailing_extents(inputs.to_extents()) != trailing_extents(d_inputs.extents()))
    {
        return KernelJob<DeviceTensorConstView4f>::failed(
            VIKA_SHAPE_ERROR("conv2d forward: input is rank %zu with %zu elements, layer expects rank %zu with %zu",
                             inputs.rank, inputs.element_count(), d_inputs.extents().size(), d_inputs.element_count()));
    }

    const usize k = inputs.extents[0];
    auto sliced_outputs = UNWRAP_OR_RETURN(outputs.view().first_n(k));

    const usize H_out = outputs.extent(1);
    const usize W_out = outputs.extent(2);
    const usize C_out = outputs.extent(3);

    dim3 block(16, 16, 1);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, k * C_out);

    return launch_kernel(conv_forward, sliced_outputs.const_view(), grid, block, stream.handle(), inputs,
                         filters.const_view(), biases.const_view(), sliced_outputs, stride, padding);
}

auto Conv2DLayer::backward(const DeviceTensorConstView4f &upstream) -> KernelJob<DeviceTensorConstView4f>
{
    if (trailing_extents(upstream.to_extents()) != trailing_extents(outputs.extents()))
    {
        return KernelJob<DeviceTensorConstView4f>::failed(VIKA_SHAPE_ERROR(
            "conv2d backward: upstream is rank %zu with %zu elements, layer expects rank %zu with %zu", upstream.rank,
            upstream.element_count(), outputs.extents().size(), outputs.element_count()));
    }

    const usize k = upstream.extents[0];
    auto sliced_d_inputs = UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const usize H_in = d_inputs.extent(1);
    const usize W_in = d_inputs.extent(2);
    const usize C_in = d_inputs.extent(3);

    dim3 block(16, 16, 1);
    dim3 grid((W_in + block.x - 1) / block.x, (H_in + block.y - 1) / block.y, k * C_in);

    return launch_kernel(conv_backward, sliced_d_inputs.const_view(), grid, block, stream.handle(), upstream,
                         filters.const_view(), sliced_d_inputs, stride, padding);
}

auto Conv2DLayer::weight_gradients(const DeviceTensorConstView4f &inputs, const DeviceTensorConstView4f &upstream)
    -> KernelJob<std::tuple<DeviceTensorConstView4f, DeviceTensorConstView1f>>
{
    const usize filter_count = d_filters.element_count();
    const usize C_out = d_biases.element_count();
    const usize threads = 256;

    const auto gradients = std::make_tuple(d_filters.const_view(), d_biases.const_view());

    auto weight_job =
        launch_kernel(conv_weight_gradients, gradients, dim3((filter_count + threads - 1) / threads), dim3(threads),
                      stream.handle(), inputs, upstream, d_filters.view(), stride, padding);
    if (weight_job.is_error())
    {
        return weight_job;
    }
    return launch_kernel(conv_bias_gradients, gradients, dim3((C_out + threads - 1) / threads), dim3(threads),
                         stream.handle(), upstream, d_biases.view());
}

auto Conv2DLayer::parameters() -> std::vector<Parameter>
{
    return {
        {filters.view(), d_filters.const_view()},
        {biases.view(), d_biases.const_view()},
    };
}

// Must stay in lockstep with the non-const overload above: same order, same count.
auto Conv2DLayer::parameters() const -> std::vector<ConstParameter>
{
    return {
        {filters.const_view(), d_filters.const_view()},
        {biases.const_view(), d_biases.const_view()},
    };
}

auto Conv2DLayer::update(std::vector<AdamState> &states, const AdamParameters &params, usize t)
    -> std::vector<KernelJob<Void>>
{
    auto params_list = parameters();
    return update_parameters(params_list, states, stream.handle(), params, t);
}

auto MaxPool2DLayer::with_extents(usize batch_size, usize input_height, usize input_width, usize channels, usize pool_h,
                                  usize pool_w, usize stride) -> Result<MaxPool2DLayer, Error>
{
    const usize out_H = window_output_extent(input_height, pool_h, stride, 0);
    const usize out_W = window_output_extent(input_width, pool_w, stride, 0);

    auto outputs = UNWRAP_OR_RETURN(DeviceOwningTensor4f::empty({batch_size, out_H, out_W, channels}));
    auto argmax = UNWRAP_OR_RETURN((DeviceOwningTensor4u::empty({batch_size, out_H, out_W, channels})));
    auto d_inputs = UNWRAP_OR_RETURN(DeviceOwningTensor4f::empty({batch_size, input_height, input_width, channels}));

    auto stream = UNWRAP_OR_RETURN(Stream::create());
    return ok(MaxPool2DLayer{
        .outputs = std::move(outputs),
        .argmax = std::move(argmax),
        .d_inputs = std::move(d_inputs),
        .pool_h = pool_h,
        .pool_w = pool_w,
        .stride = stride,
        .stream = std::move(stream),
    });
}

auto MaxPool2DLayer::forward(const DeviceTensorConstView4f &inputs) -> KernelJob<DeviceTensorConstView4f>
{
    if (trailing_extents(inputs.to_extents()) != trailing_extents(d_inputs.extents()))
    {
        return KernelJob<DeviceTensorConstView4f>::failed(
            VIKA_SHAPE_ERROR("maxpool forward: input is rank %zu with %zu elements, layer expects rank %zu with %zu",
                             inputs.rank, inputs.element_count(), d_inputs.extents().size(), d_inputs.element_count()));
    }

    const usize k = inputs.extents[0];
    auto sliced_outputs = UNWRAP_OR_RETURN(outputs.view().first_n(k));
    auto sliced_argmax = UNWRAP_OR_RETURN(argmax.view().first_n(k));

    const usize H_out = outputs.extent(1);
    const usize W_out = outputs.extent(2);
    const usize C = outputs.extent(3);

    dim3 block(16, 16, 1);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, k * C);

    return launch_kernel(maxpool_forward, sliced_outputs.const_view(), grid, block, stream.handle(), inputs,
                         sliced_outputs, sliced_argmax, pool_h, pool_w, stride);
}

auto MaxPool2DLayer::backward(const DeviceTensorConstView4f &upstream) -> KernelJob<DeviceTensorConstView4f>
{
    if (trailing_extents(upstream.to_extents()) != trailing_extents(outputs.extents()))
    {
        return KernelJob<DeviceTensorConstView4f>::failed(VIKA_SHAPE_ERROR(
            "maxpool backward: upstream is rank %zu with %zu elements, layer expects rank %zu with %zu", upstream.rank,
            upstream.element_count(), outputs.extents().size(), outputs.element_count()));
    }

    const usize k = upstream.extents[0];
    auto sliced_d_inputs = UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const usize H_out = upstream.extents[1];
    const usize W_out = upstream.extents[2];
    const usize C = d_inputs.extent(3);

    cudaMemsetAsync(d_inputs.data(), 0, d_inputs.byte_count(), stream.handle());

    dim3 block(16, 16, 1);
    dim3 grid((W_out + block.x - 1) / block.x, (H_out + block.y - 1) / block.y, k * C);

    return launch_kernel(maxpool_backward, sliced_d_inputs.const_view(), grid, block, stream.handle(), upstream,
                         UNWRAP_OR_RETURN(argmax.const_view().first_n(k)), sliced_d_inputs);
}

auto MSELoss::with_extents(const Extents &extents) -> Result<MSELoss, Error>
{
    auto loss = UNWRAP_OR_RETURN(DeviceOwningTensor1f::empty({1}));
    auto d_inputs = UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(extents));

    auto stream = UNWRAP_OR_RETURN(Stream::create());
    return ok(MSELoss{.loss = std::move(loss), .d_inputs = std::move(d_inputs), .stream = std::move(stream)});
}

auto MSELoss::forward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
    -> KernelJob<DeviceTensorConstView1f>
{
    cudaMemsetAsync(loss.data(), 0, sizeof(f32), stream.handle());

    const usize n = predictions.element_count();
    const usize threads = 256;
    return launch_kernel(mse_kernel, loss.const_view(), dim3((n + threads - 1) / threads), dim3(threads),
                         stream.handle(), predictions, targets, loss.view());
}

auto MSELoss::backward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
    -> KernelJob<DeviceTensorConstViewf>
{
    if (trailing_extents(predictions.to_extents()) != trailing_extents(d_inputs.extents()))
    {
        return KernelJob<DeviceTensorConstViewf>::failed(VIKA_SHAPE_ERROR(
            "mse backward: predictions is rank %zu with %zu elements, layer expects rank %zu with %zu",
            predictions.rank, predictions.element_count(), d_inputs.extents().size(), d_inputs.element_count()));
    }

    const usize k = predictions.extents[0];
    auto sliced_d_inputs = UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const usize n = predictions.element_count();
    const usize threads = 256;
    return launch_kernel(mse_gradient_kernel, sliced_d_inputs.const_view(), dim3((n + threads - 1) / threads),
                         dim3(threads), stream.handle(), predictions, targets, sliced_d_inputs);
}

// =============================================================================
// Computation Graph
// =============================================================================

auto ComputationGraph::next_seed() -> u32
{
    seed = pcg_hash(seed);
    return seed;
}

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

auto ComputationGraph::dense(NodeId input, usize output_features, std::optional<u32> requested_seed)
    -> Result<NodeId, Error>
{
    if (input.value >= nodes.size())
    {
        return error(VIKA_GRAPH_ERROR("dense: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() != 2)
    {
        return error(VIKA_SHAPE_ERROR("dense: input must be rank 2 [N, features]"));
    }

    const u32 resolved_seed = requested_seed.has_value() ? *requested_seed : next_seed();
    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = DenseSpec{output_features, resolved_seed},
        .output_extents = {in_extents[0], output_features},
        .inputs = {input},
    });
    return ok(id);
}

auto ComputationGraph::sigmoid(NodeId input) -> Result<NodeId, Error>
{
    if (input.value >= nodes.size())
    {
        return error(VIKA_GRAPH_ERROR("sigmoid: invalid NodeId"));
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

auto ComputationGraph::flatten(NodeId input) -> Result<NodeId, Error>
{
    if (input.value >= nodes.size())
    {
        return error(VIKA_GRAPH_ERROR("flatten: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() < 2)
    {
        return error(VIKA_SHAPE_ERROR("flatten: input must be at least rank 2"));
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
                              usize padding, std::optional<u32> requested_seed) -> Result<NodeId, Error>
{
    if (input.value >= nodes.size())
    {
        return error(VIKA_GRAPH_ERROR("conv2d: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() != 4)
    {
        return error(VIKA_SHAPE_ERROR("conv2d: input must be rank 4 [N, H, W, C]"));
    }

    const auto H = in_extents[1];
    const auto W = in_extents[2];

    if (H + 2 * padding < kernel_height)
    {
        return error(VIKA_SHAPE_ERROR("conv2d: kernel height exceeds padded input height"));
    }
    if (W + 2 * padding < kernel_width)
    {
        return error(VIKA_SHAPE_ERROR("conv2d: kernel width exceeds padded input width"));
    }

    const auto out_H = window_output_extent(H, kernel_height, stride, padding);
    const auto out_W = window_output_extent(W, kernel_width, stride, padding);

    const u32 resolved_seed = requested_seed.has_value() ? *requested_seed : next_seed();
    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = Conv2DSpec{kernel_height, kernel_width, channels_out, stride, padding, resolved_seed},
        .output_extents = {in_extents[0], out_H, out_W, channels_out},
        .inputs = {input},
    });
    return ok(id);
}

auto ComputationGraph::maxpool2d(NodeId input, usize pool_height, usize pool_width, usize stride)
    -> Result<NodeId, Error>
{
    if (input.value >= nodes.size())
    {
        return error(VIKA_GRAPH_ERROR("maxpool2d: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() != 4)
    {
        return error(VIKA_SHAPE_ERROR("maxpool2d: input must be rank 4 [N, H, W, C]"));
    }

    const auto H = in_extents[1];
    const auto W = in_extents[2];

    if (H < pool_height)
    {
        return error(VIKA_SHAPE_ERROR("maxpool2d: pool height exceeds input height"));
    }
    if (W < pool_width)
    {
        return error(VIKA_SHAPE_ERROR("maxpool2d: pool width exceeds input width"));
    }

    const auto out_H = window_output_extent(H, pool_height, stride, 0);
    const auto out_W = window_output_extent(W, pool_width, stride, 0);

    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = MaxPool2DSpec{pool_height, pool_width, stride},
        .output_extents = {in_extents[0], out_H, out_W, in_extents[3]},
        .inputs = {input},
    });
    return ok(id);
}

auto make_layer(const LayerSpec &spec, usize batch_size, const Extents &pred_extents) -> Result<Layer, Error>
{
    return std::visit(
        [&](const auto &s) -> Result<Layer, Error> {
            using T = std::decay_t<decltype(s)>;

            // Every constructed layer is wrapped the same way; map() forwards a
            // construction failure untouched instead of repeating an is_error() block.
            const auto as_trainable = [](auto layer) { return Layer::trainable(LayerKind{std::move(layer)}); };
            if constexpr (std::is_same_v<T, InputSpec>)
            {
                return ok(Layer::trainable(LayerKind{InputLayer{}}));
            }
            else if constexpr (std::is_same_v<T, DenseSpec>)
            {
                return DenseLayer::randomized(batch_size, pred_extents.at(1), s.output_features, s.seed)
                    .map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, SigmoidSpec>)
            {
                return SigmoidLayer::with_extents(pred_extents).map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, FlattenSpec>)
            {
                return ok(Layer::trainable(LayerKind{Flatten2DLayer::with_extents(pred_extents)}));
            }
            else if constexpr (std::is_same_v<T, Conv2DSpec>)
            {
                return Conv2DLayer::randomized(batch_size, pred_extents.at(1), pred_extents.at(2), s.kernel_height,
                                               s.kernel_width, pred_extents.at(3), s.channels_out, s.stride, s.padding,
                                               s.seed)
                    .map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, MaxPool2DSpec>)
            {
                return MaxPool2DLayer::with_extents(batch_size, pred_extents.at(1), pred_extents.at(2),
                                                    pred_extents.at(3), s.pool_height, s.pool_width, s.stride)
                    .map(as_trainable);
            }
            else
            {
                static_assert(sizeof(T) == 0, "unhandled LayerSpec type in make_layer");
                return error(VIKA_GRAPH_ERROR("unreachable"));
            }
        },
        spec);
}

auto ComputationGraph::compile(NodeId output) -> Result<Model, Error>
{
    if (output.value >= nodes.size())
    {
        return error(VIKA_GRAPH_ERROR("compile: invalid output NodeId"));
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
        return error(VIKA_GRAPH_ERROR("compile: graph must have exactly one input node"));
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

auto Model::forward(DeviceTensorConstViewf input) -> Result<DeviceTensorConstViewf, Error>
{
    forward_jobs.clear();
    forward_jobs.resize(layers.size());

    for (const auto node_id : execution_order)
    {
        const auto &preds = layer_inputs[node_id.value];
        if (preds.size() > 1)
        {
            return error(VIKA_UNSUPPORTED_ERROR("forward: node %zu has %zu inputs, multi-input nodes are not supported",
                                                node_id.value, preds.size()));
        }

        DeviceTensorConstViewf pred_output = input;
        if (!preds.empty())
        {
            pred_output = UNWRAP_OR_RETURN(forward_jobs[preds[0].value].value().wait());
        }

        auto job = std::visit(
            [&pred_output](auto &layer) -> KernelJob<DeviceTensorConstViewf> { return layer.forward(pred_output); },
            layers[node_id.value].kind);

        forward_jobs[node_id.value] = std::move(job);
    }

    auto result = forward_jobs[output_node.value].value().wait();
    if (result.is_error())
    {
        return error(result.unwrap_error());
    }
    return ok(result.unwrap());
}

auto Model::backward(DeviceTensorConstViewf loss_grad) -> Result<Void, Error>
{
    backward_jobs.clear();
    backward_jobs.resize(layers.size());
    backward_jobs[output_node.value] = KernelJob<DeviceTensorConstViewf>::ready(loss_grad);

    for (auto it = execution_order.rbegin(); it != execution_order.rend(); ++it)
    {
        const auto node_id = *it;
        const auto &preds = layer_inputs[node_id.value];
        if (preds.empty())
        {
            continue;
        }

        // Checked before dispatching, so an unsupported node does not launch work first.
        if (preds.size() > 1)
        {
            return error(VIKA_UNSUPPORTED_ERROR(
                "backward: node %zu has %zu inputs, multi-input nodes are not supported", node_id.value, preds.size()));
        }

        const auto upstream = UNWRAP_OR_RETURN(backward_jobs[node_id.value].value().wait());

        auto d_input_job = std::visit(
            [&upstream](auto &layer) -> KernelJob<DeviceTensorConstViewf> { return layer.backward(upstream); },
            layers[node_id.value].kind);

        backward_jobs[preds[0].value] = std::move(d_input_job);
    }

    return ok(Void{});
}

auto update_layer(LayerKind &kind, DeviceTensorConstViewf forward_input, DeviceTensorConstViewf upstream,
                  std::vector<AdamState> &states, const AdamParameters &params, usize t) -> Result<Void, Error>
{
    return std::visit(
        [&](auto &l) -> Result<Void, Error> {
            using T = std::decay_t<decltype(l)>;
            if constexpr (std::is_same_v<T, DenseLayer> || std::is_same_v<T, Conv2DLayer>)
            {
                // weight_gradients() writes into the layer's own d_weights/d_biases (or
                // d_filters/d_biases) buffers; update()'s parameters() reads them back from
                // there, so nothing needs to be threaded through by hand here.
                auto wg_result = l.weight_gradients(forward_input, upstream).wait();
                if (wg_result.is_error())
                {
                    return error(wg_result.unwrap_error());
                }
                auto update_jobs = l.update(states, params, t);
                for (auto &result : wait_on(update_jobs))
                {
                    if (result.is_error())
                    {
                        return error(result.unwrap_error());
                    }
                }
            }
            return ok(Void{});
        },
        kind);
}

auto AdamState::create(const Extents &extents) -> Result<AdamState, Error>
{
    auto m = UNWRAP_OR_RETURN(DeviceOwningTensorf::zero(extents));
    auto v = UNWRAP_OR_RETURN(DeviceOwningTensorf::zero(extents));
    return ok(AdamState{std::move(m), std::move(v)});
}

auto AdamOptimizer::from_model(const Model &model, AdamParameters params) -> Result<AdamOptimizer, Error>
{
    AdamOptimizer optimizer{params, {}};

    for (const auto node_id : model.execution_order)
    {
        const auto &layer = model.layers[node_id.value];
        auto maybe_params = std::visit(
            [](const auto &l) -> std::optional<std::vector<ConstParameter>> {
                using T = std::decay_t<decltype(l)>;
                if constexpr (std::is_same_v<T, DenseLayer> || std::is_same_v<T, Conv2DLayer>)
                {
                    return l.parameters();
                }
                else
                {
                    return std::nullopt;
                }
            },
            layer.kind);

        if (!maybe_params.has_value())
        {
            continue;
        }

        std::vector<AdamState> states;
        states.reserve(maybe_params->size());
        for (const auto &param : *maybe_params)
        {
            states.push_back(UNWRAP_OR_RETURN(AdamState::create(param.value.to_extents())));
        }
        optimizer.states.emplace(node_id.value, std::move(states));
    }

    return ok(std::move(optimizer));
}

auto Model::step(AdamOptimizer &optimizer, usize t) -> Result<Void, Error>
{
    for (const auto node_id : execution_order)
    {
        auto &layer = layers[node_id.value];
        if (layer.is_frozen)
        {
            continue;
        }

        const auto &preds = layer_inputs[node_id.value];
        if (preds.empty())
        {
            continue;
        }

        auto it = optimizer.states.find(node_id.value);
        if (it == optimizer.states.end())
        {
            continue;
        }

        const auto forward_input = UNWRAP_OR_RETURN(forward_jobs[preds[0].value].value().wait());
        const auto upstream = UNWRAP_OR_RETURN(backward_jobs[node_id.value].value().wait());

        auto result = update_layer(layer.kind, forward_input, upstream, it->second, optimizer.params, t);
        if (result.is_error())
        {
            return error(result.unwrap_error());
        }
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

__global__ auto adam_update(const AdamParameters parameters, f32 m_hat_scale, f32 v_hat_scale,
                            DeviceTensorConstViewf d_weights, DeviceTensorViewf weights, DeviceTensorViewf m_weights,
                            DeviceTensorViewf v_weights) -> void
{
    usize i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= weights.element_count())
    {
        return;
    }

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

__global__ auto add_bias(DeviceTensorConstView1f biases, DeviceTensorView2f out) -> void
{
    const usize sample_index = blockIdx.y * blockDim.y + threadIdx.y;
    const usize col = blockIdx.x * blockDim.x + threadIdx.x;

    const usize sample_count = out.extents[0];
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
// - Concatenate Forward
// - Concatenate Backward
// - Concatenate Layer
//
// - Add Forward
// - Add Backward
// - Add Layer
//
// - ConvTranspose Forward
// - ConvTranspose Backward
// - ConvTranspose Layer
//
// - Tiled matmul
//
// - Pick device?
// - Sequential model
// - Multi-input
// - Multi-output
// - unet
// - Save weights
// - Load weights
// - Bake in loss into model
