// vika - a single-header CUDA library for training small convolutional networks.
//
// ---------------------------------------------------------------------------------------------
// Including it
// ---------------------------------------------------------------------------------------------
//
// Exactly one translation unit defines VIKA_IMPLEMENTATION before the include; every other one
// just includes it and gets declarations.
//
//     #define VIKA_IMPLEMENTATION
//     #include "vika.cuh"
//
// Both are .cu files: the header defines __global__ kernels, so nvcc has to compile it.
// Configuration, if the defaults do not fit: VIKA_MAX_RANK (6) caps a tensor's rank, and
// VIKA_MAX_ERROR_MESSAGE (192) the length of an error string. Define either before the include.
//
// ---------------------------------------------------------------------------------------------
// Training a model
// ---------------------------------------------------------------------------------------------
//
//     ComputationGraph graph{batch_size};              // batch_size is a capacity, see below
//     auto x = graph.input({2}).unwrap();              // extents after the batch dimension
//     x = graph.dense(x, 8).unwrap();
//     x = graph.sigmoid(x).unwrap();
//     x = graph.dense(x, 1).unwrap();
//     x = graph.sigmoid(x).unwrap();
//
//     auto model = graph.compile(x).unwrap();          // x is the output node
//     auto loss = MSELoss::with_extents({batch_size, 1}).unwrap();
//     auto optimizer = AdamOptimizer::from_model(model, {.learning_rate = 0.01f}).unwrap();
//
//     for (usize epoch = 0; epoch < epochs; ++epoch)
//     {
//         // inputs and targets are DeviceTensorConstViewf - see Tensors below
//         const f32 value = train_step(model, loss, inputs, targets, optimizer).unwrap();
//     }
//
// train_step is forward, loss, backward and optimizer step in one call. Run them separately and
// the order matters, because each reads what the one before it wrote:
//
//     const auto predictions = model.forward(inputs).unwrap();
//     const auto gradient = loss.backward(predictions, targets).wait().unwrap();
//     model.backward(gradient).unwrap();
//     model.step(optimizer).unwrap();
//
// Calling them out of order is reported, not undefined. The optimizer owns its own step count,
// so nothing needs to thread a t through the loop.
//
// ---------------------------------------------------------------------------------------------
// Layers on their own
// ---------------------------------------------------------------------------------------------
//
// Every layer works without a graph, and validates its own inputs when used that way:
//
//     auto layer = DenseLayer::randomized(batch_size, 4, 3, seed).unwrap();
//     const auto out = layer.forward({inputs}).wait().unwrap();
//     const auto grad = layer.backward(upstream)[0].wait().unwrap();
//
// forward() takes a vector because a layer may have several inputs (Add, Concat); backward()
// returns one gradient per input for the same reason. Both hand back a KernelJob, which is work
// already queued on the layer's stream - wait() blocks until it is done and yields a Result.
//
// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------
//
// Anything a caller can get wrong comes back as Result<T, Error>, never an exception and never a
// crash: bad shapes, bad graphs, a stride of zero, calling the passes out of order, CUDA itself
// failing. Result is [[nodiscard]], and unwrap() aborts on an error, so .unwrap() is a claim that
// this one cannot fail. describe() renders an error with its originating file and line.
//
// Two things do abort, both deliberately. VIKA_PANIC covers invariants whose violation means the
// library is broken rather than misused (unwrap() on an error, FixedVector bounds), and
// VIKA_ASSERT the same but compiled out under NDEBUG.
//
// ---------------------------------------------------------------------------------------------
// Tensors
// ---------------------------------------------------------------------------------------------
//
// HostTensor<T> owns host memory, DeviceOwningTensor<T> owns device memory, and DeviceTensorView
// borrows either - the views are what kernels take. All of them are move-only; copying host data
// takes an explicit clone(), and upload()/download() move between the two sides.
//
//     const auto host = HostTensorf::from({1, 2, 3, 4}, {2, 2}).unwrap();
//     const auto device = upload(host).unwrap();
//     const auto back = download(device).unwrap();
//
// Rank lives in the extents at runtime, not in the type, and every layer checks the rank it
// expects at its own boundary.
//
// ---------------------------------------------------------------------------------------------
// Batch size is a capacity
// ---------------------------------------------------------------------------------------------
//
// The batch_size a graph is compiled with sizes every buffer once, up front - a training step
// never allocates. A forward pass may then use any batch up to that capacity, and the rest of the
// pass follows it:
//
//     auto model = graph.compile(x).unwrap();                  // capacity 64
//     model.forward(inputs.view().first_n(8).unwrap().const_view()).unwrap();   // runs 8
//
// first_n() slices the leading dimension without copying, since row-major storage makes those
// rows a contiguous prefix. Exceeding the capacity is an error, not a reallocation.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// A precondition that can only be a bug in the calling code, so a release build should not pay to
// re-check it. Host-only, like VIKA_PANIC itself.
//
// The NDEBUG branch evaluates nothing but still counts as a use of what the condition names, so a
// variable existing only for an assert does not become an unused one under -Werror.
#ifdef NDEBUG
#define VIKA_ASSERT(expr, fmt, ...) ((void)sizeof((expr) ? 1 : 0))
#else
#define VIKA_ASSERT(expr, fmt, ...) VIKA_PANIC_IF(!(expr), fmt, ##__VA_ARGS__)
#endif

// Works in any function returning something constructible from Err<Error>: Result<T, Error> and
// KernelJob<T> directly, and std::vector<KernelJob<T>> - a layer's backward() - through its
// initializer-list constructor. Hence the braces, which are a no-op for the first two and the
// whole point for the third, so one macro covers all three.
#define VIKA_UNWRAP_OR_RETURN(expr)                                                                                    \
    ({                                                                                                                 \
        auto _res = (expr);                                                                                            \
        if (_res.is_error())                                                                                           \
        {                                                                                                              \
            return {error(_res.unwrap_error())};                                                                       \
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

// The read accessors are __host__ __device__ because Extents/Strides live inside
// DeviceTensorView, which kernels take by value. The mutators and at() cannot be: they report
// failure through VIKA_PANIC, and fprintf/exit have no device equivalent.
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

    __host__ __device__ auto operator[](usize idx) -> T &
    {
        return m_data[idx];
    }

    __host__ __device__ auto operator[](usize idx) const -> const T &
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

    __host__ __device__ auto front() -> T &
    {
        return m_data[0];
    }

    __host__ __device__ auto front() const -> const T &
    {
        return m_data[0];
    }

    __host__ __device__ auto back() -> T &
    {
        return m_data[m_size - 1];
    }

    __host__ __device__ auto back() const -> const T &
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

    __host__ __device__ auto size() const -> usize
    {
        return m_size;
    }

    static constexpr auto capacity() -> usize
    {
        return Capacity;
    }

    __host__ __device__ auto empty() const -> bool
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

    // Assignment, not placement new: every slot of m_data already holds a live T, and
    // constructing over one without destroying it leaks whatever it owned. Same reason pop_back()
    // and clear() only move m_size - the objects stay alive until the array itself dies.
    template <typename... Args>
    auto emplace_back(Args &&...args) -> T &
    {
        check_not_full();
        T *slot = &m_data[m_size++];
        *slot = T(std::forward<Args>(args)...);
        return *slot;
    }

    auto pop_back() -> void
    {
        VIKA_PANIC_IF(m_size == 0, "FixedVector::pop_back on an empty vector");
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
    // Zero-initialised because DeviceTensorView holds two of these and is copied whole into every
    // kernel launch, so entries past size() travel along as padding.
    T m_data[Capacity] = {};
    usize m_size;

    auto check_bounds(usize idx) const -> void
    {
        VIKA_PANIC_IF(idx >= m_size, "FixedVector::at index %zu out of bounds (size=%zu)", idx, m_size);
    }

    auto check_not_full() const -> void
    {
        VIKA_PANIC_IF(m_size >= Capacity, "FixedVector is at capacity (%zu)", Capacity);
    }
};

using Extents = FixedVector<usize, VIKA_MAX_RANK>;
using Strides = Extents;

template <typename Node>
using AdjacencyGraph = std::unordered_map<Node, std::vector<Node>>;

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
    // storage is a variant<T, E>, so ok and error are told apart by type alone - which a
    // Result<T, T> cannot do.
    static_assert(!std::is_same_v<T, E>, "Result's value and error types must differ - a Result<T, T> cannot tell "
                                         "an ok from an error");

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

// ===========================================================================
// Generic Utilities, Result-aware
// ===========================================================================

// As generic as the section above, and here only because they need Result and Error complete.
template <typename T, typename UnaryOperation>
auto map(const std::vector<T> &in, UnaryOperation op)
{
    using namespace std;
    using out_type = decltype(op(*begin(in)));
    std::vector<out_type> out;
    out.reserve(in.size());
    transform(begin(in), end(in), back_inserter(out), op);
    return out;
}

template <typename T, typename UnaryOperation>
auto try_map(const std::vector<T> &in, UnaryOperation op) -> Result<std::vector<decltype(op(in[0]).unwrap())>, Error>
{
    using out_type = decltype(op(in[0]).unwrap());
    std::vector<out_type> out;
    out.reserve(in.size());
    for (const auto &item : in)
    {
        auto res = op(item);
        if (res.is_error())
        {
            return error(res.unwrap_error());
        }
        out.push_back(std::move(res.unwrap()));
    }
    return ok(std::move(out));
}

template <typename T, typename Predicate>
auto all_of(const std::vector<T> &in, Predicate pred) -> bool
{
    using namespace std;
    return all_of(begin(in), end(in), pred);
}

template <typename Node>
auto topological_sort(const AdjacencyGraph<Node> &adj) -> Result<std::vector<Node>, Error>
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
        return ok(std::move(order));
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

// Element count, rejecting the three ways extents can be unusable: empty, containing a zero, or
// overflowing usize. Every host and device tensor factory funnels through this, the device ones
// via checked_byte_count below.
auto checked_size(const Extents &extents) -> Result<usize, Error>;

template <typename T>
inline auto checked_byte_count(const Extents &extents) -> Result<usize, Error>
{
    const usize elements = VIKA_UNWRAP_OR_RETURN(checked_size(extents));
    if (elements > std::numeric_limits<usize>::max() / sizeof(T))
    {
        return error(VIKA_SHAPE_ERROR("byte count overflows usize: %zu elements of %zu bytes", elements, sizeof(T)));
    }
    return ok(elements * sizeof(T));
}

// Extents as text, for error messages - "{2, 3}". A std::string is fine here: error formatting is
// not a hot path, and Error::make copies the result into its own fixed buffer regardless.
inline auto describe(const Extents &extents) -> std::string
{
    std::string out = "{";
    for (usize i = 0; i < extents.size(); ++i)
    {
        out += (i == 0 ? "" : ", ") + std::to_string(extents[i]);
    }
    return out + "}";
}

// Output extent of one spatial dimension for a sliding window. Convolution and pooling
// share this geometry; pooling is simply the padding == 0 case.
// Requires stride > 0 and input + 2 * padding >= window.
inline constexpr auto window_output_extent(usize input, usize window, usize stride, usize padding) -> usize
{
    return (input + 2 * padding - window) / stride + 1;
}

// Inverse of window_output_extent: the input size a sliding window of the given geometry would
// need to produce `input` as its own output - i.e. the output size of a ConvTranspose2D, which is
// exactly a Conv2D run backward (see ConvTranspose2DLayer's own doc comment). Requires
// stride > 0 and (input - 1) * stride + window >= 2 * padding.
inline constexpr auto transposed_window_output_extent(usize input, usize window, usize stride, usize padding) -> usize
{
    return (input - 1) * stride + window - 2 * padding;
}

// Validates that every entry in `extents` agrees on every dimension except the last, and returns
// the concatenated output shape: extents[0] with the last dimension replaced by the sum of every
// input's last dimension. Shared by ComputationGraph::concat() (validating declared graph shapes)
// and ConcatLayer::with_extents() (validating actual constructed shapes - layers are first-class
// and must defend themselves even when used standalone, same reason Dense/Conv2D validate at both
// their graph builder and their factory).
inline auto concat_output_extents(const std::vector<Extents> &extents) -> Result<Extents, Error>
{
    if (extents.size() < 2)
    {
        return error(VIKA_SHAPE_ERROR("concat: expects at least 2 inputs, got %zu", extents.size()));
    }

    const auto rank = extents[0].size();
    if (rank == 0)
    {
        return error(VIKA_SHAPE_ERROR("concat: inputs must be at least rank 1, got rank 0"));
    }

    usize total_last_dim = 0;
    for (usize i = 0; i < extents.size(); ++i)
    {
        if (extents[i].size() != rank)
        {
            return error(VIKA_SHAPE_ERROR("concat: input %zu is rank %zu, expected rank %zu matching input 0", i,
                                          extents[i].size(), rank));
        }
        for (usize d = 0; d + 1 < rank; ++d)
        {
            if (extents[i][d] != extents[0][d])
            {
                return error(VIKA_SHAPE_ERROR("concat: input %zu's dimension %zu is %zu, expected %zu matching input 0",
                                              i, d, extents[i][d], extents[0][d]));
            }
        }
        total_last_dim += extents[i][rank - 1];
    }

    Extents output_extents = extents[0];
    output_extents[rank - 1] = total_last_dim;
    return ok(output_extents);
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
    // Move-only: duplicating the buffer takes an explicit clone(), the same question
    // DeviceOwningTensor's unique_ptr forces on its own.
    HostTensor(const HostTensor &) = delete;
    auto operator=(const HostTensor &) -> HostTensor & = delete;
    HostTensor(HostTensor &&) noexcept = default;
    auto operator=(HostTensor &&) noexcept -> HostTensor & = default;

    auto clone() const -> Self
    {
        return HostTensor(std::vector<T>(_data), _extents);
    }

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
        VIKA_ASSERT(_extents.size() == 2, "HostTensor: 2-argument accessor on a rank %zu tensor", _extents.size());
        return _data[r * _extents[1] + c];
    }

    auto operator()(usize r, usize c) const -> const T &
    {
        VIKA_ASSERT(_extents.size() == 2, "HostTensor: 2-argument accessor on a rank %zu tensor", _extents.size());
        return _data[r * _extents[1] + c];
    }

    auto operator()(usize n, usize h, usize w, usize c) -> T &
    {
        VIKA_ASSERT(_extents.size() == 4, "HostTensor: 4-argument accessor on a rank %zu tensor", _extents.size());
        const auto height = _extents[1];
        const auto width = _extents[2];
        const auto channels = _extents[3];
        return _data[((n * height + h) * width + w) * channels + c];
    }

    auto operator()(usize n, usize h, usize w, usize c) const -> const T &
    {
        VIKA_ASSERT(_extents.size() == 4, "HostTensor: 4-argument accessor on a rank %zu tensor", _extents.size());
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

    static auto zero(const Extents &extents) -> Result<Self, Error>
    {
        const usize count = VIKA_UNWRAP_OR_RETURN(checked_size(extents));
        return ok(HostTensor(std::vector<T>(count, T{}), extents));
    }

    static auto empty(const Extents &extents) -> Result<Self, Error>
    {
        const usize count = VIKA_UNWRAP_OR_RETURN(checked_size(extents));
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
        const usize count = VIKA_UNWRAP_OR_RETURN(checked_size(extents));
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
        // Guaranteed by checked_size in every factory, so these only fire if one stops calling it.
        VIKA_ASSERT(!_extents.empty(), "HostTensor: constructed with empty extents");
        VIKA_ASSERT(element_count(_extents) != 0, "HostTensor: constructed with a zero extent");
        VIKA_ASSERT(_data.size() == element_count(_extents),
                    "HostTensor: %zu elements of data for extents describing %zu", _data.size(),
                    element_count(_extents));
    }

  private:
    std::vector<T> _data{};
    Extents _extents{};
};

// Unsuffixed, for the reason step 13 gave when it dropped the device-side suffixes: rank lives at
// runtime in the extents, never in the type, so HostTensor2f and HostTensor4f were always the
// same HostTensor<f32> wearing different names for a distinction that does not exist.
using HostTensorf = HostTensor<f32>;
using HostTensoru = HostTensor<u32>;

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
        const usize bytes = VIKA_UNWRAP_OR_RETURN(vika::checked_byte_count<T>(extents));

        T *ptr = nullptr;
        VIKA_RETURN_ON_CUDA_ERROR(cudaMalloc(&ptr, bytes));
        return ok(Self(ptr, extents));
    }

    static auto from(const std::vector<T> &data, const Extents &extents) -> Result<Self, Error>
    {
        const usize count = VIKA_UNWRAP_OR_RETURN(vika::checked_element_count(extents));
        if (data.size() != count)
        {
            return error(
                VIKA_SHAPE_ERROR("device tensor data holds %zu elements but extents describe %zu", data.size(), count));
        }

        auto tensor = VIKA_UNWRAP_OR_RETURN(empty(extents));
        VIKA_RETURN_ON_CUDA_ERROR(cudaMemcpy(tensor.data(), data.data(), tensor.byte_count(), cudaMemcpyHostToDevice));
        return ok(std::move(tensor));
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
        auto tensor = VIKA_UNWRAP_OR_RETURN(empty(extents));
        VIKA_RETURN_ON_CUDA_ERROR(cudaMemset(tensor.data(), 0, tensor.byte_count()));
        return ok(std::move(tensor));
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
        return error(VIKA_SHAPE_ERROR("copy device -> host: source is %s, destination is %s",
                                      describe(src.extents()).c_str(), describe(dst.extents()).c_str()));
    }

    VIKA_RETURN_ON_CUDA_ERROR(cudaMemcpy(dst.data(), src.data(), src.byte_count(), cudaMemcpyDeviceToHost));
    return ok(Void{});
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto copy(const DeviceTensorView<const T> &src, HostTensor<T> &dst) -> Result<Void, Error>
{
    if (dst.extents() != src.extents)
    {
        return error(VIKA_SHAPE_ERROR("copy view -> host: source is %s, destination is %s",
                                      describe(src.extents).c_str(), describe(dst.extents()).c_str()));
    }

    VIKA_RETURN_ON_CUDA_ERROR(cudaMemcpy(dst.data(), src.data, src.byte_count(), cudaMemcpyDeviceToHost));
    return ok(Void{});
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto copy(const HostTensor<T> &src, DeviceOwningTensor<T> &dst) -> Result<Void, Error>
{
    if (src.extents() != dst.extents())
    {
        return error(VIKA_SHAPE_ERROR("copy host -> device: source is %s, destination is %s",
                                      describe(src.extents()).c_str(), describe(dst.extents()).c_str()));
    }

    VIKA_RETURN_ON_CUDA_ERROR(cudaMemcpy(dst.data(), src.data(), dst.byte_count(), cudaMemcpyHostToDevice));
    return ok(Void{});
}

// Device-to-device, async on the given stream - unlike the host<->device overloads above, which
// are synchronous. Views rather than owning tensors, because every caller only has a view: either
// sliced by first_n() or handed back from a KernelJob.
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto copy(DeviceTensorView<const T> src, DeviceTensorView<T> dst, cudaStream_t stream) -> Result<Void, Error>
{
    if (src.extents != dst.extents)
    {
        return error(VIKA_SHAPE_ERROR("copy device -> device: source is %s, destination is %s",
                                      describe(src.extents).c_str(), describe(dst.extents).c_str()));
    }

    VIKA_RETURN_ON_CUDA_ERROR(cudaMemcpyAsync(dst.data, src.data, src.byte_count(), cudaMemcpyDeviceToDevice, stream));
    return ok(Void{});
}

// Async zero-fill on the given stream.
template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto zero(DeviceTensorView<T> dst, cudaStream_t stream) -> Result<Void, Error>
{
    VIKA_RETURN_ON_CUDA_ERROR(cudaMemsetAsync(dst.data, 0, dst.byte_count(), stream));
    return ok(Void{});
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto upload(const HostTensor<T> &src) -> Result<DeviceOwningTensor<T>, Error>
{
    auto dst = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensor<T>::empty(src.extents()));
    VIKA_UNWRAP_OR_RETURN(copy(src, dst));
    return ok(std::move(dst));
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
auto download(const DeviceTensorView<const T> &src) -> Result<HostTensor<T>, Error>
{
    auto dst = VIKA_UNWRAP_OR_RETURN(HostTensor<T>::empty(src.extents));
    VIKA_UNWRAP_OR_RETURN(copy(src, dst));
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
    DeviceTensorView(T *data_, const Extents &extents_) : data(data_), extents(extents_)
    {
        // Row-major: the last axis is contiguous, each earlier stride the product of the extents
        // after it. Copied from extents only to get one entry per dimension; all are overwritten.
        strides = extents;
        usize stride = 1;
        for (usize i = extents.size(); i-- > 0;)
        {
            strides[i] = stride;
            stride *= extents[i];
        }
    }

    T *data = nullptr;
    Extents extents{};
    Strides strides{};

    __host__ __device__ auto rank() const -> usize
    {
        return extents.size();
    }

    // Slices the leading dimension to its first n rows. Row-major storage makes those a
    // contiguous prefix of the same buffer, so this reinterprets extents and copies nothing.
    auto first_n(usize n) const -> Result<DeviceTensorView, Error>
    {
        if (n > extents[0])
        {
            return error(VIKA_SHAPE_ERROR("first_n: requested %zu rows but tensor only has %zu", n, extents[0]));
        }
        auto sliced_extents = extents;
        sliced_extents[0] = n;
        return ok(DeviceTensorView(data, sliced_extents));
    }

    auto const_view() const -> DeviceTensorView<const T>
    {
        return DeviceTensorView<const T>(data, extents);
    }

    __host__ __device__ inline usize element_count() const
    {
        usize count = 1;
        for (usize i = 0; i < extents.size(); ++i)
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

// Unsuffixed on purpose: rank lives in the extents at runtime, never in the type, and each layer
// checks the rank it expects at its own boundary. Kept dynamic rather than templating tensors on a
// compile-time rank, so loading an unknown-until-runtime rank from disk stays one type.
using DeviceOwningTensorf = DeviceOwningTensor<f32>;
using DeviceOwningTensoru = DeviceOwningTensor<u32>;

using DeviceTensorViewf = DeviceTensorView<f32>;
using DeviceTensorViewu = DeviceTensorView<u32>;

using DeviceTensorConstViewf = DeviceTensorView<const f32>;
using DeviceTensorConstViewu = DeviceTensorView<const u32>;

// Rank 2 only: the swap touches extents[1]/strides[1], which on a lower-rank view are the zeroed
// slots past size() - yielding a view of extent 0 that every kernel then skips silently.
auto transposed(const DeviceTensorConstViewf &view) -> Result<DeviceTensorConstViewf, Error>;

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

// Owns a CUDA stream, move-only via unique_ptr. Created non-blocking so it never implicitly
// synchronizes with the legacy default stream, as a plain cudaStreamCreate() stream would.
class Stream
{
    using Self = Stream;

  public:
    static auto create() -> Result<Self, Error>
    {
        cudaStream_t stream = nullptr;
        VIKA_RETURN_ON_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
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
    // Implicit on purpose, mirroring Result's: it lets a function returning KernelJob<T> write
    // `return error(...)` without naming T again.
    KernelJob(Err<Error> err) : result(std::move(err)), stream(nullptr)
    {
    }

    // Work that was enqueued on a stream.
    static auto launched(T value, cudaStream_t stream) -> KernelJob
    {
        return KernelJob(std::move(value), stream);
    }

    // A request rejected before anything was enqueued, so there is no stream to wait on. Holding
    // a Result rather than a value plus a flag means there is no placeholder value to read by
    // mistake, and is_error() answers without synchronising.
    static auto failed(Error err) -> KernelJob
    {
        return KernelJob(error(std::move(err)));
    }

    // A value that was never asynchronous - a passthrough view, a host-supplied gradient seeding
    // a backward pass. wait() sees the null stream and skips synchronising, rather than
    // synchronising the legacy default stream for work nobody queued.
    static auto ready(T value) -> KernelJob
    {
        return KernelJob(std::move(value), nullptr);
    }

    auto wait() const -> Result<T, Error>
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

// A thread's global index along one grid dimension. The cast matters: blockIdx and blockDim are
// both unsigned int, so `blockIdx.x * blockDim.x` computes in 32 bits and wraps before any
// widening on the receiving end can help. It takes 4.29e9 threads to wrap - 17 GB of f32, so not
// on a 12 GB card but ordinary on an A100 - and the wrapped index still passes an
// element_count() check, so the thread silently works on the wrong element.
__device__ inline auto global_thread_x() -> usize
{
    return (usize)blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline auto global_thread_y() -> usize
{
    return (usize)blockIdx.y * blockDim.y + threadIdx.y;
}

// The grid needed to cover x * y * z items with blocks of `block`. Counts are usize because an
// element count needs 64 bits on a large card; dim3 is unsigned int because that is CUDA's only
// constructor. This is the one place the two meet, so the one place that narrows - overflowing a
// u32 block count would take ~1e12 elements, 4 TB at f32.
inline auto grid_covering(dim3 block, usize x, usize y = 1, usize z = 1) -> dim3
{
    const auto blocks = [](usize count, u32 block_size) -> u32 { return (u32)((count + block_size - 1) / block_size); };
    return dim3(blocks(x, block.x), blocks(y, block.y), blocks(z, block.z));
}

// The launch geometry every [N, H, W, C] layer uses: one thread per output column and row, with
// the grid's z dimension covering batch * channels. Sized from the extents of the tensor the
// kernel writes - taking them from the other one silently under-covers it whenever the two differ,
// which is what 76ee8e7 fixed in DenseLayer::weight_gradients.
inline auto nhwc_grid(dim3 block, const Extents &target, usize batch) -> dim3
{
    return grid_covering(block, target.at(2), target.at(1), batch * target.at(3));
}

// Launches `kernel` and reads cudaGetLastError() immediately, because cudaStreamSynchronize -
// what KernelJob::wait() checks - does not report launch-configuration failures such as
// cudaErrorInvalidConfiguration. A rejected launch becomes a failed job rather than a job holding
// data no kernel produced.
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

template <typename T>
auto wait_on(const KernelJob<T> &job) -> Result<T, Error>
{
    return job.wait();
}

template <typename T>
auto wait_on_all(const std::vector<KernelJob<T>> &jobs) -> std::vector<Result<T, Error>>
{
    return vika::map(jobs, wait_on<T>);
}

// ===========================================================================
// Parameters and Optimizer
// ===========================================================================

struct AdamParameters
{
    f32 learning_rate = 1e-1f;
    f32 beta1 = 0.9f;
    f32 beta2 = 0.999f;
    f32 epsilon = 1e-8f;
};

// One trainable tensor: the optimizer mutates value in place and only ever reads grad. Layers
// hand out their whole parameter list this way, so no caller needs to know that Dense calls them
// weights/biases and Conv2D filters/biases.
struct ParameterView
{
    DeviceTensorViewf value;
    DeviceTensorConstViewf grad;
};

// What the const parameters() overload returns, for callers that only inspect - discovering
// shapes, say. Must stay in the same order and count as ParameterView: AdamState is positional.
struct ConstParameterView
{
    DeviceTensorConstViewf value;
    DeviceTensorConstViewf grad;
};

// What a layer actually stores. Same view()/const_view() names as DeviceOwningTensor, one level
// up: a parameter bundled with its gradient.
struct OwningParameter
{
    DeviceOwningTensorf value;
    DeviceOwningTensorf grad;

    auto view() -> ParameterView
    {
        return {value.view(), grad.const_view()};
    }

    auto const_view() const -> ConstParameterView
    {
        return {value.const_view(), grad.const_view()};
    }
};

// Per parameter, not per layer: a layer with N parameters gets N independent AdamStates
// (see Layer::parameters()), so adding a layer type with a different parameter count needs no
// changes here.
struct AdamState
{
    DeviceOwningTensorf m;
    DeviceOwningTensorf v;

    static auto create(const Extents &extents) -> Result<AdamState, Error>;
    static auto from_parameters(const ConstParameterView &parameters) -> Result<AdamState, Error>;
};

// Launches one adam_update per parameter on the given stream and returns every job unresolved
// (mirroring forward()/backward()/weight_gradients()): the caller decides when to wait, so
// independent parameters - and independent layers, each on their own stream - can all be
// in flight before anything blocks.
auto update_parameters(std::vector<ParameterView> &parameters, std::vector<AdamState> &states, cudaStream_t stream,
                       const AdamParameters &params, usize t) -> std::vector<KernelJob<Void>>;

// ===========================================================================
// Kernels
// ===========================================================================

__global__ auto matmul_kernel(DeviceTensorConstViewf a, DeviceTensorConstViewf b, DeviceTensorViewf out) -> void;

// filters: [kH, kW, C_in, C_out], inputs: [N, H, W, C_in], out: [N, out_H, out_W, C_out]
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C_out), block: (bx, by, 1)
__global__ auto conv_forward(DeviceTensorConstViewf inputs, DeviceTensorConstViewf filters,
                             DeviceTensorConstViewf biases, DeviceTensorViewf out, usize stride, usize padding) -> void;

// filters: [kH, kW, C_in, C_out], upstream: [N, out_H, out_W, C_out], d_inputs: [N, H, W, C_in]
// grid: (ceil(W_in/bx), ceil(H_in/by), N*C_in), block: (bx, by, 1)
__global__ auto conv_backward(DeviceTensorConstViewf upstream, DeviceTensorConstViewf filters,
                              DeviceTensorViewf d_inputs, usize stride, usize padding) -> void;

// inputs: [N, H, W, C_in], upstream: [N, out_H, out_W, C_out], d_filters: [kH, kW, C_in, C_out]
// 1D grid over all filter elements
__global__ auto conv_weight_gradients(DeviceTensorConstViewf inputs, DeviceTensorConstViewf upstream,
                                      DeviceTensorViewf d_filters, usize stride, usize padding) -> void;

// upstream: [N, out_H, out_W, C_out], d_biases: [C_out], 1D grid over C_out
__global__ auto conv_bias_gradients(DeviceTensorConstViewf upstream, DeviceTensorViewf d_biases) -> void;

// ConvTranspose2D is Conv2D run inside out: its forward is exactly conv_backward's data-gradient
// computation, and its backward is exactly conv_forward. That duality is why only the weight and
// data gradients need their own kernels - conv_bias_gradients is a per-output-channel sum either
// way - and why filters are stored [kH, kW, C_out, C_in], swapped against Conv2DLayer, so the
// reused indexing lands on the same taps.
//
// inputs: [N, H_in, W_in, C_in] (the small tensor), filters: [kH, kW, C_out, C_in],
// biases: [C_out], out: [N, H_out, W_out, C_out] (the large tensor)
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C_out), block: (bx, by, 1)
__global__ auto conv_transpose_forward(DeviceTensorConstViewf inputs, DeviceTensorConstViewf filters,
                                       DeviceTensorConstViewf biases, DeviceTensorViewf out, usize stride,
                                       usize padding) -> void;

// upstream: [N, H_out, W_out, C_out] (the large gradient), filters: [kH, kW, C_out, C_in],
// d_inputs: [N, H_in, W_in, C_in] (the small gradient)
// grid: (ceil(W_in/bx), ceil(H_in/by), N*C_in), block: (bx, by, 1)
__global__ auto conv_transpose_backward(DeviceTensorConstViewf upstream, DeviceTensorConstViewf filters,
                                        DeviceTensorViewf d_inputs, usize stride, usize padding) -> void;

// inputs: [N, H_in, W_in, C_in] (this layer's own forward input, the small tensor), upstream:
// [N, H_out, W_out, C_out] (the large upstream gradient), d_filters: [kH, kW, C_out, C_in]
// 1D grid over all filter elements
__global__ auto conv_transpose_weight_gradients(DeviceTensorConstViewf inputs, DeviceTensorConstViewf upstream,
                                                DeviceTensorViewf d_filters, usize stride, usize padding) -> void;

// inputs: [N, H, W, C], out: [N, out_H, out_W, C], argmax: [N, out_H, out_W, C]
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C), block: (bx, by, 1)
__global__ auto maxpool_forward(DeviceTensorConstViewf inputs, DeviceTensorViewf out, DeviceTensorViewu argmax,
                                usize pool_h, usize pool_w, usize stride) -> void;

// upstream: [N, out_H, out_W, C], argmax: [N, out_H, out_W, C], d_inputs: [N, H, W, C]
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C), block: (bx, by, 1)
__global__ auto maxpool_backward(DeviceTensorConstViewf upstream, DeviceTensorConstViewu argmax,
                                 DeviceTensorViewf d_inputs) -> void;

// Nearest-neighbor upsampling: out(n, oh, ow, c) = inputs(n, oh/scale, ow/scale, c), i.e. every
// input pixel is replicated into a scale x scale block of output pixels.
// inputs: [N, H, W, C], out: [N, H*scale, W*scale, C]
// grid: (ceil(W_out/bx), ceil(H_out/by), N*C), block: (bx, by, 1)
__global__ auto upsample2d_forward(DeviceTensorConstViewf inputs, DeviceTensorViewf out, usize scale) -> void;

// Reverse of upsample2d_forward: unlike maxpool_backward, this needs no atomics and no prior
// zero-fill of d_inputs - every output pixel maps to exactly one input pixel (no overlap, unlike
// pooling with stride < window), so gathering is one thread per *input* pixel summing its own
// scale x scale block of upstream, each writing its own d_inputs slot exactly once.
// upstream: [N, H*scale, W*scale, C], d_inputs: [N, H, W, C]
// grid: (ceil(W_in/bx), ceil(H_in/by), N*C), block: (bx, by, 1)
__global__ auto upsample2d_backward(DeviceTensorConstViewf upstream, DeviceTensorViewf d_inputs, usize scale) -> void;

__global__ auto uniform_tensor_kernel(DeviceTensorViewf tensor, u32 seed) -> void;
__global__ auto xavier_tensor_kernel(DeviceTensorViewf tensor, u32 seed, f32 limit) -> void;
__global__ auto sigmoid_forward(DeviceTensorConstViewf a, DeviceTensorViewf out) -> void;
__global__ auto sigmoid_backward(DeviceTensorConstViewf a, DeviceTensorConstViewf upstream,
                                 DeviceTensorViewf out) -> void;

// Normalizes along the last axis only - both tensors treated as flattened [rows, width]
// regardless of actual rank, same convention as concat_copy/concat_split, since row-major storage
// makes the last axis the contiguous one to reduce over. One thread per row (not per element):
// each row's max/sum/normalize all depend on every other element in that same row, so there is
// nothing to parallelize below row granularity without a shared-memory reduction, which nothing
// else in this file uses either (see sum_rows' identical one-thread-per-row shape).
__global__ auto softmax_forward(DeviceTensorConstViewf input, DeviceTensorViewf out) -> void;

// outputs: this layer's own forward() result (the softmax probabilities), not its input - the
// Jacobian-vector product only needs y, never the pre-softmax input. d_input_i = y_i * (upstream_i
// - dot), where dot = sum_j(y_j * upstream_j) over the same row.
__global__ auto softmax_backward(DeviceTensorConstViewf outputs, DeviceTensorConstViewf upstream,
                                 DeviceTensorViewf out) -> void;

__global__ auto adam_update(const AdamParameters parameters, f32 m_hat_scale, f32 v_hat_scale,
                            DeviceTensorConstViewf d_weights, DeviceTensorViewf weights, DeviceTensorViewf m_weights,
                            DeviceTensorViewf v_weights) -> void;

__global__ auto add_bias(DeviceTensorConstViewf biases, DeviceTensorViewf out) -> void;

__global__ auto sum_rows(DeviceTensorConstViewf matrix, DeviceTensorViewf out) -> void;

__global__ auto mse_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets, DeviceTensorViewf out)
    -> void;
__global__ auto mse_gradient_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets,
                                    DeviceTensorViewf out) -> void;

// Averaged per sample (divides by row count, i.e. batch size), not per element like mse_kernel
// above - the standard cross-entropy convention, since targets are one-hot along the last axis
// and dividing by element count too would shrink the loss as the class count grows for no
// meaningful reason. predictions are expected to already be probabilities (e.g. Softmax's own
// output), not raw logits - this loss does not fuse a softmax internally.
__global__ auto cce_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets, DeviceTensorViewf out)
    -> void;
__global__ auto cce_gradient_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets,
                                    DeviceTensorViewf out) -> void;

// accum[i] += delta[i] for every element. Two unrelated uses share this: summing multiple
// gradient contributions into one node's upstream when a node has more than one consumer (see
// Model::accumulate_output_gradient), and AddLayer's forward (out = a + b, computed as out = a
// then accumulate_into(b, out)). Safe to have accum also be one of the buffers a caller reads
// elsewhere: every thread only ever touches its own index, so there is no cross-element
// read/write to race.
__global__ auto accumulate_into(DeviceTensorConstViewf delta, DeviceTensorViewf accum) -> void;

// Copies `src` into a column-offset slice of `dst` along the last axis - Concat's forward, one
// launch per input. Both tensors are treated as flattened [rows, width] regardless of actual
// rank: row-major storage makes the last axis contiguous, so every other dimension (including
// batch) just stacks as more rows. One thread per src element (src is the narrow side).
__global__ auto concat_copy(DeviceTensorConstViewf src, DeviceTensorViewf dst, usize dst_col_offset) -> void;

// Reverse of concat_copy: copies a column-offset slice of `src` (the wide upstream gradient) into
// `dst` (one input's own narrow d_input buffer) - Concat's backward, one launch per input. One
// thread per dst element (dst is the narrow side here, unlike concat_copy's src).
__global__ auto concat_split(DeviceTensorConstViewf src, usize src_col_offset, DeviceTensorViewf dst) -> void;

__host__ __device__ auto sigmoid(f32 x) -> f32;

auto uniform_tensor(DeviceTensorViewf tensor, u32 seed, cudaStream_t stream) -> KernelJob<Void>;
auto xavier_tensor(DeviceTensorViewf tensor, u32 seed, usize fan_in, usize fan_out, cudaStream_t stream)
    -> KernelJob<Void>;

// =============================================================================
// Layers
// =============================================================================

struct DenseLayer
{
    static auto with_weights(usize batch_size, DeviceOwningTensorf weights, DeviceOwningTensorf biases)
        -> Result<DenseLayer, Error>;

    static auto randomized(usize batch_size, usize input_features, usize neuron_count, u32 seed)
        -> Result<DenseLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>;

    auto backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    auto weight_gradients(const DeviceTensorConstViewf &inputs, const DeviceTensorConstViewf &upstream)
        -> KernelJob<std::tuple<DeviceTensorConstViewf, DeviceTensorConstViewf>>;

    auto parameters() -> std::vector<ParameterView>;
    auto parameters() const -> std::vector<ConstParameterView>;

    auto update(std::vector<AdamState> &states, const AdamParameters &params, usize t) -> std::vector<KernelJob<Void>>;

    DeviceOwningTensorf outputs;
    DeviceOwningTensorf d_inputs;
    OwningParameter weights;
    OwningParameter biases;

    Stream stream;
};

struct SigmoidLayer
{
    static auto with_extents(const Extents &extents) -> Result<SigmoidLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>;

    auto backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    DeviceOwningTensorf outputs;
    DeviceOwningTensorf d_inputs;

    Stream stream;
};

struct SoftmaxLayer
{
    static auto with_extents(const Extents &extents) -> Result<SoftmaxLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>;

    auto backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    DeviceOwningTensorf outputs;
    DeviceOwningTensorf d_inputs;

    Stream stream;
};

struct Flatten2DLayer
{
    // Rank 2 or higher: below that there is nothing to flatten, and output_extents() reads
    // extents[0]. The only layer factory that could not fail until it started checking.
    static auto with_extents(const Extents &extents) -> Result<Flatten2DLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) const -> KernelJob<DeviceTensorConstViewf>;

    auto backward(const DeviceTensorConstViewf &upstream) const -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    // The shape forward() produces. Stands in for the outputs.extents() a buffer-owning layer
    // checks its upstream against; this layer owns no buffer, it only reinterprets its input's.
    auto output_extents() const -> Extents;

    Extents extents;
};

struct Conv2DLayer
{
    // filters: [kH, kW, C_in, C_out]
    static auto with_weights(usize batch_size, usize input_height, usize input_width, DeviceOwningTensorf filters,
                             DeviceOwningTensorf biases, usize stride, usize padding) -> Result<Conv2DLayer, Error>;

    static auto randomized(usize batch_size, usize input_height, usize input_width, usize kH, usize kW, usize C_in,
                           usize C_out, usize stride, usize padding, u32 seed) -> Result<Conv2DLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>;

    auto backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    auto weight_gradients(const DeviceTensorConstViewf &inputs, const DeviceTensorConstViewf &upstream)
        -> KernelJob<std::tuple<DeviceTensorConstViewf, DeviceTensorConstViewf>>;

    auto parameters() -> std::vector<ParameterView>;
    auto parameters() const -> std::vector<ConstParameterView>;

    auto update(std::vector<AdamState> &states, const AdamParameters &params, usize t) -> std::vector<KernelJob<Void>>;

    DeviceOwningTensorf outputs;
    DeviceOwningTensorf d_inputs;
    OwningParameter filters;
    OwningParameter biases;

    usize stride;
    usize padding;

    Stream stream;
};

// Learned upsampling - see conv_transpose_forward's doc comment for the duality with Conv2D's
// backward that this whole layer is built on.
struct ConvTranspose2DLayer
{
    // filters: [kH, kW, C_out, C_in] - C_out/C_in swapped vs Conv2DLayer's [kH, kW, C_in, C_out],
    // see conv_transpose_forward's doc comment for why.
    static auto with_weights(usize batch_size, usize input_height, usize input_width, DeviceOwningTensorf filters,
                             DeviceOwningTensorf biases, usize stride, usize padding)
        -> Result<ConvTranspose2DLayer, Error>;

    static auto randomized(usize batch_size, usize input_height, usize input_width, usize kH, usize kW, usize C_in,
                           usize C_out, usize stride, usize padding, u32 seed) -> Result<ConvTranspose2DLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>;

    auto backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    auto weight_gradients(const DeviceTensorConstViewf &inputs, const DeviceTensorConstViewf &upstream)
        -> KernelJob<std::tuple<DeviceTensorConstViewf, DeviceTensorConstViewf>>;

    auto parameters() -> std::vector<ParameterView>;
    auto parameters() const -> std::vector<ConstParameterView>;

    auto update(std::vector<AdamState> &states, const AdamParameters &params, usize t) -> std::vector<KernelJob<Void>>;

    DeviceOwningTensorf outputs;
    DeviceOwningTensorf d_inputs;
    OwningParameter filters;
    OwningParameter biases;

    usize stride;
    usize padding;

    Stream stream;
};

struct MaxPool2DLayer
{
    static auto with_extents(usize batch_size, usize input_height, usize input_width, usize channels, usize pool_h,
                             usize pool_w, usize stride) -> Result<MaxPool2DLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>;

    auto backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    DeviceOwningTensorf outputs;
    DeviceOwningTensoru argmax;
    DeviceOwningTensorf d_inputs;

    usize pool_h;
    usize pool_w;
    usize stride;

    Stream stream;
};

struct Upsample2DLayer
{
    static auto with_extents(usize batch_size, usize input_height, usize input_width, usize channels, usize scale)
        -> Result<Upsample2DLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>;

    auto backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    DeviceOwningTensorf outputs;
    DeviceOwningTensorf d_inputs;

    usize scale;

    Stream stream;
};

// N-ary: elementwise sum of every input, all the same shape. backward() needs no workspace or
// kernel launch at all - d/dx_i(sum) = 1 for every input, so the same upstream gradient flows
// unchanged to all of them (see AddLayer::backward). forward() does need its own outputs buffer:
// it can't write into any input's own buffer in place, since another consumer downstream of that
// input (fan-out) may still need its value unchanged.
struct AddLayer
{
    // input_count must be at least 2 - stored because backward() has no other way to know how
    // many (identical) gradients to return; nothing about its own signature communicates it.
    static auto with_extents(const Extents &extents, usize input_count) -> Result<AddLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>;

    // input_count jobs out, not one - the same upstream gradient routed to every input.
    auto backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    DeviceOwningTensorf outputs;
    usize input_count;

    Stream stream;
};

// N-ary, joins along the last axis only. Unlike AddLayer, genuinely needs a kernel launch in
// both directions (concat_copy/concat_split) - concatenation is a strided row/column copy, not
// elementwise, so nothing here can reuse an existing kernel the way Add's forward did. Also
// unlike AddLayer, backward doesn't return the same view N times: it must produce N distinct
// buffers (one per original input, each that input's own shape), so this layer owns d_inputs as
// a vector of owning tensors instead of a single shared one.
struct ConcatLayer
{
    static auto with_extents(const std::vector<Extents> &input_extents) -> Result<ConcatLayer, Error>;

    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>;

    // One job per input, each from its own d_inputs[i] buffer - unlike AddLayer's identical jobs,
    // these differ (each is a different slice of upstream, split back to its own input's shape).
    auto backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>;

    DeviceOwningTensorf outputs;
    std::vector<DeviceOwningTensorf> d_inputs;

    Stream stream;
};

struct MSELoss
{
    static auto with_extents(const Extents &extents) -> Result<MSELoss, Error>;

    auto forward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
        -> KernelJob<DeviceTensorConstViewf>;

    auto backward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
        -> KernelJob<DeviceTensorConstViewf>;

    DeviceOwningTensorf loss;
    DeviceOwningTensorf d_inputs;

    Stream stream;
};

// Categorical cross-entropy. Same forward(predictions, targets)/backward(predictions, targets)
// shape as MSELoss - see cce_kernel's doc comment for the normalization convention, and
// train_step's for why sharing that shape is what actually matters (it's what lets train_step be
// templated on Loss rather than needing a LossKind variant).
struct CCELoss
{
    static auto with_extents(const Extents &extents) -> Result<CCELoss, Error>;

    auto forward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
        -> KernelJob<DeviceTensorConstViewf>;

    auto backward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
        -> KernelJob<DeviceTensorConstViewf>;

    DeviceOwningTensorf loss;
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

struct SoftmaxSpec
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

struct ConvTranspose2DSpec
{
    usize kernel_height, kernel_width, channels_out, stride, padding;
    u32 seed;
};

struct MaxPool2DSpec
{
    usize pool_height, pool_width, stride;
};

struct Upsample2DSpec
{
    usize scale;
};

// N-ary: however many inputs Node::inputs holds, elementwise-summed, all the same shape. No
// count stored here - make_layer already receives one Extents per input (pred_extents.size()),
// so a separate field would just be a second, could-get-out-of-sync source of the same number.
struct AddSpec
{
};

// N-ary, joined along the last axis only. No count or shape info stored here either, for the
// same reason as AddSpec: make_layer already gets one Extents per input (pred_extents), so a
// second copy of that information here would just be a could-get-out-of-sync duplicate.
struct ConcatSpec
{
};

using LayerSpec = std::variant<InputSpec, DenseSpec, Conv2DSpec, ConvTranspose2DSpec, SigmoidSpec, SoftmaxSpec,
                               MaxPool2DSpec, Upsample2DSpec, FlattenSpec, AddSpec, ConcatSpec>;

struct Node
{
    LayerSpec spec;
    Extents output_extents;
    std::vector<NodeId> inputs;
};

struct InputLayer
{
    auto forward(const std::vector<DeviceTensorConstViewf> &inputs) const -> KernelJob<DeviceTensorConstViewf>;
    auto backward(const DeviceTensorConstViewf &upstream) const -> std::vector<KernelJob<DeviceTensorConstViewf>>;
};

using LayerKind = std::variant<InputLayer, DenseLayer, SigmoidLayer, SoftmaxLayer, Conv2DLayer, ConvTranspose2DLayer,
                               MaxPool2DLayer, Upsample2DLayer, Flatten2DLayer, AddLayer, ConcatLayer>;

// Whether a layer type carries trainable parameters, asked by parameters(const Layer &) and
// update_layer.
//
// Detection is deliberately loose - "does parameters() exist" - rather than demanding the exact
// return type. Demanding it would report false for a layer whose signature is subtly wrong,
// silently treating it as having no parameters; the static_assert below pins the signature only
// once a layer has claimed to have one, so a mistake is a readable compile error either way.
template <typename T, typename = void>
inline constexpr bool has_parameters_v = false;

template <typename T>
inline constexpr bool has_parameters_v<T, std::void_t<decltype(std::declval<const T &>().parameters())>> = true;

template <typename T>
inline constexpr auto is_trainable_layer() -> bool
{
    if constexpr (has_parameters_v<T>)
    {
        static_assert(std::is_same_v<decltype(std::declval<const T &>().parameters()), std::vector<ConstParameterView>>,
                      "a layer declaring parameters() must return std::vector<ConstParameterView> from its const "
                      "overload - see DenseLayer::parameters");
        return true;
    }
    else
    {
        return false;
    }
}

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
    std::vector<Layer> layers;
    std::vector<std::vector<NodeId>> layer_inputs;
    std::vector<NodeId> execution_order;
    NodeId input_node;
    NodeId output_node;

    // Optional because KernelJob cannot be default-constructed, not because a slot goes
    // unwritten: a successful forward() fills every reachable one. Empty means forward() has not
    // run, which is what forward_output() reports.
    std::vector<std::optional<KernelJob<DeviceTensorConstViewf>>> forward_jobs{};

    // One entry per consumer that has run, in the order they ran, to be summed rather than
    // overwritten (see accumulate_output_gradient). Empty by a node's own turn means nothing
    // reached it, which backward() treats as nothing to propagate.
    std::vector<std::vector<KernelJob<DeviceTensorConstViewf>>> backward_jobs{};

    // Where a fan-out node's gradient contributions are summed. Allocated at compile() time only
    // for nodes with more than one consumer; nullopt elsewhere, since with a single consumer the
    // value already lives in that consumer's own buffer. Read, never allocated, during a step.
    std::vector<std::optional<DeviceOwningTensorf>> d_outputs;

    // For graph-level work that belongs to no layer, like accumulate_output_gradient. Flatten2D
    // and InputLayer launch nothing and so have no stream of their own to borrow.
    Stream stream;

    auto forward(DeviceTensorConstViewf input) -> Result<DeviceTensorConstViewf, Error>;

    // The waited output of node_id's forward pass, or an error if forward() has not run.
    auto forward_output(NodeId node_id) -> Result<DeviceTensorConstViewf, Error>;
    auto backward(DeviceTensorConstViewf loss_grad) -> Result<Void, Error>;
    auto step(AdamOptimizer &optimizer) -> Result<Void, Error>;

    // The gradient of the loss w.r.t. node_id's output, summed if several consumers produced one.
    // A single consumer is just a wait; more than one accumulates into d_outputs[node_id].
    auto accumulate_output_gradient(NodeId node_id) -> Result<DeviceTensorConstViewf, Error>;
};

struct AdamOptimizer
{
    AdamParameters params;
    // One entry per node, indexed by NodeId::value like Model's own arrays, each holding one
    // AdamState per parameter. Empty means that node's layer has no parameters - exact, since a
    // trainable layer always has at least one.
    std::vector<std::vector<AdamState>> states;

    // Owned here rather than passed in, because Adam's bias correction divides by 1 - beta^t and
    // is undefined at t = 0. Model::step() increments before using it, so the first step is t = 1
    // whatever a caller's own loop counts from.
    usize steps_taken = 0;

    static auto from_model(const Model &model, AdamParameters params) -> Result<AdamOptimizer, Error>;
};

// One full training step: forward, loss, backward, optimizer step. Returns the loss value, so a
// caller can log progress without a second loss.forward() of its own.
//
// Templated on Loss rather than taking a variant: anything with MSELoss's
// forward(predictions, targets)/backward(predictions, targets) shape works here unchanged.
template <typename Loss>
auto train_step(Model &model, Loss &loss_fn, DeviceTensorConstViewf inputs, DeviceTensorConstViewf targets,
                AdamOptimizer &optimizer) -> Result<f32, Error>
{
    const auto predictions = VIKA_UNWRAP_OR_RETURN(model.forward(inputs));
    const auto loss_value = VIKA_UNWRAP_OR_RETURN(loss_fn.forward(predictions, targets).wait());
    const auto loss_grad = VIKA_UNWRAP_OR_RETURN(loss_fn.backward(predictions, targets).wait());
    VIKA_UNWRAP_OR_RETURN(model.backward(loss_grad));
    VIKA_UNWRAP_OR_RETURN(model.step(optimizer));

    const auto loss_cpu = VIKA_UNWRAP_OR_RETURN(download(loss_value));
    return ok(loss_cpu[0]);
}

struct ComputationGraph
{
    usize batch_size;
    std::vector<Node> nodes{};
    u32 seed = 0;

    auto input(Extents spatial_extents) -> Result<NodeId, Error>;
    auto dense(NodeId input, usize output_features, std::optional<u32> requested_seed = std::nullopt)
        -> Result<NodeId, Error>;
    auto sigmoid(NodeId input) -> Result<NodeId, Error>;
    auto softmax(NodeId input) -> Result<NodeId, Error>;
    auto flatten(NodeId input) -> Result<NodeId, Error>;
    auto conv2d(NodeId input, usize kernel_height, usize kernel_width, usize channels_out, usize stride, usize padding,
                std::optional<u32> requested_seed = std::nullopt) -> Result<NodeId, Error>;
    auto conv_transpose2d(NodeId input, usize kernel_height, usize kernel_width, usize channels_out, usize stride,
                          usize padding, std::optional<u32> requested_seed = std::nullopt) -> Result<NodeId, Error>;
    auto maxpool2d(NodeId input, usize pool_height, usize pool_width, usize stride) -> Result<NodeId, Error>;
    auto upsample2d(NodeId input, usize scale) -> Result<NodeId, Error>;
    auto add(std::vector<NodeId> inputs) -> Result<NodeId, Error>;
    auto concat(std::vector<NodeId> inputs) -> Result<NodeId, Error>;
    auto compile(NodeId output) -> Result<Model, Error>;

  private:
    // Rolls the graph's own seed forward and returns the new value, so each layer that omits an
    // explicit seed gets a distinct, deterministic one derived from this graph's seed alone.
    auto next_seed() -> u32;
};

// One Extents per predecessor, in Node::inputs order - plural because AddSpec (and future
// multi-input specs) need more than one. Every existing single-input spec just reads
// pred_extents[0]; compile() is the only caller and always passes one entry per input.
auto make_layer(const LayerSpec &spec, usize batch_size, const std::vector<Extents> &pred_extents)
    -> Result<Layer, Error>;

auto parameters(const Layer &layer) -> std::optional<std::vector<ConstParameterView>>;

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

// forward() takes a vector, so nothing at the type level stops a caller passing the wrong count.
// Every forward() checks this before touching inputs[0].
auto _check_input_count(const std::vector<DeviceTensorConstViewf> &inputs, usize expected, const char *context,
                        const char *file, i32 line) -> Result<Void, Error>
{
    if (inputs.size() != expected)
    {
        return error(Error::make(ErrorKind::Shape, file, line, "%s: expects exactly %zu input(s), got %zu", context,
                                 expected, inputs.size()));
    }
    return ok(Void{});
}

#define VIKA_CHECK_INPUT_COUNT(inputs, expected)                                                                       \
    ::vika::_check_input_count((inputs), (expected), __func__, __FILE__, __LINE__)

// A runtime tensor's leading dimension is the batch, which varies per call; every other dimension
// must match what the layer was built for. Call it through VIKA_CHECK_TRAILING_EXTENTS below,
// which fills in the context, name, file and line from the call site.
auto _check_trailing_extents(const Extents &actual, const Extents &expected, const char *context, const char *name,
                             const char *file, i32 line) -> Result<Void, Error>
{
    if (trailing_extents(actual) != trailing_extents(expected))
    {
        return error(Error::make(ErrorKind::Shape, file, line, "%s: %s is %s, layer expects %s (batch aside)", context,
                                 name, describe(actual).c_str(), describe(expected).c_str()));
    }
    return ok(Void{});
}

// The two per-input sites in AddLayer/ConcatLayer::forward call _check_trailing_extents directly
// instead: their name is built at runtime and carries the offending input's index.
#define VIKA_CHECK_TRAILING_EXTENTS(actual, expected)                                                                  \
    ::vika::_check_trailing_extents((actual), (expected), __func__, #actual, __FILE__, __LINE__)

// A zero stride is a window that never advances. window_output_extent divides by it, and the
// transposed form defers the division to a kernel, so neither can express it.
auto _check_stride(usize stride, const char *context, const char *file, i32 line) -> Result<Void, Error>
{
    if (stride == 0)
    {
        return error(Error::make(ErrorKind::Shape, file, line, "%s: stride must be at least 1, got 0", context));
    }
    return ok(Void{});
}

#define VIKA_CHECK_STRIDE(stride) ::vika::_check_stride((stride), __func__, __FILE__, __LINE__)

// The window has to fit the padded input, or window_output_extent's subtraction underflows into
// an enormous extent that surfaces much later as an allocation failure. A window of zero fits any
// input, so the size check below never sees it, and it is just as unusable: it selects nothing.
auto _check_window_fits(usize input, usize window, usize padding, const char *axis, const char *context,
                        const char *file, i32 line) -> Result<Void, Error>
{
    if (window == 0)
    {
        return error(
            Error::make(ErrorKind::Shape, file, line, "%s: window %s must be at least 1, got 0", context, axis));
    }
    if (input + 2 * padding < window)
    {
        return error(Error::make(ErrorKind::Shape, file, line,
                                 "%s: window %s %zu does not fit an input of %zu padded by %zu", context, axis, window,
                                 input, padding));
    }
    return ok(Void{});
}

#define VIKA_CHECK_WINDOW_FITS(input, window, padding, axis)                                                           \
    ::vika::_check_window_fits((input), (window), (padding), (axis), __func__, __FILE__, __LINE__)

// The same for the inverse formula, which subtracts 2 * padding from (input - 1) * stride +
// window. Its input must also be non-empty, since (input - 1) underflows at zero.
auto _check_transposed_window_fits(usize input, usize window, usize stride, usize padding, const char *axis,
                                   const char *context, const char *file, i32 line) -> Result<Void, Error>
{
    if (window == 0)
    {
        return error(
            Error::make(ErrorKind::Shape, file, line, "%s: window %s must be at least 1, got 0", context, axis));
    }
    if (input == 0)
    {
        return error(
            Error::make(ErrorKind::Shape, file, line, "%s: input %s must be at least 1, got 0", context, axis));
    }
    if ((input - 1) * stride + window < 2 * padding)
    {
        return error(Error::make(ErrorKind::Shape, file, line,
                                 "%s: padding %zu exceeds (input %s %zu - 1) * stride %zu + window %zu", context,
                                 padding, axis, input, stride, window));
    }
    return ok(Void{});
}

#define VIKA_CHECK_TRANSPOSED_WINDOW_FITS(input, window, stride, padding, axis)                                        \
    ::vika::_check_transposed_window_fits((input), (window), (stride), (padding), (axis), __func__, __FILE__, __LINE__)

// Construction-time counterpart of VIKA_CHECK_INPUT_COUNT, for make_layer's reads of
// pred_extents[0]. A minimum rather than an exact count, because Add and Concat take any number
// from two up; forward() still checks the exact arity per call.
//
// No context parameter, unlike the checkers above: every call sits inside make_layer's std::visit
// lambda, where __func__ is the useless "operator()".
auto _check_pred_count(const std::vector<Extents> &pred_extents, usize at_least, const char *file, i32 line)
    -> Result<Void, Error>
{
    if (pred_extents.size() < at_least)
    {
        return error(Error::make(ErrorKind::Graph, file, line,
                                 "make_layer: this layer needs at least %zu "
                                 "predecessor(s), got %zu",
                                 at_least, pred_extents.size()));
    }
    return ok(Void{});
}

#define VIKA_CHECK_PRED_COUNT(pred_extents, at_least)                                                                  \
    ::vika::_check_pred_count((pred_extents), (at_least), __FILE__, __LINE__)

// Two runtime tensors that must line up row for row. _check_trailing_extents ignores the batch
// dimension by design, so nothing else catches two independently-supplied tensors disagreeing on
// it - and every kernel here sizes its launch off one while indexing straight into the other.
auto _check_batch_agreement(const DeviceTensorConstViewf &a, const DeviceTensorConstViewf &b, const char *context,
                            const char *a_name, const char *b_name, const char *file, i32 line) -> Result<Void, Error>
{
    if (a.extents[0] != b.extents[0])
    {
        return error(Error::make(ErrorKind::Shape, file, line, "%s: %s has batch %zu but %s has batch %zu", context,
                                 a_name, a.extents[0], b_name, b.extents[0]));
    }
    return ok(Void{});
}

// Context and operand names come from the call site rather than being spelled out by every caller.
// The file and line are forwarded and Error built with them, because VIKA_SHAPE_ERROR would
// capture this helper's line and report it for all fifty call sites - and __func__ is the bare
// function name ("forward"), so the line is what tells MSELoss apart from CCELoss.
#define VIKA_CHECK_BATCH_AGREEMENT(a, b) ::vika::_check_batch_agreement((a), (b), __func__, #a, #b, __FILE__, __LINE__)

auto checked_size(const Extents &extents) -> Result<usize, Error>
{
    if (extents.empty())
    {
        return error(VIKA_SHAPE_ERROR("tensor extents are empty"));
    }

    const usize count = VIKA_UNWRAP_OR_RETURN(checked_element_count(extents));
    if (count == 0)
    {
        return error(VIKA_SHAPE_ERROR("tensor extents contain a zero extent"));
    }
    return ok(count);
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

auto transposed(const DeviceTensorConstViewf &view) -> Result<DeviceTensorConstViewf, Error>
{
    if (view.rank() != 2)
    {
        return error(VIKA_SHAPE_ERROR("transposed: expects a rank 2 view, got rank %zu", view.rank()));
    }

    auto transposed_view = view;
    std::swap(transposed_view.strides[0], transposed_view.strides[1]);
    std::swap(transposed_view.extents[0], transposed_view.extents[1]);
    return ok(transposed_view);
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
    const dim3 block(256);
    return launch_kernel(uniform_tensor_kernel, Void{}, grid_covering(block, n), block, stream, tensor, seed);
}

auto xavier_tensor(DeviceTensorViewf tensor, u32 seed, usize fan_in, usize fan_out, cudaStream_t stream)
    -> KernelJob<Void>
{
    const f32 limit = std::sqrt(6.0f / (f32)(fan_in + fan_out));
    const usize n = tensor.element_count();
    const dim3 block(256);
    return launch_kernel(xavier_tensor_kernel, Void{}, grid_covering(block, n), block, stream, tensor, seed, limit);
}

// ===========================================================================
// Parameters and Optimizer
// ===========================================================================

// Adam bias-correction scales. These depend only on the step count, so computing them
// once on the host beats recomputing pow() in every thread.
inline auto adam_bias_correction(const AdamParameters &params, usize t) -> std::pair<f32, f32>
{
    const auto correct = [t](f32 beta) -> f32 { return (f32)(1.0 / (1.0 - std::pow((double)beta, (double)t))); };
    return {correct(params.beta1), correct(params.beta2)};
}

auto update_parameters(std::vector<ParameterView> &parameters, std::vector<AdamState> &states, cudaStream_t stream,
                       const AdamParameters &params, usize t) -> std::vector<KernelJob<Void>>
{
    if (t == 0)
    {
        // Model::step() makes this unreachable by owning the counter, but a layer's update() is
        // public and takes t straight from its caller - see AdamOptimizer::steps_taken for what
        // t = 0 does to the bias correction.
        return {error(VIKA_UNSUPPORTED_ERROR("adam: step t must be at least 1, got 0"))};
    }

    if (states.size() != parameters.size())
    {
        // states[i] is indexed in lockstep with parameters[i] below, with nothing structural
        // tying the two together - states comes from AdamOptimizer::from_model, built once at
        // construction from the const parameters() overload, while parameters here comes from
        // the non-const overload called fresh every step. If a layer's two overloads ever drift
        // out of sync (a parameter added to one but not the other), this is the one place both
        // vectors meet, so it's the one place that can catch it before indexing off the end.
        return {error(VIKA_UNSUPPORTED_ERROR("update_parameters: %zu states but %zu parameters", states.size(),
                                             parameters.size()))};
    }

    const auto [m_hat_scale, v_hat_scale] = adam_bias_correction(params, t);
    const dim3 block(256);

    std::vector<KernelJob<Void>> jobs;
    jobs.reserve(parameters.size());
    for (usize i = 0; i < parameters.size(); ++i)
    {
        const auto count = parameters[i].value.element_count();
        jobs.push_back(launch_kernel(adam_update, Void{}, grid_covering(block, count), block, stream, params,
                                     m_hat_scale, v_hat_scale, parameters[i].grad, parameters[i].value,
                                     states[i].m.view(), states[i].v.view()));
    }
    return jobs;
}

// =============================================================================
// Layers
// =============================================================================

auto DenseLayer::with_weights(usize batch_size, DeviceOwningTensorf weights, DeviceOwningTensorf biases)
    -> Result<DenseLayer, Error>
{
    if (weights.extents().size() != 2)
    {
        return error(VIKA_SHAPE_ERROR("dense: weights must be rank 2 [features, neurons], got rank %zu",
                                      weights.extents().size()));
    }
    if (biases.extents().size() != 1)
    {
        return error(VIKA_SHAPE_ERROR("dense: biases must be rank 1 [neurons], got rank %zu", biases.extents().size()));
    }

    const auto feature_count = weights.extent(0);
    const auto neuron_count = weights.extent(1);

    if (biases.extent(0) != neuron_count)
    {
        return error(
            VIKA_SHAPE_ERROR("dense: biases has %zu neurons, weights expects %zu", biases.extent(0), neuron_count));
    }

    auto outputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({batch_size, neuron_count}));

    auto d_inputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({batch_size, feature_count}));
    auto d_weights = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty_like(weights));
    auto d_biases = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty_like(biases));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok<DenseLayer>({
        .outputs = std::move(outputs),
        .d_inputs = std::move(d_inputs),
        .weights = {std::move(weights), std::move(d_weights)},
        .biases = {std::move(biases), std::move(d_biases)},
        .stream = std::move(stream),
    });
}

auto DenseLayer::randomized(usize batch_size, usize input_features, usize neuron_count, u32 seed)
    -> Result<DenseLayer, Error>
{
    auto weights = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({input_features, neuron_count}));
    auto biases = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::from(std::vector<f32>(neuron_count, 0.0f)));

    // Built first so xavier_tensor can run on the layer's own stream instead of standing up a
    // second, throwaway one just for initialization.
    auto layer = VIKA_UNWRAP_OR_RETURN(with_weights(batch_size, std::move(weights), std::move(biases)));
    VIKA_UNWRAP_OR_RETURN(
        xavier_tensor(layer.weights.value.view(), seed, input_features, neuron_count, layer.stream.handle()).wait());
    return ok(std::move(layer));
}

auto DenseLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, 1));
    const auto &input = inputs[0];

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(input.extents, d_inputs.extents()));

    const usize k = input.extents[0];
    auto sliced_outputs = VIKA_UNWRAP_OR_RETURN(outputs.view().first_n(k));
    auto sliced_outputs_const = sliced_outputs.const_view();

    const usize M = k;
    const usize N = outputs.extent(1);
    const dim3 block(16, 16);
    const dim3 grid = grid_covering(block, N, M);
    auto matmul_job = launch_kernel(matmul_kernel, sliced_outputs_const, grid, block, stream.handle(), input,
                                    weights.value.const_view(), sliced_outputs);
    if (matmul_job.is_error())
    {
        return matmul_job;
    }
    return launch_kernel(add_bias, sliced_outputs_const, grid, block, stream.handle(), biases.value.const_view(),
                         sliced_outputs);
}

auto DenseLayer::backward(const DeviceTensorConstViewf &upstream)
    -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents()));

    const usize k = upstream.extents[0];
    auto sliced_d_inputs = VIKA_UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const usize M = k;
    const usize N = d_inputs.extent(1);
    const auto transposed_weights = VIKA_UNWRAP_OR_RETURN(transposed(weights.value.const_view()));

    const dim3 block(16, 16);
    const dim3 grid = grid_covering(block, N, M);
    return {launch_kernel(matmul_kernel, sliced_d_inputs.const_view(), grid, block, stream.handle(), upstream,
                          transposed_weights, sliced_d_inputs)};
}

auto DenseLayer::weight_gradients(const DeviceTensorConstViewf &inputs, const DeviceTensorConstViewf &upstream)
    -> KernelJob<std::tuple<DeviceTensorConstViewf, DeviceTensorConstViewf>>
{

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(inputs.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_BATCH_AGREEMENT(inputs, upstream));

    // Grid must cover weights.grad's own shape (feature_count x neuron_count) - the tensor the
    // matmul below actually writes - not d_inputs' (batch_capacity x feature_count). Sizing from
    // the wrong tensor silently under-covers weights.grad whenever feature_count or neuron_count
    // exceeds batch_capacity, leaving stale data in the uncovered elements.
    const usize M = weights.grad.extent(0);
    const usize N = weights.grad.extent(1);
    const dim3 block(16, 16);
    const dim3 grid = grid_covering(block, N, M);
    const auto gradients = std::make_tuple(weights.grad.const_view(), biases.grad.const_view());
    const auto transposed_inputs = VIKA_UNWRAP_OR_RETURN(transposed(inputs));
    const auto transposed_upstream = VIKA_UNWRAP_OR_RETURN(transposed(upstream));

    // NOTE: Run in separate streams?
    auto matmul_job = launch_kernel(matmul_kernel, gradients, grid, block, stream.handle(), transposed_inputs,
                                    upstream, weights.grad.view());
    if (matmul_job.is_error())
    {
        return matmul_job;
    }

    const dim3 row_block(256);
    const auto row_grid = grid_covering(row_block, upstream.extents[1]);
    return launch_kernel(sum_rows, gradients, row_grid, row_block, stream.handle(), transposed_upstream,
                         biases.grad.view());
}

auto DenseLayer::parameters() -> std::vector<ParameterView>
{
    return {weights.view(), biases.view()};
}

// Must stay in lockstep with the non-const overload above: same order, same count.
auto DenseLayer::parameters() const -> std::vector<ConstParameterView>
{
    return {weights.const_view(), biases.const_view()};
}

auto DenseLayer::update(std::vector<AdamState> &states, const AdamParameters &params, usize t)
    -> std::vector<KernelJob<Void>>
{
    auto params_list = parameters();
    return update_parameters(params_list, states, stream.handle(), params, t);
}

auto SigmoidLayer::with_extents(const Extents &extents) -> Result<SigmoidLayer, Error>
{
    auto outputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(extents));
    auto d_inputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(extents));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(
        SigmoidLayer{.outputs = std::move(outputs), .d_inputs = std::move(d_inputs), .stream = std::move(stream)});
}

auto SigmoidLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, 1));
    const auto &input = inputs[0];

    const auto input_extents = input.extents;
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(input_extents, outputs.extents()));

    const usize k = input_extents[0];
    auto sliced_outputs = VIKA_UNWRAP_OR_RETURN(outputs.view().first_n(k));

    const dim3 block(256);
    const auto grid = grid_covering(block, input.element_count());
    return launch_kernel(sigmoid_forward, sliced_outputs.const_view(), grid, block, stream.handle(), input,
                         sliced_outputs);
}

auto SigmoidLayer::backward(const DeviceTensorConstViewf &upstream)
    -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    const auto upstream_extents = upstream.extents;
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream_extents, d_inputs.extents()));

    // Invariant: with_extents allocates both from the same extents.
    if (d_inputs.extents() != outputs.extents())
    {
        return {KernelJob<DeviceTensorConstViewf>::failed(
            VIKA_SHAPE_ERROR("sigmoid backward: gradient holds %zu elements but outputs hold %zu",
                             d_inputs.element_count(), outputs.element_count()))};
    }

    const usize k = upstream_extents[0];
    auto sliced_d_inputs = VIKA_UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const dim3 block(256);
    const auto grid = grid_covering(block, upstream.element_count());
    return {launch_kernel(sigmoid_backward, sliced_d_inputs.const_view(), grid, block, stream.handle(),
                          VIKA_UNWRAP_OR_RETURN(outputs.const_view().first_n(k)), upstream,
                          sliced_d_inputs)};
}

auto SoftmaxLayer::with_extents(const Extents &extents) -> Result<SoftmaxLayer, Error>
{
    auto outputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(extents));
    auto d_inputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(extents));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(
        SoftmaxLayer{.outputs = std::move(outputs), .d_inputs = std::move(d_inputs), .stream = std::move(stream)});
}

auto SoftmaxLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, 1));
    const auto &input = inputs[0];

    const auto input_extents = input.extents;
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(input_extents, outputs.extents()));

    const usize k = input_extents[0];
    auto sliced_outputs = VIKA_UNWRAP_OR_RETURN(outputs.view().first_n(k));

    // One thread per row, not per element - see softmax_forward's own doc comment for why.
    const usize width = input.extents.back();
    const usize row_count = input.element_count() / width;
    const dim3 block(256);
    const auto grid = grid_covering(block, row_count);
    return launch_kernel(softmax_forward, sliced_outputs.const_view(), grid, block, stream.handle(), input,
                         sliced_outputs);
}

auto SoftmaxLayer::backward(const DeviceTensorConstViewf &upstream)
    -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    const auto upstream_extents = upstream.extents;
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream_extents, d_inputs.extents()));

    // Invariant: with_extents allocates both from the same extents.
    if (d_inputs.extents() != outputs.extents())
    {
        return {KernelJob<DeviceTensorConstViewf>::failed(
            VIKA_SHAPE_ERROR("softmax backward: gradient holds %zu elements but outputs hold %zu",
                             d_inputs.element_count(), outputs.element_count()))};
    }

    const usize k = upstream_extents[0];
    auto sliced_d_inputs = VIKA_UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const usize width = upstream.extents.back();
    const usize row_count = upstream.element_count() / width;
    const dim3 block(256);
    const auto grid = grid_covering(block, row_count);
    return {launch_kernel(softmax_backward, sliced_d_inputs.const_view(), grid, block, stream.handle(),
                          VIKA_UNWRAP_OR_RETURN(outputs.const_view().first_n(k)), upstream,
                          sliced_d_inputs)};
}

auto Flatten2DLayer::with_extents(const Extents &extents) -> Result<Flatten2DLayer, Error>
{
    if (extents.size() < 2)
    {
        return error(VIKA_SHAPE_ERROR("flatten: extents must be rank 2 or higher, got %s", describe(extents).c_str()));
    }
    return ok(Flatten2DLayer{extents});
}

auto Flatten2DLayer::output_extents() const -> Extents
{
    return {extents[0], element_count(trailing_extents(extents))};
}

auto Flatten2DLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) const
    -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, 1));
    const auto &input = inputs[0];

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(input.extents, extents));

    const usize k = input.extents[0];
    const usize features = element_count(trailing_extents(extents));
    return KernelJob<DeviceTensorConstViewf>::ready(DeviceTensorConstViewf(input.data, {k, features}));
}

auto InputLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) const -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, 1));
    return KernelJob<DeviceTensorConstViewf>::ready(inputs[0]);
}

auto InputLayer::backward(const DeviceTensorConstViewf &upstream) const
    -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    return {KernelJob<DeviceTensorConstViewf>::ready(upstream)};
}

auto Flatten2DLayer::backward(const DeviceTensorConstViewf &upstream) const
    -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    // Against output_extents(), not extents: upstream is the gradient of the flattened output,
    // so it is rank 2 [batch, features].
    const auto upstream_extents = upstream.extents;
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream_extents, output_extents()));

    Extents input_extents = extents;
    input_extents[0] = upstream_extents[0];
    return {KernelJob<DeviceTensorConstViewf>::ready(DeviceTensorConstViewf(upstream.data, input_extents))};
}

auto Conv2DLayer::with_weights(usize batch_size, usize input_height, usize input_width, DeviceOwningTensorf filters,
                               DeviceOwningTensorf biases, usize stride, usize padding) -> Result<Conv2DLayer, Error>
{
    if (filters.extents().size() != 4)
    {
        return error(VIKA_SHAPE_ERROR("conv2d: filters must be rank 4 [kH, kW, C_in, C_out], got rank %zu",
                                      filters.extents().size()));
    }
    if (biases.extents().size() != 1)
    {
        return error(VIKA_SHAPE_ERROR("conv2d: biases must be rank 1 [C_out], got rank %zu", biases.extents().size()));
    }

    const usize kH = filters.extent(0);
    const usize kW = filters.extent(1);
    const usize C_out = filters.extent(3);

    const usize C_in = filters.extent(2);

    if (biases.extent(0) != C_out)
    {
        return error(VIKA_SHAPE_ERROR("conv2d: biases has %zu channels, filters expects %zu", biases.extent(0), C_out));
    }

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_STRIDE(stride));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_WINDOW_FITS(input_height, kH, padding, "height"));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_WINDOW_FITS(input_width, kW, padding, "width"));

    const usize out_H = window_output_extent(input_height, kH, stride, padding);
    const usize out_W = window_output_extent(input_width, kW, stride, padding);

    auto outputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({batch_size, out_H, out_W, C_out}));
    auto d_inputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({batch_size, input_height, input_width, C_in}));
    auto d_filters = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty_like(filters));
    auto d_biases = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty_like(biases));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(Conv2DLayer{
        .outputs = std::move(outputs),
        .d_inputs = std::move(d_inputs),
        .filters = {std::move(filters), std::move(d_filters)},
        .biases = {std::move(biases), std::move(d_biases)},
        .stride = stride,
        .padding = padding,
        .stream = std::move(stream),
    });
}

auto Conv2DLayer::randomized(usize batch_size, usize input_height, usize input_width, usize kH, usize kW, usize C_in,
                             usize C_out, usize stride, usize padding, u32 seed) -> Result<Conv2DLayer, Error>
{
    auto filters = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({kH, kW, C_in, C_out}));
    auto biases = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::from(std::vector<f32>(C_out, 0.0f)));

    // Built first so xavier_tensor can run on the layer's own stream instead of standing up a
    // second, throwaway one just for initialization.
    auto layer = VIKA_UNWRAP_OR_RETURN(
        with_weights(batch_size, input_height, input_width, std::move(filters), std::move(biases), stride, padding));
    VIKA_UNWRAP_OR_RETURN(
        xavier_tensor(layer.filters.value.view(), seed, kH * kW * C_in, kH * kW * C_out, layer.stream.handle()).wait());
    return ok(std::move(layer));
}

auto Conv2DLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, 1));
    const auto &input = inputs[0];

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(input.extents, d_inputs.extents()));

    const usize k = input.extents[0];
    auto sliced_outputs = VIKA_UNWRAP_OR_RETURN(outputs.view().first_n(k));

    const dim3 block(16, 16, 1);
    const dim3 grid = nhwc_grid(block, outputs.extents(), k);

    return launch_kernel(conv_forward, sliced_outputs.const_view(), grid, block, stream.handle(), input,
                         filters.value.const_view(), biases.value.const_view(), sliced_outputs, stride, padding);
}

auto Conv2DLayer::backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents()));

    const usize k = upstream.extents[0];
    auto sliced_d_inputs = VIKA_UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const dim3 block(16, 16, 1);
    const dim3 grid = nhwc_grid(block, d_inputs.extents(), k);

    return {launch_kernel(conv_backward, sliced_d_inputs.const_view(), grid, block, stream.handle(), upstream,
                          filters.value.const_view(), sliced_d_inputs, stride, padding)};
}

auto Conv2DLayer::weight_gradients(const DeviceTensorConstViewf &inputs, const DeviceTensorConstViewf &upstream)
    -> KernelJob<std::tuple<DeviceTensorConstViewf, DeviceTensorConstViewf>>
{

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(inputs.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_BATCH_AGREEMENT(inputs, upstream));

    const usize filter_count = filters.grad.element_count();
    const usize C_out = biases.grad.element_count();
    const dim3 block(256);

    const auto gradients = std::make_tuple(filters.grad.const_view(), biases.grad.const_view());

    auto weight_job = launch_kernel(conv_weight_gradients, gradients, grid_covering(block, filter_count), block,
                                    stream.handle(), inputs, upstream, filters.grad.view(), stride, padding);
    if (weight_job.is_error())
    {
        return weight_job;
    }
    return launch_kernel(conv_bias_gradients, gradients, grid_covering(block, C_out), block, stream.handle(), upstream,
                         biases.grad.view());
}

auto Conv2DLayer::parameters() -> std::vector<ParameterView>
{
    return {filters.view(), biases.view()};
}

// Must stay in lockstep with the non-const overload above: same order, same count.
auto Conv2DLayer::parameters() const -> std::vector<ConstParameterView>
{
    return {filters.const_view(), biases.const_view()};
}

auto Conv2DLayer::update(std::vector<AdamState> &states, const AdamParameters &params, usize t)
    -> std::vector<KernelJob<Void>>
{
    auto params_list = parameters();
    return update_parameters(params_list, states, stream.handle(), params, t);
}

auto ConvTranspose2DLayer::with_weights(usize batch_size, usize input_height, usize input_width,
                                        DeviceOwningTensorf filters, DeviceOwningTensorf biases, usize stride,
                                        usize padding) -> Result<ConvTranspose2DLayer, Error>
{
    if (filters.extents().size() != 4)
    {
        return error(VIKA_SHAPE_ERROR("conv_transpose2d: filters must be rank 4 [kH, kW, C_out, C_in], got rank %zu",
                                      filters.extents().size()));
    }
    if (biases.extents().size() != 1)
    {
        return error(
            VIKA_SHAPE_ERROR("conv_transpose2d: biases must be rank 1 [C_out], got rank %zu", biases.extents().size()));
    }

    const usize kH = filters.extent(0);
    const usize kW = filters.extent(1);
    const usize C_out = filters.extent(2);
    const usize C_in = filters.extent(3);

    if (biases.extent(0) != C_out)
    {
        return error(VIKA_SHAPE_ERROR("conv_transpose2d: biases has %zu channels, filters expects %zu",
                                      biases.extent(0), C_out));
    }

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_STRIDE(stride));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRANSPOSED_WINDOW_FITS(input_height, kH, stride, padding, "height"));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRANSPOSED_WINDOW_FITS(input_width, kW, stride, padding, "width"));

    const usize out_H = transposed_window_output_extent(input_height, kH, stride, padding);
    const usize out_W = transposed_window_output_extent(input_width, kW, stride, padding);

    auto outputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({batch_size, out_H, out_W, C_out}));
    auto d_inputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({batch_size, input_height, input_width, C_in}));
    auto d_filters = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty_like(filters));
    auto d_biases = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty_like(biases));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(ConvTranspose2DLayer{
        .outputs = std::move(outputs),
        .d_inputs = std::move(d_inputs),
        .filters = {std::move(filters), std::move(d_filters)},
        .biases = {std::move(biases), std::move(d_biases)},
        .stride = stride,
        .padding = padding,
        .stream = std::move(stream),
    });
}

auto ConvTranspose2DLayer::randomized(usize batch_size, usize input_height, usize input_width, usize kH, usize kW,
                                      usize C_in, usize C_out, usize stride, usize padding, u32 seed)
    -> Result<ConvTranspose2DLayer, Error>
{
    auto filters = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({kH, kW, C_out, C_in}));
    auto biases = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::from(std::vector<f32>(C_out, 0.0f)));

    // Built first so xavier_tensor can run on the layer's own stream instead of standing up a
    // second, throwaway one just for initialization.
    auto layer = VIKA_UNWRAP_OR_RETURN(
        with_weights(batch_size, input_height, input_width, std::move(filters), std::move(biases), stride, padding));
    // fan_in/fan_out are about the operation's actual connectivity (kH*kW inputs feeding each
    // output channel, kH*kW outputs fed by each input channel), not the filters tensor's literal
    // axis order - same values Conv2DLayer::randomized passes despite the swapped C_out/C_in
    // layout.
    VIKA_UNWRAP_OR_RETURN(
        xavier_tensor(layer.filters.value.view(), seed, kH * kW * C_in, kH * kW * C_out, layer.stream.handle()).wait());
    return ok(std::move(layer));
}

auto ConvTranspose2DLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs)
    -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, 1));
    const auto &input = inputs[0];

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(input.extents, d_inputs.extents()));

    const usize k = input.extents[0];
    auto sliced_outputs = VIKA_UNWRAP_OR_RETURN(outputs.view().first_n(k));

    const dim3 block(16, 16, 1);
    const dim3 grid = nhwc_grid(block, outputs.extents(), k);

    return launch_kernel(conv_transpose_forward, sliced_outputs.const_view(), grid, block, stream.handle(), input,
                         filters.value.const_view(), biases.value.const_view(), sliced_outputs, stride, padding);
}

auto ConvTranspose2DLayer::backward(const DeviceTensorConstViewf &upstream)
    -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents()));

    const usize k = upstream.extents[0];
    auto sliced_d_inputs = VIKA_UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const dim3 block(16, 16, 1);
    const dim3 grid = nhwc_grid(block, d_inputs.extents(), k);

    return {launch_kernel(conv_transpose_backward, sliced_d_inputs.const_view(), grid, block, stream.handle(), upstream,
                          filters.value.const_view(), sliced_d_inputs, stride, padding)};
}

auto ConvTranspose2DLayer::weight_gradients(const DeviceTensorConstViewf &inputs,
                                            const DeviceTensorConstViewf &upstream)
    -> KernelJob<std::tuple<DeviceTensorConstViewf, DeviceTensorConstViewf>>
{

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(inputs.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_BATCH_AGREEMENT(inputs, upstream));

    const usize filter_count = filters.grad.element_count();
    const usize C_out = biases.grad.element_count();
    const dim3 block(256);

    const auto gradients = std::make_tuple(filters.grad.const_view(), biases.grad.const_view());

    auto weight_job = launch_kernel(conv_transpose_weight_gradients, gradients, grid_covering(block, filter_count),
                                    block, stream.handle(), inputs, upstream, filters.grad.view(), stride, padding);
    if (weight_job.is_error())
    {
        return weight_job;
    }
    // Identical in shape to Conv2D's own bias gradient - see conv_transpose_forward's doc comment
    // for why conv_bias_gradients is reused here unchanged rather than getting its own variant.
    return launch_kernel(conv_bias_gradients, gradients, grid_covering(block, C_out), block, stream.handle(), upstream,
                         biases.grad.view());
}

auto ConvTranspose2DLayer::parameters() -> std::vector<ParameterView>
{
    return {filters.view(), biases.view()};
}

// Must stay in lockstep with the non-const overload above: same order, same count.
auto ConvTranspose2DLayer::parameters() const -> std::vector<ConstParameterView>
{
    return {filters.const_view(), biases.const_view()};
}

auto ConvTranspose2DLayer::update(std::vector<AdamState> &states, const AdamParameters &params, usize t)
    -> std::vector<KernelJob<Void>>
{
    auto params_list = parameters();
    return update_parameters(params_list, states, stream.handle(), params, t);
}

auto MaxPool2DLayer::with_extents(usize batch_size, usize input_height, usize input_width, usize channels, usize pool_h,
                                  usize pool_w, usize stride) -> Result<MaxPool2DLayer, Error>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_STRIDE(stride));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_WINDOW_FITS(input_height, pool_h, 0, "height"));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_WINDOW_FITS(input_width, pool_w, 0, "width"));

    const usize out_H = window_output_extent(input_height, pool_h, stride, 0);
    const usize out_W = window_output_extent(input_width, pool_w, stride, 0);

    auto outputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({batch_size, out_H, out_W, channels}));
    auto argmax = VIKA_UNWRAP_OR_RETURN((DeviceOwningTensoru::empty({batch_size, out_H, out_W, channels})));
    auto d_inputs =
        VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({batch_size, input_height, input_width, channels}));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
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

auto MaxPool2DLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, 1));
    const auto &input = inputs[0];

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(input.extents, d_inputs.extents()));

    const usize k = input.extents[0];
    auto sliced_outputs = VIKA_UNWRAP_OR_RETURN(outputs.view().first_n(k));
    auto sliced_argmax = VIKA_UNWRAP_OR_RETURN(argmax.view().first_n(k));

    const dim3 block(16, 16, 1);
    const dim3 grid = nhwc_grid(block, outputs.extents(), k);

    return launch_kernel(maxpool_forward, sliced_outputs.const_view(), grid, block, stream.handle(), input,
                         sliced_outputs, sliced_argmax, pool_h, pool_w, stride);
}

auto MaxPool2DLayer::backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents()));

    const usize k = upstream.extents[0];
    auto sliced_d_inputs = VIKA_UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    // The slice, not the whole buffer: pooling scatters, so d_inputs has to start clean, but only
    // the k rows this call writes and hands back are part of the result.
    VIKA_UNWRAP_OR_RETURN(zero(sliced_d_inputs, stream.handle()));

    // One thread per *output* position, unlike every other backward here - maxpool_backward
    // scatters from the pooled grid back into d_inputs rather than gathering into it.
    const dim3 block(16, 16, 1);
    const dim3 grid = nhwc_grid(block, outputs.extents(), k);

    return {launch_kernel(maxpool_backward, sliced_d_inputs.const_view(), grid, block, stream.handle(), upstream,
                          VIKA_UNWRAP_OR_RETURN(argmax.const_view().first_n(k)), sliced_d_inputs)};
}

auto Upsample2DLayer::with_extents(usize batch_size, usize input_height, usize input_width, usize channels, usize scale)
    -> Result<Upsample2DLayer, Error>
{
    if (scale == 0)
    {
        return error(VIKA_SHAPE_ERROR("upsample2d: scale must be at least 1, got %zu", scale));
    }

    auto outputs = VIKA_UNWRAP_OR_RETURN(
        DeviceOwningTensorf::empty({batch_size, input_height * scale, input_width * scale, channels}));
    auto d_inputs =
        VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({batch_size, input_height, input_width, channels}));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(Upsample2DLayer{
        .outputs = std::move(outputs),
        .d_inputs = std::move(d_inputs),
        .scale = scale,
        .stream = std::move(stream),
    });
}

auto Upsample2DLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, 1));
    const auto &input = inputs[0];

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(input.extents, d_inputs.extents()));

    const usize k = input.extents[0];
    auto sliced_outputs = VIKA_UNWRAP_OR_RETURN(outputs.view().first_n(k));

    const dim3 block(16, 16, 1);
    const dim3 grid = nhwc_grid(block, outputs.extents(), k);

    return launch_kernel(upsample2d_forward, sliced_outputs.const_view(), grid, block, stream.handle(), input,
                         sliced_outputs, scale);
}

auto Upsample2DLayer::backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents()));

    const usize k = upstream.extents[0];
    auto sliced_d_inputs = VIKA_UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const dim3 block(16, 16, 1);
    const dim3 grid = nhwc_grid(block, d_inputs.extents(), k);

    // No zero() first, unlike MaxPool2DLayer::backward - see upsample2d_backward's own doc comment
    // for why: every d_inputs slot is written exactly once, so there is nothing left over to clear.
    return {launch_kernel(upsample2d_backward, sliced_d_inputs.const_view(), grid, block, stream.handle(), upstream,
                          sliced_d_inputs, scale)};
}

auto AddLayer::with_extents(const Extents &extents, usize input_count) -> Result<AddLayer, Error>
{
    if (input_count < 2)
    {
        return error(VIKA_SHAPE_ERROR("add: input_count must be at least 2, got %zu", input_count));
    }

    auto outputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(extents));
    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(AddLayer{.outputs = std::move(outputs), .input_count = input_count, .stream = std::move(stream)});
}

auto AddLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, input_count));

    for (usize i = 0; i < inputs.size(); ++i)
    {
        // Called directly, not through VIKA_CHECK_TRAILING_EXTENTS: the name carries the offending
        // input's index, which the macro's stringified argument ("inputs[i]") cannot.
        VIKA_UNWRAP_OR_RETURN(_check_trailing_extents(inputs[i].extents, outputs.extents(), __func__,
                                                      ("input " + std::to_string(i)).c_str(), __FILE__, __LINE__));
        if (inputs[i].extents[0] != inputs[0].extents[0])
        {
            return error(VIKA_SHAPE_ERROR("add forward: input %zu has batch %zu but input 0 has batch %zu", i,
                                          inputs[i].extents[0], inputs[0].extents[0]));
        }
    }

    const usize k = inputs[0].extents[0];
    auto sliced_outputs = VIKA_UNWRAP_OR_RETURN(outputs.view().first_n(k));

    // out = inputs[0], then accumulate every remaining input into it - see accumulate_into's
    // doc comment for why this kernel, written for gradient fan-in summing, also does
    // elementwise add.
    VIKA_UNWRAP_OR_RETURN(copy(inputs[0], sliced_outputs, stream.handle()));

    const dim3 block(256);
    const auto grid = grid_covering(block, inputs[0].element_count());
    for (usize i = 1; i < inputs.size() - 1; ++i)
    {
        // Checked, not waited: these launches share a stream and are already ordered. is_error()
        // only inspects what launch_kernel captured from cudaGetLastError, which is the only
        // chance to see a bad launch config - a runtime failure still surfaces on whichever job
        // the caller eventually waits on.
        auto job = launch_kernel(accumulate_into, sliced_outputs.const_view(), grid, block, stream.handle(), inputs[i],
                                 sliced_outputs);
        if (job.is_error())
        {
            return job;
        }
    }
    return launch_kernel(accumulate_into, sliced_outputs.const_view(), grid, block, stream.handle(), inputs.back(),
                         sliced_outputs);
}

auto AddLayer::backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    const auto check = VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents());
    if (check.is_error())
    {
        return std::vector<KernelJob<DeviceTensorConstViewf>>(
            input_count, KernelJob<DeviceTensorConstViewf>::failed(check.unwrap_error()));
    }

    // d/dx_i(sum) = 1 for every input: the same upstream gradient flows unchanged to all of them,
    // so this is a pure pass-through with no workspace or kernel launch needed, same idea as
    // InputLayer/Flatten2DLayer's "ready" jobs.
    return std::vector<KernelJob<DeviceTensorConstViewf>>(input_count,
                                                          KernelJob<DeviceTensorConstViewf>::ready(upstream));
}

auto ConcatLayer::with_extents(const std::vector<Extents> &input_extents) -> Result<ConcatLayer, Error>
{
    const auto output_extents = VIKA_UNWRAP_OR_RETURN(concat_output_extents(input_extents));
    auto outputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(output_extents));

    auto d_inputs = VIKA_UNWRAP_OR_RETURN(try_map(input_extents, &DeviceOwningTensorf::empty));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(ConcatLayer{.outputs = std::move(outputs), .d_inputs = std::move(d_inputs), .stream = std::move(stream)});
}

auto ConcatLayer::forward(const std::vector<DeviceTensorConstViewf> &inputs) -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_INPUT_COUNT(inputs, d_inputs.size()));

    for (usize i = 0; i < inputs.size(); ++i)
    {
        // Called directly for the same reason as AddLayer::forward above - the index in the name.
        VIKA_UNWRAP_OR_RETURN(_check_trailing_extents(inputs[i].extents, d_inputs[i].extents(), __func__,
                                                      ("input " + std::to_string(i)).c_str(), __FILE__, __LINE__));
        if (inputs[i].extents[0] != inputs[0].extents[0])
        {
            return error(VIKA_SHAPE_ERROR("concat forward: input %zu has batch %zu but input 0 has batch %zu", i,
                                          inputs[i].extents[0], inputs[0].extents[0]));
        }
    }

    const usize k = inputs[0].extents[0];
    auto sliced_outputs = VIKA_UNWRAP_OR_RETURN(outputs.view().first_n(k));

    const dim3 block(256);
    usize col_offset = 0;
    for (usize i = 0; i < inputs.size() - 1; ++i)
    {
        // Checked, not waited - see AddLayer::forward for why.
        const auto grid = grid_covering(block, inputs[i].element_count());
        auto job = launch_kernel(concat_copy, sliced_outputs.const_view(), grid, block, stream.handle(), inputs[i],
                                 sliced_outputs, col_offset);
        if (job.is_error())
        {
            return job;
        }
        col_offset += inputs[i].extents.back();
    }

    const auto &last_input = inputs.back();
    const auto grid = grid_covering(block, last_input.element_count());
    return launch_kernel(concat_copy, sliced_outputs.const_view(), grid, block, stream.handle(), last_input,
                         sliced_outputs, col_offset);
}

auto ConcatLayer::backward(const DeviceTensorConstViewf &upstream) -> std::vector<KernelJob<DeviceTensorConstViewf>>
{
    const auto check = VIKA_CHECK_TRAILING_EXTENTS(upstream.extents, outputs.extents());
    if (check.is_error())
    {
        return std::vector<KernelJob<DeviceTensorConstViewf>>(
            d_inputs.size(), KernelJob<DeviceTensorConstViewf>::failed(check.unwrap_error()));
    }

    const usize k = upstream.extents[0];
    const dim3 block(256);
    std::vector<KernelJob<DeviceTensorConstViewf>> jobs;
    jobs.reserve(d_inputs.size());

    usize col_offset = 0;
    for (auto &d_input : d_inputs)
    {
        auto sliced_result = d_input.view().first_n(k);
        if (sliced_result.is_error())
        {
            // Fail every job uniformly rather than return a vector shorter than d_inputs.size().
            return std::vector<KernelJob<DeviceTensorConstViewf>>(
                d_inputs.size(), KernelJob<DeviceTensorConstViewf>::failed(sliced_result.unwrap_error()));
        }
        auto sliced_d_input = sliced_result.unwrap();

        const auto grid = grid_covering(block, sliced_d_input.element_count());
        jobs.push_back(launch_kernel(concat_split, sliced_d_input.const_view(), grid, block, stream.handle(), upstream,
                                     col_offset, sliced_d_input));

        col_offset += sliced_d_input.extents.back();
    }
    return jobs;
}

auto MSELoss::with_extents(const Extents &extents) -> Result<MSELoss, Error>
{
    auto loss = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({1}));
    auto d_inputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(extents));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(MSELoss{.loss = std::move(loss), .d_inputs = std::move(d_inputs), .stream = std::move(stream)});
}

auto MSELoss::forward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
    -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(predictions.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(targets.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_BATCH_AGREEMENT(predictions, targets));

    VIKA_UNWRAP_OR_RETURN(zero(loss.view(), stream.handle()));

    const usize n = predictions.element_count();
    const dim3 block(256);
    return launch_kernel(mse_kernel, loss.const_view(), grid_covering(block, n), block, stream.handle(), predictions,
                         targets, loss.view());
}

auto MSELoss::backward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
    -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(predictions.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(targets.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_BATCH_AGREEMENT(predictions, targets));

    const usize k = predictions.extents[0];
    auto sliced_d_inputs = VIKA_UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const usize n = predictions.element_count();
    const dim3 block(256);
    return launch_kernel(mse_gradient_kernel, sliced_d_inputs.const_view(), grid_covering(block, n), block,
                         stream.handle(), predictions, targets, sliced_d_inputs);
}

auto CCELoss::with_extents(const Extents &extents) -> Result<CCELoss, Error>
{
    auto loss = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty({1}));
    auto d_inputs = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(extents));

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(CCELoss{.loss = std::move(loss), .d_inputs = std::move(d_inputs), .stream = std::move(stream)});
}

auto CCELoss::forward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
    -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(predictions.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(targets.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_BATCH_AGREEMENT(predictions, targets));

    VIKA_UNWRAP_OR_RETURN(zero(loss.view(), stream.handle()));

    const usize n = predictions.element_count();
    const dim3 block(256);
    return launch_kernel(cce_kernel, loss.const_view(), grid_covering(block, n), block, stream.handle(), predictions,
                         targets, loss.view());
}

auto CCELoss::backward(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets)
    -> KernelJob<DeviceTensorConstViewf>
{
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(predictions.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRAILING_EXTENTS(targets.extents, d_inputs.extents()));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_BATCH_AGREEMENT(predictions, targets));

    const usize k = predictions.extents[0];
    auto sliced_d_inputs = VIKA_UNWRAP_OR_RETURN(d_inputs.view().first_n(k));

    const usize n = predictions.element_count();
    const dim3 block(256);
    return launch_kernel(cce_gradient_kernel, sliced_d_inputs.const_view(), grid_covering(block, n), block,
                         stream.handle(), predictions, targets, sliced_d_inputs);
}

// =============================================================================
// Computation Graph
// =============================================================================

auto ComputationGraph::next_seed() -> u32
{
    seed = pcg_hash(seed);
    return seed;
}

auto ComputationGraph::input(Extents spatial_extents) -> Result<NodeId, Error>
{
    // The batch dimension is prepended, so one fewer than the maximum rank is available here.
    // Unchecked, the push_back below would panic on extents a caller can legitimately construct.
    if (spatial_extents.size() + 1 > Extents::capacity())
    {
        return error(VIKA_SHAPE_ERROR("input: %zu spatial dimensions plus the batch exceed the maximum rank of %zu",
                                      spatial_extents.size(), Extents::capacity()));
    }

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
    return ok(id);
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

auto ComputationGraph::softmax(NodeId input) -> Result<NodeId, Error>
{
    if (input.value >= nodes.size())
    {
        return error(VIKA_GRAPH_ERROR("softmax: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = SoftmaxSpec{},
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

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_STRIDE(stride));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_WINDOW_FITS(H, kernel_height, padding, "height"));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_WINDOW_FITS(W, kernel_width, padding, "width"));

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

auto ComputationGraph::conv_transpose2d(NodeId input, usize kernel_height, usize kernel_width, usize channels_out,
                                        usize stride, usize padding, std::optional<u32> requested_seed)
    -> Result<NodeId, Error>
{
    if (input.value >= nodes.size())
    {
        return error(VIKA_GRAPH_ERROR("conv_transpose2d: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() != 4)
    {
        return error(VIKA_SHAPE_ERROR("conv_transpose2d: input must be rank 4 [N, H, W, C]"));
    }

    const auto H = in_extents[1];
    const auto W = in_extents[2];

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_STRIDE(stride));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRANSPOSED_WINDOW_FITS(H, kernel_height, stride, padding, "height"));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_TRANSPOSED_WINDOW_FITS(W, kernel_width, stride, padding, "width"));

    const auto out_H = transposed_window_output_extent(H, kernel_height, stride, padding);
    const auto out_W = transposed_window_output_extent(W, kernel_width, stride, padding);

    const u32 resolved_seed = requested_seed.has_value() ? *requested_seed : next_seed();
    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = ConvTranspose2DSpec{kernel_height, kernel_width, channels_out, stride, padding, resolved_seed},
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

    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_STRIDE(stride));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_WINDOW_FITS(H, pool_height, 0, "height"));
    VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_WINDOW_FITS(W, pool_width, 0, "width"));

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

auto ComputationGraph::upsample2d(NodeId input, usize scale) -> Result<NodeId, Error>
{
    if (input.value >= nodes.size())
    {
        return error(VIKA_GRAPH_ERROR("upsample2d: invalid NodeId"));
    }

    const auto in_extents = nodes[input.value].output_extents;
    if (in_extents.size() != 4)
    {
        return error(VIKA_SHAPE_ERROR("upsample2d: input must be rank 4 [N, H, W, C]"));
    }
    if (scale == 0)
    {
        return error(VIKA_SHAPE_ERROR("upsample2d: scale must be at least 1, got %zu", scale));
    }

    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = Upsample2DSpec{scale},
        .output_extents = {in_extents[0], in_extents[1] * scale, in_extents[2] * scale, in_extents[3]},
        .inputs = {input},
    });
    return ok(id);
}

auto ComputationGraph::add(std::vector<NodeId> inputs) -> Result<NodeId, Error>
{
    if (inputs.size() < 2)
    {
        return error(VIKA_GRAPH_ERROR("add: expects at least 2 inputs, got %zu", inputs.size()));
    }
    if (!all_of(inputs, [&](const NodeId &n) { return n.value < nodes.size(); }))
    {
        return error(VIKA_GRAPH_ERROR("add: invalid NodeId"));
    }

    const auto &first_extents = nodes[inputs[0].value].output_extents;
    if (!all_of(inputs, [&](const NodeId &n) { return nodes[n.value].output_extents == first_extents; }))
    {
        return error(VIKA_SHAPE_ERROR("add: inputs must have the same shape"));
    }

    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = AddSpec{},
        .output_extents = first_extents,
        .inputs = std::move(inputs),
    });
    return ok(id);
}

auto ComputationGraph::concat(std::vector<NodeId> inputs) -> Result<NodeId, Error>
{
    // Ahead of concat_output_extents' own arity check, which would otherwise report a graph
    // mistake - the wrong number of nodes named here - as a shape error, unlike add() beside it.
    if (inputs.size() < 2)
    {
        return error(VIKA_GRAPH_ERROR("concat: expects at least 2 inputs, got %zu", inputs.size()));
    }
    if (!all_of(inputs, [&](const NodeId &n) { return n.value < nodes.size(); }))
    {
        return error(VIKA_GRAPH_ERROR("concat: invalid NodeId"));
    }

    const auto input_extents = map(inputs, [&](const NodeId &input) { return nodes[input.value].output_extents; });
    const auto output_extents = VIKA_UNWRAP_OR_RETURN(concat_output_extents(input_extents));

    const NodeId id{nodes.size()};
    nodes.push_back(Node{
        .spec = ConcatSpec{},
        .output_extents = output_extents,
        .inputs = std::move(inputs),
    });
    return ok(id);
}

auto make_layer(const LayerSpec &spec, usize batch_size, const std::vector<Extents> &pred_extents)
    -> Result<Layer, Error>
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
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 1));
                return DenseLayer::randomized(batch_size, pred_extents[0].at(1), s.output_features, s.seed)
                    .map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, SigmoidSpec>)
            {
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 1));
                return SigmoidLayer::with_extents(pred_extents[0]).map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, SoftmaxSpec>)
            {
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 1));
                return SoftmaxLayer::with_extents(pred_extents[0]).map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, FlattenSpec>)
            {
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 1));
                return Flatten2DLayer::with_extents(pred_extents[0]).map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, Conv2DSpec>)
            {
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 1));
                return Conv2DLayer::randomized(batch_size, pred_extents[0].at(1), pred_extents[0].at(2),
                                               s.kernel_height, s.kernel_width, pred_extents[0].at(3), s.channels_out,
                                               s.stride, s.padding, s.seed)
                    .map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, ConvTranspose2DSpec>)
            {
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 1));
                return ConvTranspose2DLayer::randomized(batch_size, pred_extents[0].at(1), pred_extents[0].at(2),
                                                        s.kernel_height, s.kernel_width, pred_extents[0].at(3),
                                                        s.channels_out, s.stride, s.padding, s.seed)
                    .map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, MaxPool2DSpec>)
            {
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 1));
                return MaxPool2DLayer::with_extents(batch_size, pred_extents[0].at(1), pred_extents[0].at(2),
                                                    pred_extents[0].at(3), s.pool_height, s.pool_width, s.stride)
                    .map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, Upsample2DSpec>)
            {
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 1));
                return Upsample2DLayer::with_extents(batch_size, pred_extents[0].at(1), pred_extents[0].at(2),
                                                     pred_extents[0].at(3), s.scale)
                    .map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, AddSpec>)
            {
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 2));
                return AddLayer::with_extents(pred_extents[0], pred_extents.size()).map(as_trainable);
            }
            else if constexpr (std::is_same_v<T, ConcatSpec>)
            {
                VIKA_UNWRAP_OR_RETURN(VIKA_CHECK_PRED_COUNT(pred_extents, 2));
                return ConcatLayer::with_extents(pred_extents).map(as_trainable);
            }
            else
            {
                static_assert(sizeof(T) == 0, "unhandled LayerSpec type in make_layer");
                return error(VIKA_GRAPH_ERROR("unreachable"));
            }
        },
        spec);
}

auto parameters(const Layer &layer) -> std::optional<std::vector<ConstParameterView>>
{
    return std::visit(
        [](const auto &l) -> std::optional<std::vector<ConstParameterView>> {
            using T = std::decay_t<decltype(l)>;
            if constexpr (is_trainable_layer<T>())
            {
                return l.parameters();
            }
            else
            {
                return std::nullopt;
            }
        },
        layer.kind);
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

    for (usize i = 0; i < nodes.size(); ++i)
    {
        for (const auto &pred : nodes[i].inputs)
        {
            if (pred.value >= nodes.size())
            {
                // nodes is public, so a hand-assembled graph can name one that does not exist -
                // and the walk below would index straight off the end.
                return error(
                    VIKA_GRAPH_ERROR("compile: node %zu names predecessor %zu, which does not exist", i, pred.value));
            }
        }
    }

    // The nodes `output` actually depends on: a graph can hold branches nothing merges back, and
    // building or running those spends memory and launches on a result nobody reads. Slots outside
    // this set stay default constructed and never enter execution_order.
    std::vector<bool> reachable(nodes.size(), false);
    std::vector<usize> pending{output.value};
    reachable[output.value] = true;
    while (!pending.empty())
    {
        const auto idx = pending.back();
        pending.pop_back();
        for (const auto &pred : nodes[idx].inputs)
        {
            if (!reachable[pred.value])
            {
                reachable[pred.value] = true;
                pending.push_back(pred.value);
            }
        }
    }

    AdjacencyGraph<usize> adj{};
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

    const auto topo_order = VIKA_UNWRAP_OR_RETURN(topological_sort(adj));
    if (topo_order.size() != nodes.size())
    {
        // The slots below are filled by index, so a node missing from the sort would leave a
        // default-constructed InputLayer behind - a silent pass-through rather than a failure.
        return error(VIKA_GRAPH_ERROR("compile: sorted %zu of %zu nodes", topo_order.size(), nodes.size()));
    }

    // Indexed by NodeId::value, not by position in topo_order: NodeId is the index every consumer
    // uses. Filling them in traversal order mis-wires any graph whose nodes are not already in
    // topological order, which only the builders guarantee.
    std::vector<Layer> layers(nodes.size());
    std::vector<std::vector<NodeId>> layer_inputs_result(nodes.size());

    for (const auto idx : topo_order)
    {
        if (!reachable[idx])
        {
            continue;
        }

        const auto &node = nodes[idx];
        layer_inputs_result[idx] = node.inputs;

        const auto pred_extents =
            map(node.inputs, [&](const NodeId &pred) { return nodes[pred.value].output_extents; });

        layers[idx] = VIKA_UNWRAP_OR_RETURN(make_layer(node.spec, batch_size, pred_extents));
    }

    std::vector<NodeId> execution_order;
    execution_order.reserve(topo_order.size());
    for (const auto idx : topo_order)
    {
        if (reachable[idx])
        {
            execution_order.push_back(NodeId{idx});
        }
    }

    // Allocated once, here, so that a training step never calls cudaMalloc. adj already says who
    // consumes each node, so this costs no extra pass.
    std::vector<std::optional<DeviceOwningTensorf>> d_outputs(nodes.size());
    for (usize idx = 0; idx < nodes.size(); ++idx)
    {
        if (!reachable[idx])
        {
            continue;
        }

        // backward() skips a node with no predecessors before it accumulates anything, having
        // nothing to propagate the sum to, so a buffer here would be a batch-sized allocation
        // nothing ever reads. Spelled the way backward() spells it, so the two cannot drift.
        if (nodes[idx].inputs.empty())
        {
            continue;
        }

        // Reachable consumers only: one that never runs never contributes a gradient.
        usize live_consumers = 0;
        for (const auto consumer : adj[idx])
        {
            live_consumers += reachable[consumer] ? 1u : 0u;
        }
        if (live_consumers > 1)
        {
            d_outputs[idx] = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::empty(nodes[idx].output_extents));
        }
    }

    auto stream = VIKA_UNWRAP_OR_RETURN(Stream::create());
    return ok(Model{
        .layers = std::move(layers),
        .layer_inputs = std::move(layer_inputs_result),
        .execution_order = std::move(execution_order),
        .input_node = input_node,
        .output_node = output,
        .d_outputs = std::move(d_outputs),
        .stream = std::move(stream),
    });
}

auto Model::forward_output(NodeId node_id) -> Result<DeviceTensorConstViewf, Error>
{
    if (node_id.value >= forward_jobs.size() || !forward_jobs[node_id.value].has_value())
    {
        return error(
            VIKA_GRAPH_ERROR("node %zu has no forward output; was forward() called on this model?", node_id.value));
    }
    return forward_jobs[node_id.value]->wait();
}

auto Model::forward(DeviceTensorConstViewf input) -> Result<DeviceTensorConstViewf, Error>
{
    forward_jobs.clear();
    forward_jobs.resize(layers.size());

    // The last backward differentiated outputs this pass overwrites, so it does not survive.
    backward_jobs.clear();

    for (const auto node_id : execution_order)
    {
        const auto &preds = layer_inputs[node_id.value];

        std::vector<DeviceTensorConstViewf> pred_inputs;
        if (preds.empty())
        {
            pred_inputs.push_back(input);
        }
        else
        {
            pred_inputs =
                VIKA_UNWRAP_OR_RETURN(try_map(preds, [&](const NodeId &pred) { return forward_output(pred); }));
        }

        auto job = std::visit(
            [&pred_inputs](auto &layer) -> KernelJob<DeviceTensorConstViewf> { return layer.forward(pred_inputs); },
            layers[node_id.value].kind);

        forward_jobs[node_id.value] = std::move(job);
    }

    return forward_output(output_node);
}

// Order of summation is the order jobs were pushed, which is execution_order's (deterministic)
// order - floating-point addition isn't associative, so this keeps repeated backward() calls on
// the same model bit-for-bit reproducible.
auto Model::accumulate_output_gradient(NodeId node_id) -> Result<DeviceTensorConstViewf, Error>
{
    if (node_id.value >= backward_jobs.size())
    {
        return error(VIKA_GRAPH_ERROR("accumulate_output_gradient: node %zu is out of range; was backward() called?",
                                      node_id.value));
    }

    auto &jobs = backward_jobs[node_id.value];
    if (jobs.empty())
    {
        // backward() skips such a node before calling here, but this method is public, so a
        // direct call must fail rather than index jobs[0] on an empty vector.
        return error(
            VIKA_GRAPH_ERROR("accumulate_output_gradient: node %zu has no gradient contributions yet", node_id.value));
    }
    if (jobs.size() == 1)
    {
        return jobs[0].wait();
    }

    // compile() allocates this slot for exactly the nodes that can reach the line below - those
    // with more than one consumer. Reaching it without one means the two have drifted apart, which
    // a hand-assembled Model can do, so say so rather than dereference an empty optional.
    if (node_id.value >= d_outputs.size() || !d_outputs[node_id.value].has_value())
    {
        return error(VIKA_GRAPH_ERROR("accumulate_output_gradient: node %zu has %zu contributions to sum but no "
                                      "accumulation buffer; was this model built by compile()?",
                                      node_id.value, jobs.size()));
    }

    const auto first = VIKA_UNWRAP_OR_RETURN(jobs[0].wait());
    const usize k = first.extents[0];
    auto sliced_accum = VIKA_UNWRAP_OR_RETURN(d_outputs[node_id.value]->view().first_n(k));

    VIKA_UNWRAP_OR_RETURN(copy(first, sliced_accum, stream.handle()));

    const dim3 block(256);
    const auto grid = grid_covering(block, first.element_count());
    for (usize i = 1; i + 1 < jobs.size(); ++i)
    {
        const auto contribution = VIKA_UNWRAP_OR_RETURN(jobs[i].wait());
        // Checked, not waited - see AddLayer::forward. These share one stream, so they are already
        // ordered against each other and against the copy above.
        auto job = launch_kernel(accumulate_into, sliced_accum.const_view(), grid, block, stream.handle(), contribution,
                                 sliced_accum);
        if (job.is_error())
        {
            return error(job.result.unwrap_error());
        }
    }

    // Only the last one is waited, and it has to be: what this returns is read by the node's own
    // backward() on a different stream, and travels on as a `ready` job that nothing waits on.
    const auto last = VIKA_UNWRAP_OR_RETURN(jobs.back().wait());
    VIKA_UNWRAP_OR_RETURN(
        launch_kernel(accumulate_into, sliced_accum.const_view(), grid, block, stream.handle(), last, sliced_accum)
            .wait());

    return ok(sliced_accum.const_view());
}

auto Model::backward(DeviceTensorConstViewf loss_grad) -> Result<Void, Error>
{
    // Every layer's backward() reads what its own forward() wrote - Sigmoid and Softmax read
    // `outputs`, MaxPool2D reads `argmax` - and those buffers start uninitialised.
    if (forward_jobs.size() != layers.size())
    {
        return error(VIKA_GRAPH_ERROR("backward: no forward pass to differentiate; call forward() first"));
    }

    backward_jobs.clear();
    backward_jobs.resize(layers.size());
    backward_jobs[output_node.value].push_back(KernelJob<DeviceTensorConstViewf>::ready(loss_grad));

    for (auto it = execution_order.rbegin(); it != execution_order.rend(); ++it)
    {
        const auto node_id = *it;
        const auto &preds = layer_inputs[node_id.value];
        if (preds.empty())
        {
            continue;
        }

        auto &contributions = backward_jobs[node_id.value];
        if (contributions.empty())
        {
            // Nothing reached this node - a branch that never merges back to the output.
            continue;
        }

        const auto upstream = VIKA_UNWRAP_OR_RETURN(accumulate_output_gradient(node_id));

        // Collapse to the accumulated value: step() re-reads this slot for weight_gradients() and
        // needs the sum, not the pieces.
        contributions = {KernelJob<DeviceTensorConstViewf>::ready(upstream)};

        // One job per predecessor, in the same order as preds - every layer type returns this
        // uniformly now (a 1-element vector for every single-input layer), so no per-type
        // special-casing is needed here regardless of how many inputs a layer takes.
        auto pred_jobs =
            std::visit([&upstream](auto &layer) { return layer.backward(upstream); }, layers[node_id.value].kind);
        if (pred_jobs.size() != preds.size())
        {
            // A layer's arity is only knowable by calling backward(), so its work is already in
            // flight. Wait it out and discard, rather than leave it racing the caller.
            for (auto &job : pred_jobs)
            {
                static_cast<void>(job.wait());
            }
            return error(
                VIKA_UNSUPPORTED_ERROR("backward: node %zu's layer returned %zu gradients but has %zu predecessors",
                                       node_id.value, pred_jobs.size(), preds.size()));
        }
        for (usize i = 0; i < preds.size(); ++i)
        {
            backward_jobs[preds[i].value].push_back(std::move(pred_jobs[i]));
        }
    }

    return ok(Void{});
}

auto update_layer(LayerKind &kind, DeviceTensorConstViewf forward_input, DeviceTensorConstViewf upstream,
                  std::vector<AdamState> &states, const AdamParameters &params, usize t) -> Result<Void, Error>
{
    return std::visit(
        [&](auto &l) -> Result<Void, Error> {
            using T = std::decay_t<decltype(l)>;
            if constexpr (is_trainable_layer<T>())
            {
                // weight_gradients() writes into the layer's own grad buffers and update() reads
                // them back from there, so nothing is threaded through here.
                VIKA_UNWRAP_OR_RETURN(l.weight_gradients(forward_input, upstream).wait());

                auto update_jobs = l.update(states, params, t);
                for (auto &result : wait_on_all(update_jobs))
                {
                    VIKA_UNWRAP_OR_RETURN(std::move(result));
                }
            }
            return ok(Void{});
        },
        kind);
}

auto AdamState::create(const Extents &extents) -> Result<AdamState, Error>
{
    auto m = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::zero(extents));
    auto v = VIKA_UNWRAP_OR_RETURN(DeviceOwningTensorf::zero(extents));
    return ok(AdamState{std::move(m), std::move(v)});
}

auto AdamState::from_parameters(const ConstParameterView &parameters) -> Result<AdamState, Error>
{
    return AdamState::create(parameters.value.extents);
}

auto AdamOptimizer::from_model(const Model &model, AdamParameters params) -> Result<AdamOptimizer, Error>
{
    AdamOptimizer optimizer{params, std::vector<std::vector<AdamState>>(model.layers.size()), 0};
    for (const auto node_id : model.execution_order)
    {
        auto maybe_params = parameters(model.layers[node_id.value]);
        if (!maybe_params.has_value())
        {
            continue;
        }

        optimizer.states[node_id.value] = VIKA_UNWRAP_OR_RETURN(try_map(*maybe_params, &AdamState::from_parameters));
    }

    return ok(std::move(optimizer));
}

auto Model::step(AdamOptimizer &optimizer) -> Result<Void, Error>
{
    // Both vectors are sized by their own pass and start empty.
    if (forward_jobs.size() != layers.size())
    {
        return error(VIKA_GRAPH_ERROR("step: no forward pass to step from; call forward() first"));
    }
    if (backward_jobs.size() != layers.size())
    {
        return error(VIKA_GRAPH_ERROR("step: no backward pass to step from; call backward() first"));
    }

    if (optimizer.states.size() != layers.size())
    {
        // An optimizer built from a different model, which would otherwise skip every layer and
        // report success having trained nothing.
        return error(VIKA_GRAPH_ERROR("step: optimizer holds state for %zu nodes but this model has %zu",
                                      optimizer.states.size(), layers.size()));
    }

    // After the guards, so a rejected call does not consume a step.
    ++optimizer.steps_taken;
    const usize t = optimizer.steps_taken;

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

        auto &layer_states = optimizer.states[node_id.value];
        if (layer_states.empty())
        {
            continue;
        }

        // Empty means backward() never reached this node, so there is no gradient to step with.
        if (backward_jobs[node_id.value].empty())
        {
            continue;
        }

        // Only a layer with parameters gets this far, and weight_gradients() takes exactly one
        // forward input - so preds[0] is the whole story only while every trainable layer is
        // single-input. It is the arity counterpart of backward()'s gradient-count check.
        if (preds.size() != 1)
        {
            return error(
                VIKA_UNSUPPORTED_ERROR("step: node %zu has %zu predecessors, but a trainable layer must have exactly "
                                       "one to differentiate against",
                                       node_id.value, preds.size()));
        }

        const auto forward_input = VIKA_UNWRAP_OR_RETURN(forward_output(preds[0]));
        const auto upstream = VIKA_UNWRAP_OR_RETURN(backward_jobs[node_id.value][0].wait());

        VIKA_UNWRAP_OR_RETURN(update_layer(layer.kind, forward_input, upstream, layer_states, optimizer.params, t));
    }

    return ok(Void{});
}

// =============================================================================
// Kernels
// =============================================================================

__global__ auto uniform_tensor_kernel(DeviceTensorViewf tensor, u32 seed) -> void
{
    const usize i = global_thread_x();
    if (i >= tensor.element_count())
    {
        return;
    }
    tensor[i] = uniform_f32((u32)i ^ seed);
}

__global__ auto xavier_tensor_kernel(DeviceTensorViewf tensor, u32 seed, f32 limit) -> void
{
    const usize i = global_thread_x();
    if (i >= tensor.element_count())
    {
        return;
    }
    tensor[i] = (uniform_f32((u32)i ^ seed) * 2.0f - 1.0f) * limit;
}

__global__ auto mse_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets, DeviceTensorViewf out)
    -> void
{
    const usize i = global_thread_x();
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
    const usize i = global_thread_x();
    if (i >= predictions.element_count())
    {
        return;
    }
    out[i] = 2.0f * (predictions[i] - targets[i]) / (f32)predictions.element_count();
}

__global__ auto cce_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets, DeviceTensorViewf out)
    -> void
{
    const usize i = global_thread_x();
    if (i >= predictions.element_count())
    {
        return;
    }
    // Clamped away from 0, not the raw prediction - a probability that's genuinely (or numerically)
    // zero at the one-hot target class would otherwise make log() produce -inf.
    const f32 p = predictions[i] < 1e-12f ? 1e-12f : predictions[i];
    atomicAdd(&out[0], -targets[i] * std::log(p) / (f32)predictions.extents[0]);
}

__global__ auto cce_gradient_kernel(DeviceTensorConstViewf predictions, DeviceTensorConstViewf targets,
                                    DeviceTensorViewf out) -> void
{
    const usize i = global_thread_x();
    if (i >= predictions.element_count())
    {
        return;
    }
    const f32 p = predictions[i] < 1e-12f ? 1e-12f : predictions[i];
    out[i] = -targets[i] / p / (f32)predictions.extents[0];
}

__global__ auto adam_update(const AdamParameters parameters, f32 m_hat_scale, f32 v_hat_scale,
                            DeviceTensorConstViewf d_weights, DeviceTensorViewf weights, DeviceTensorViewf m_weights,
                            DeviceTensorViewf v_weights) -> void
{
    usize i = global_thread_x();
    if (i >= weights.element_count())
    {
        return;
    }

    f32 g = d_weights[i];
    f32 m = m_weights[i];
    f32 v = v_weights[i];

    m = parameters.beta1 * m + (1.0f - parameters.beta1) * g;
    v = parameters.beta2 * v + (1.0f - parameters.beta2) * (g * g);

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
    const usize i = global_thread_x();
    if (i < a.element_count())
    {
        out[i] = sigmoid(a[i]);
    }
}

__global__ auto sigmoid_backward(DeviceTensorConstViewf a, DeviceTensorConstViewf upstream,
                                 DeviceTensorViewf out) -> void
{
    const usize i = global_thread_x();
    if (i < a.element_count())
    {

        out[i] = a[i] * (1.0f - a[i]) * upstream[i];
    }
}

__global__ auto softmax_forward(DeviceTensorConstViewf input, DeviceTensorViewf out) -> void
{
    const usize width = input.extents.back();
    const usize row_count = input.element_count() / width;

    const usize row = global_thread_x();
    if (row >= row_count)
    {
        return;
    }

    // Subtract the row max before exponentiating - same shift-invariance trick as every softmax
    // implementation, so a row of large logits doesn't overflow expf into inf/nan.
    f32 max_val = input[row * width];
    for (usize col = 1; col < width; ++col)
    {
        const f32 val = input[row * width + col];
        if (val > max_val)
        {
            max_val = val;
        }
    }

    f32 sum = 0.0f;
    for (usize col = 0; col < width; ++col)
    {
        const f32 e = std::exp(input[row * width + col] - max_val);
        out[row * width + col] = e;
        sum += e;
    }

    for (usize col = 0; col < width; ++col)
    {
        out[row * width + col] /= sum;
    }
}

__global__ auto softmax_backward(DeviceTensorConstViewf outputs, DeviceTensorConstViewf upstream,
                                 DeviceTensorViewf out) -> void
{
    const usize width = outputs.extents.back();
    const usize row_count = outputs.element_count() / width;

    const usize row = global_thread_x();
    if (row >= row_count)
    {
        return;
    }

    f32 dot = 0.0f;
    for (usize col = 0; col < width; ++col)
    {
        dot += outputs[row * width + col] * upstream[row * width + col];
    }
    for (usize col = 0; col < width; ++col)
    {
        const usize idx = row * width + col;
        out[idx] = outputs[idx] * (upstream[idx] - dot);
    }
}

__global__ auto accumulate_into(DeviceTensorConstViewf delta, DeviceTensorViewf accum) -> void
{
    const usize i = global_thread_x();
    if (i < accum.element_count())
    {
        accum[i] += delta[i];
    }
}

__global__ auto concat_copy(DeviceTensorConstViewf src, DeviceTensorViewf dst, usize dst_col_offset) -> void
{
    const usize i = global_thread_x();
    if (i >= src.element_count())
    {
        return;
    }
    const usize src_width = src.extents.back();
    const usize dst_width = dst.extents.back();
    const usize row = i / src_width;
    const usize col = i % src_width;
    dst[row * dst_width + dst_col_offset + col] = src[i];
}

__global__ auto concat_split(DeviceTensorConstViewf src, usize src_col_offset, DeviceTensorViewf dst) -> void
{
    const usize i = global_thread_x();
    if (i >= dst.element_count())
    {
        return;
    }
    const usize src_width = src.extents.back();
    const usize dst_width = dst.extents.back();
    const usize row = i / dst_width;
    const usize col = i % dst_width;
    dst[i] = src[row * src_width + src_col_offset + col];
}

__global__ auto sum_rows(DeviceTensorConstViewf matrix, DeviceTensorViewf out) -> void
{
    // NOTE: Reduce in blocks?
    const usize row = global_thread_x();
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

__global__ auto add_bias(DeviceTensorConstViewf biases, DeviceTensorViewf out) -> void
{
    const usize sample_index = global_thread_y();
    const usize col = global_thread_x();

    const usize sample_count = out.extents[0];
    const usize bias_count = biases.extents[0];

    if (sample_index >= sample_count || col >= bias_count)
    {
        return;
    }

    out(sample_index, col) += biases[col];
}

__global__ auto matmul_kernel(DeviceTensorConstViewf a, DeviceTensorConstViewf b, DeviceTensorViewf out) -> void
{
    // NOTE: Multiply in tiles?
    const usize row = global_thread_y();
    const usize col = global_thread_x();

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

__global__ auto conv_forward(DeviceTensorConstViewf inputs, DeviceTensorConstViewf filters,
                             DeviceTensorConstViewf biases, DeviceTensorViewf out, usize stride, usize padding) -> void
{
    const usize ow = global_thread_x();
    const usize oh = global_thread_y();
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

__global__ auto conv_backward(DeviceTensorConstViewf upstream, DeviceTensorConstViewf filters,
                              DeviceTensorViewf d_inputs, usize stride, usize padding) -> void
{
    const usize w = global_thread_x();
    const usize h = global_thread_y();
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

__global__ auto conv_weight_gradients(DeviceTensorConstViewf inputs, DeviceTensorConstViewf upstream,
                                      DeviceTensorViewf d_filters, usize stride, usize padding) -> void
{
    const usize idx = global_thread_x();
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

__global__ auto conv_bias_gradients(DeviceTensorConstViewf upstream, DeviceTensorViewf d_biases) -> void
{
    const usize oc = global_thread_x();
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

__global__ auto conv_transpose_forward(DeviceTensorConstViewf inputs, DeviceTensorConstViewf filters,
                                       DeviceTensorConstViewf biases, DeviceTensorViewf out, usize stride,
                                       usize padding) -> void
{
    const usize ow = global_thread_x();
    const usize oh = global_thread_y();
    const usize oc = blockIdx.z % out.extents[3];
    const usize n = blockIdx.z / out.extents[3];

    if (ow >= out.extents[2] || oh >= out.extents[1] || n >= out.extents[0])
    {
        return;
    }

    const usize kH = filters.extents[0];
    const usize kW = filters.extents[1];
    const usize C_in = filters.extents[3];
    const usize H_in = inputs.extents[1];
    const usize W_in = inputs.extents[2];

    f32 sum = biases[oc];
    for (usize kh = 0; kh < kH; ++kh)
    {
        for (usize kw = 0; kw < kW; ++kw)
        {
            // The input position that maps to (oh, ow) via filter (kh, kw) - same relation
            // conv_backward uses to go from an output position to an input position, since this
            // is that exact computation with the roles of "small" and "large" tensor swapped:
            // ih * stride = oh + padding - kh  (must be a non-negative multiple of stride)
            if (oh + padding < kh || ow + padding < kw)
            {
                continue;
            }
            const usize ih_unpadded = oh + padding - kh;
            const usize iw_unpadded = ow + padding - kw;
            if (ih_unpadded % stride != 0 || iw_unpadded % stride != 0)
            {
                continue;
            }
            const usize ih = ih_unpadded / stride;
            const usize iw = iw_unpadded / stride;
            if (ih >= H_in || iw >= W_in)
            {
                continue;
            }
            for (usize ic = 0; ic < C_in; ++ic)
            {
                sum += inputs(n, ih, iw, ic) * filters(kh, kw, oc, ic);
            }
        }
    }
    out(n, oh, ow, oc) = sum;
}

__global__ auto conv_transpose_backward(DeviceTensorConstViewf upstream, DeviceTensorConstViewf filters,
                                        DeviceTensorViewf d_inputs, usize stride, usize padding) -> void
{
    const usize iw = global_thread_x();
    const usize ih = global_thread_y();
    const usize ic = blockIdx.z % d_inputs.extents[3];
    const usize n = blockIdx.z / d_inputs.extents[3];

    if (iw >= d_inputs.extents[2] || ih >= d_inputs.extents[1] || n >= d_inputs.extents[0])
    {
        return;
    }

    const usize kH = filters.extents[0];
    const usize kW = filters.extents[1];
    const usize C_out = filters.extents[2];
    const usize H_out = upstream.extents[1];
    const usize W_out = upstream.extents[2];

    // Same accumulation shape as conv_forward - each output position this input position feeds
    // is found by sliding the kernel the ordinary (not inverse) way, since backward-of-transpose
    // is dual to forward-of-conv.
    f32 sum = 0.0f;
    for (usize kh = 0; kh < kH; ++kh)
    {
        for (usize kw = 0; kw < kW; ++kw)
        {
            const usize oh_unpadded = ih * stride + kh;
            const usize ow_unpadded = iw * stride + kw;
            if (oh_unpadded >= padding && oh_unpadded - padding < H_out && ow_unpadded >= padding &&
                ow_unpadded - padding < W_out)
            {
                const usize oh = oh_unpadded - padding;
                const usize ow = ow_unpadded - padding;
                for (usize oc = 0; oc < C_out; ++oc)
                {
                    sum += upstream(n, oh, ow, oc) * filters(kh, kw, oc, ic);
                }
            }
        }
    }
    d_inputs(n, ih, iw, ic) = sum;
}

__global__ auto conv_transpose_weight_gradients(DeviceTensorConstViewf inputs, DeviceTensorConstViewf upstream,
                                                DeviceTensorViewf d_filters, usize stride, usize padding) -> void
{
    const usize idx = global_thread_x();
    if (idx >= d_filters.element_count())
    {
        return;
    }

    const usize C_in = d_filters.extents[3];
    const usize C_out = d_filters.extents[2];
    const usize kW = d_filters.extents[1];

    const usize ic = idx % C_in;
    const usize oc = (idx / C_in) % C_out;
    const usize kw = (idx / (C_in * C_out)) % kW;
    const usize kh = idx / (C_in * C_out * kW);

    const usize N = inputs.extents[0];
    const usize H_in = inputs.extents[1];
    const usize W_in = inputs.extents[2];
    const usize H_out = upstream.extents[1];
    const usize W_out = upstream.extents[2];

    // Sums over the (oh, ow) positions this filter tap contributes to, mapping each back to its
    // (ih, iw) source the same way conv_transpose_backward does (this is that same large-to-small
    // coordinate mapping, not conv_weight_gradients' small-to-large one).
    f32 sum = 0.0f;
    for (usize n = 0; n < N; ++n)
    {
        for (usize oh = 0; oh < H_out; ++oh)
        {
            if (oh + padding < kh)
            {
                continue;
            }
            const usize ih_unpadded = oh + padding - kh;
            if (ih_unpadded % stride != 0)
            {
                continue;
            }
            const usize ih = ih_unpadded / stride;
            if (ih >= H_in)
            {
                continue;
            }
            for (usize ow = 0; ow < W_out; ++ow)
            {
                if (ow + padding < kw)
                {
                    continue;
                }
                const usize iw_unpadded = ow + padding - kw;
                if (iw_unpadded % stride != 0)
                {
                    continue;
                }
                const usize iw = iw_unpadded / stride;
                if (iw >= W_in)
                {
                    continue;
                }
                sum += inputs(n, ih, iw, ic) * upstream(n, oh, ow, oc);
            }
        }
    }
    d_filters(kh, kw, oc, ic) = sum;
}

__global__ auto maxpool_forward(DeviceTensorConstViewf inputs, DeviceTensorViewf out, DeviceTensorViewu argmax,
                                usize pool_h, usize pool_w, usize stride) -> void
{
    const usize ow = global_thread_x();
    const usize oh = global_thread_y();
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

__global__ auto maxpool_backward(DeviceTensorConstViewf upstream, DeviceTensorConstViewu argmax,
                                 DeviceTensorViewf d_inputs) -> void
{
    const usize ow = global_thread_x();
    const usize oh = global_thread_y();
    const usize c = blockIdx.z % upstream.extents[3];
    const usize n = blockIdx.z / upstream.extents[3];

    if (ow >= upstream.extents[2] || oh >= upstream.extents[1] || n >= upstream.extents[0])
    {
        return;
    }

    const usize H_in = d_inputs.extents[1];
    const usize W_in = d_inputs.extents[2];
    const u32 idx = argmax(n, oh, ow, c);

    // argmax is written by forward() and read here, and a standalone layer's backward() can be
    // called first - on memory from empty(). Unchecked, the scatter below would write at an
    // arbitrary offset.
    if (idx >= H_in * W_in)
    {
        return;
    }

    const usize ih = idx / W_in;
    const usize iw = idx % W_in;

    atomicAdd(&d_inputs(n, ih, iw, c), upstream(n, oh, ow, c));
}

__global__ auto upsample2d_forward(DeviceTensorConstViewf inputs, DeviceTensorViewf out, usize scale) -> void
{
    const usize ow = global_thread_x();
    const usize oh = global_thread_y();
    const usize c = blockIdx.z % out.extents[3];
    const usize n = blockIdx.z / out.extents[3];

    if (ow >= out.extents[2] || oh >= out.extents[1] || n >= out.extents[0])
    {
        return;
    }

    out(n, oh, ow, c) = inputs(n, oh / scale, ow / scale, c);
}

__global__ auto upsample2d_backward(DeviceTensorConstViewf upstream, DeviceTensorViewf d_inputs, usize scale) -> void
{
    const usize iw = global_thread_x();
    const usize ih = global_thread_y();
    const usize c = blockIdx.z % d_inputs.extents[3];
    const usize n = blockIdx.z / d_inputs.extents[3];

    if (iw >= d_inputs.extents[2] || ih >= d_inputs.extents[1] || n >= d_inputs.extents[0])
    {
        return;
    }

    f32 sum = 0.0f;
    for (usize dh = 0; dh < scale; ++dh)
    {
        for (usize dw = 0; dw < scale; ++dw)
        {
            sum += upstream(n, ih * scale + dh, iw * scale + dw, c);
        }
    }
    d_inputs(n, ih, iw, c) = sum;
}

}; // namespace vika
#endif

// TODO (ecrt):
// - Tiled matmul
//
// - Pick device?
// - Sequential model
// - Multi-output
// - unet
// - Save weights
// - Load weights
