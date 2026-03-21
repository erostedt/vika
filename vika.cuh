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
class CudaOwningBuffer
{
    using Self = CudaOwningBuffer<T, Dimensions>;

  public:
    static auto create(const std::array<usize, Dimensions> &extents) -> Result<Self, cudaError_t>
    {
        T *ptr = nullptr;
        const auto err = cudaMalloc(&ptr, ::byte_count<T>(extents) * sizeof(T));
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

    auto element_count() const -> usize
    {
        return ::element_count<Dimensions>(_extents);
    }

    auto byte_count() const -> usize
    {
        return ::byte_count<T>(_extents);
    }

    auto data() -> T *
    {
        return _data.get();
    }

  private:
    CudaOwningBuffer(T *data, const std::array<usize, Dimensions> &extents) : _data(data), _extents(extents)
    {
    }

  private:
    std::unique_ptr<T[], CudaDeleter<T>> _data;
    std::array<usize, Dimensions> _extents;
};

__global__ void double_kernel(f32 *data, u32 n)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        data[i] *= 2.0f;
    }
}
