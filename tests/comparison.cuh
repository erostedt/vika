#include <algorithm>
#include <vector>

#include "vika.cuh"

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
inline auto are_equal(const vika::HostTensor<T, Rank> &actual, const vika::HostTensor<T, Rank> &expected) -> bool
{
    if (actual.extents() != expected.extents())
    {
        return false;
    }
    return std::equal(actual.begin(), actual.end(), expected.begin());
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
inline auto are_close(const vika::HostTensor<T, Rank> &actual, const vika::HostTensor<T, Rank> &expected, T tol) -> bool
{
    if (actual.extents() != expected.extents())
    {
        return false;
    }
    for (usize i = 0; i < actual.size(); ++i)
    {
        if (std::abs(actual[i] - expected[i]) > tol)
        {
            return false;
        }
    }
    return true;
}

template <typename T, usize Rank, typename = std::enable_if_t<(std::is_arithmetic_v<T> && Rank > 0)>>
inline auto are_close(const vika::HostTensor<T, Rank> &actual, const std::vector<T> &expected, T tol) -> bool
{
    if (actual.size() != expected.size())
    {
        return false;
    }
    for (usize i = 0; i < actual.size(); ++i)
    {
        if (std::abs(actual[i] - expected[i]) > tol)
        {
            return false;
        }
    }
    return true;
}
