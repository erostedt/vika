#include <algorithm>
#include <vector>

#include "vika.cuh"

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto are_equal(const vika::HostTensor<T> &actual, const vika::HostTensor<T> &expected) -> bool
{
    if (actual.extents() != expected.extents())
    {
        return false;
    }
    return std::equal(actual.begin(), actual.end(), expected.begin());
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto are_close(const vika::HostTensor<T> &actual, const vika::HostTensor<T> &expected, T tol) -> bool
{
    if (actual.extents() != expected.extents())
    {
        return false;
    }
    for (vika::usize i = 0; i < actual.size(); ++i)
    {
        // Negated rather than written as `> tol`: every comparison against a NaN is false, so the
        // direct form lets a NaN fall through to `return true` and pass. This way it fails.
        if (!(std::abs(actual[i] - expected[i]) <= tol))
        {
            return false;
        }
    }
    return true;
}

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
inline auto are_close(const vika::HostTensor<T> &actual, const std::vector<T> &expected, T tol) -> bool
{
    if (actual.size() != expected.size())
    {
        return false;
    }
    for (vika::usize i = 0; i < actual.size(); ++i)
    {
        // Negated rather than written as `> tol`: every comparison against a NaN is false, so the
        // direct form lets a NaN fall through to `return true` and pass. This way it fails.
        if (!(std::abs(actual[i] - expected[i]) <= tol))
        {
            return false;
        }
    }
    return true;
}
