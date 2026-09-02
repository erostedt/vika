#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "vika.cuh"

// Error-path assertions. is_error() on its own is satisfied by any failure at all, so a test named
// for one mistake is happy with another - a stride check passing on an arity error, say. This pins
// the kind and prints what actually arrived, which EXPECT_TRUE cannot do on its own.
//
// Deliberately not matching the message too: that couples a test to wording, and a refactor that
// only rephrases an error should not fail one.
template <typename R>
inline auto failed_with(const R &result, vika::ErrorKind kind) -> bool
{
    if (!result.is_error())
    {
        printf("      expected a %s error, got a value\n", vika::error_kind_name(kind));
        return false;
    }

    const auto &err = result.unwrap_error();
    if (err.kind() != kind)
    {
        printf("      expected a %s error\n           got %s\n", vika::error_kind_name(kind),
               err.describe().c_str());
        return false;
    }
    return true;
}

// A KernelJob carries its Result rather than being one, and several call sites assert on the job
// straight from forward() without waiting.
template <typename T>
inline auto failed_with(const vika::KernelJob<T> &job, vika::ErrorKind kind) -> bool
{
    return failed_with(job.result, kind);
}

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
