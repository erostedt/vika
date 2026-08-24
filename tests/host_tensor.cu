#include "utest.h"

#include "vika.cuh"

using namespace vika;
UTEST(tensor, zero_1d)
{
    const auto t = HostTensor1f::zero(4).unwrap();
    EXPECT_EQ(t.size(), 4u);
    EXPECT_EQ(t.extent(0), 4u);
    for (usize i = 0; i < t.size(); ++i)
    {
        EXPECT_EQ(t[i], 0.0f);
    }
}

UTEST(tensor, zero_2d)
{
    const auto t = HostTensor2f::zero({2, 3}).unwrap();
    EXPECT_EQ(t.extent(0), 2u);
    EXPECT_EQ(t.extent(1), 3u);
    EXPECT_EQ(t.size(), 6u);
    for (usize i = 0; i < t.size(); ++i)
    {
        EXPECT_EQ(t[i], 0.0f);
    }
}

UTEST(tensor, size_from_extents)
{
    const auto count = HostTensor2f::size({3, 4});
    EXPECT_EQ(count, 12u);
}

UTEST(tensor, zero_like)
{
    const auto base = HostTensor2f::zero({3, 2}).unwrap();
    const auto copy = HostTensor2f::zero_like(base).unwrap();
    EXPECT_TRUE(copy.extents() == base.extents());
    for (usize i = 0; i < copy.size(); ++i)
    {
        EXPECT_EQ(copy[i], 0.0f);
    }
}

UTEST(tensor, row_major_indexing)
{
    auto t = HostTensor2f::zero({2, 3}).unwrap();
    t(0, 0) = 1.0f;
    t(0, 1) = 2.0f;
    t(0, 2) = 3.0f;
    t(1, 0) = 4.0f;
    t(1, 1) = 5.0f;
    t(1, 2) = 6.0f;

    EXPECT_EQ(t[0], 1.0f);
    EXPECT_EQ(t[1], 2.0f);
    EXPECT_EQ(t[2], 3.0f);
    EXPECT_EQ(t[3], 4.0f);
    EXPECT_EQ(t[4], 5.0f);
    EXPECT_EQ(t[5], 6.0f);
}

// The copy in DeviceOwningTensor::from is sized from extents, not from the vector, so a length
// mismatch used to read past the vector's end and report success.
UTEST(tensor, device_from_rejects_wrong_length)
{
    const auto too_short = DeviceOwningTensorf::from(std::vector<f32>{1.0f, 2.0f}, {8});
    EXPECT_TRUE(too_short.is_error());

    const auto too_long = DeviceOwningTensorf::from(std::vector<f32>(8, 1.0f), {2});
    EXPECT_TRUE(too_long.is_error());

    const auto exact = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    ASSERT_TRUE(exact.is_ok());

    const auto host = download(exact.unwrap()).unwrap();
    EXPECT_EQ(host.size(), 6u);
    EXPECT_EQ(host(1, 2), 6.0f);
}
