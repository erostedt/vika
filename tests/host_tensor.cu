#include "utest.h"

#include "comparison.cuh"
#include "vika.cuh"

using namespace vika;
UTEST(tensor, zero_1d)
{
    const auto t = HostTensorf::zero(4).unwrap();
    EXPECT_EQ(t.size(), 4u);
    EXPECT_EQ(t.extent(0), 4u);
    for (usize i = 0; i < t.size(); ++i)
    {
        EXPECT_EQ(t[i], 0.0f);
    }
}

UTEST(tensor, zero_2d)
{
    const auto t = HostTensorf::zero({2, 3}).unwrap();
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
    const auto count = element_count({3, 4});
    EXPECT_EQ(count, 12u);
}

UTEST(tensor, zero_like)
{
    const auto base = HostTensorf::zero({3, 2}).unwrap();
    const auto copy = HostTensorf::zero_like(base).unwrap();
    EXPECT_TRUE(copy.extents() == base.extents());
    for (usize i = 0; i < copy.size(); ++i)
    {
        EXPECT_EQ(copy[i], 0.0f);
    }
}

UTEST(tensor, row_major_indexing)
{
    auto t = HostTensorf::zero({2, 3}).unwrap();
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
    EXPECT_TRUE(failed_with(too_short, ErrorKind::Shape, "holds 2 elements but extents describe 8"));

    const auto too_long = DeviceOwningTensorf::from(std::vector<f32>(8, 1.0f), {2});
    EXPECT_TRUE(failed_with(too_long, ErrorKind::Shape, "holds 8 elements but extents describe 2"));

    const auto exact = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    ASSERT_TRUE(exact.is_ok());

    const auto host = download(exact.unwrap()).unwrap();
    EXPECT_EQ(host.size(), 6u);
    EXPECT_EQ(host(1, 2), 6.0f);
}

// The device side used to check only for overflow, so these two returned a zero-byte and a rank-0
// allocation respectively, failing much later as an invalid launch configuration.
UTEST(tensor, device_empty_rejects_degenerate_extents)
{
    EXPECT_TRUE(failed_with(DeviceOwningTensorf::empty({0, 5}), ErrorKind::Shape, "tensor extents contain a zero extent"));
    EXPECT_TRUE(failed_with(DeviceOwningTensorf::empty({}), ErrorKind::Shape, "tensor extents are empty"));
    EXPECT_TRUE(failed_with(HostTensorf::zero({0, 5}), ErrorKind::Shape, "tensor extents contain a zero extent"));

    ASSERT_TRUE(DeviceOwningTensorf::empty({2, 3}).is_ok());
}

// transposed() swaps extents[0]/[1] and strides[0]/[1]; on a lower-rank view those are the zeroed
// slots past size(), which produced a view of extent 0 that every kernel then skipped silently.
UTEST(tensor, transposed_requires_rank_2)
{
    const auto matrix = DeviceOwningTensorf::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}).unwrap();
    const auto flipped = transposed(matrix.const_view());
    ASSERT_TRUE(flipped.is_ok());
    EXPECT_EQ(flipped.unwrap().extents[0], 3u);
    EXPECT_EQ(flipped.unwrap().extents[1], 2u);
    EXPECT_EQ(flipped.unwrap().strides[0], 1u);
    EXPECT_EQ(flipped.unwrap().strides[1], 3u);

    const auto vector = DeviceOwningTensorf::from({1.0f, 2.0f}, {2}).unwrap();
    EXPECT_TRUE(failed_with(transposed(vector.const_view()), ErrorKind::Shape, "got rank 1"));

    const auto cube = DeviceOwningTensorf::zero({2, 2, 2}).unwrap();
    EXPECT_TRUE(failed_with(transposed(cube.const_view()), ErrorKind::Shape, "got rank 3"));
}
