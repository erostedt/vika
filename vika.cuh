#pragma once

#include <cstdint>
#include <cuda_runtime.h>

using u32 = uint32_t;
using i32 = int32_t;

__global__ void double_kernel(float *data, u32 n)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        data[i] *= 2.0f;
    }
}
