// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#ifndef FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANRASTERIZEKERNEL_CUH
#define FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANRASTERIZEKERNEL_CUH

#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>

#include <cstddef>

namespace fvdb::detail::ops {

// Device-only channel specialization. Keep these traits out of host argument types so tensor
// preparation and launch code are compiled once per argument layout.
template <auto NUM_CHANNELS, auto NUM_SHARED_CHANNELS = NUM_CHANNELS> struct RasterizeChannels {
    static constexpr auto COUNT         = NUM_CHANNELS;
    static constexpr auto SHARED_COUNT  = NUM_SHARED_CHANNELS;
    static constexpr size_t CHUNK_COUNT = (COUNT + SHARED_COUNT - 1) / SHARED_COUNT;
    static constexpr bool IS_CHUNKED    = COUNT != SHARED_COUNT;

    // Fully unroll small feature vectors, but bound compiler work for large channel counts.
    static constexpr auto UNROLL        = COUNT <= 32 ? COUNT : 4;
    static constexpr auto SHARED_UNROLL = COUNT <= 32 || SHARED_COUNT <= 32 ? SHARED_COUNT : 4;
};

template <typename Args> using RasterizeKernel = void (*)(Args);

// Tie the selected kernel to its argument type; type erasure is confined to the CUDA API call.
template <typename Args>
inline void
launchRasterizeKernel(RasterizeKernel<Args> kernel,
                      dim3 gridDim,
                      dim3 blockDim,
                      Args &args,
                      size_t sharedMem,
                      cudaStream_t stream) {
    void *kernelArgs[] = {&args};
    C10_CUDA_CHECK(cudaLaunchKernel(
        reinterpret_cast<const void *>(kernel), gridDim, blockDim, kernelArgs, sharedMem, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace fvdb::detail::ops

#endif // FVDB_DETAIL_UTILS_GSPLAT_GAUSSIANRASTERIZEKERNEL_CUH
