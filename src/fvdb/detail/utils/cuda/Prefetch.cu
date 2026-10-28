// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//

#include <fvdb/detail/utils/cuda/Prefetch.h>

#include <nanovdb/util/cuda/Util.h>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

namespace fvdb {
namespace detail {

void
memDiscardAndPrefetchBatchAsync(std::vector<void *> &prefetchPointers,
                                std::vector<size_t> &prefetchSizes,
                                int deviceId,
                                cudaStream_t stream) {
    TORCH_CHECK(stream, "cudaMemDiscardAndPrefetchBatchAsync does not support the default stream");
    if (prefetchPointers.empty()) {
        return;
    }

#if (CUDART_VERSION < 13000)
    for (size_t i = 0; i < prefetchPointers.size(); ++i) {
        C10_CUDA_CHECK(nanovdb::util::cuda::memPrefetchAsync(
            prefetchPointers[i], prefetchSizes[i], deviceId, stream));
    }
#else
    const cudaMemLocation location                 = {cudaMemLocationTypeDevice, deviceId};
    std::vector<cudaMemLocation> prefetchLocations = {location};
    std::vector<size_t> prefetchLocationIndices    = {0};

    C10_CUDA_CHECK(cudaMemDiscardAndPrefetchBatchAsync(prefetchPointers.data(),
                                                       prefetchSizes.data(),
                                                       prefetchPointers.size(),
                                                       prefetchLocations.data(),
                                                       prefetchLocationIndices.data(),
                                                       prefetchLocations.size(),
                                                       0,
                                                       stream));
#endif
}

} // namespace detail
} // namespace fvdb
