// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
#define FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H

#include <fvdb/detail/utils/cuda/LocalGradient.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <torch/csrc/cuda/nccl.h>
#include <torch/types.h>

#include <vector>

namespace fvdb::detail {

// Reduce in place into each rank's owned slice of its local gradient buffer. Vector order defines
// ranks; the tensors may reside on any distinct CUDA devices. Inputs must come from
// makeLocalGradient() with padding for localGradients.size() equally sized NCCL shards.
inline void
reduceGradientShards(const std::vector<torch::Tensor> &localGradients) {
    TORCH_CHECK(!localGradients.empty(), "Gradient reduction requires at least one local gradient");
    const int64_t numElements = localGradients.front().numel();
    if (numElements == 0) {
        return;
    }

    const int64_t rankCount         = localGradients.size();
    const int64_t shardSize         = localGradientShardSize(numElements, rankCount);
    const int64_t paddedNumElements = shardSize * rankCount;
    std::vector<torch::Tensor> paddedGradients(rankCount);
    std::vector<torch::Tensor> reducedShards(rankCount);
    for (const auto rank: c10::irange(rankCount)) {
        const auto &localGradient = localGradients[rank];
        TORCH_CHECK(
            localGradient.storage_offset() == 0,
            "Local gradient must start at the beginning of its storage; use makeLocalGradient()");
        const size_t storageBytes = localGradient.storage().nbytes();
        const size_t elementSize  = localGradient.element_size();
        TORCH_CHECK(
            storageBytes % elementSize == 0 &&
                storageBytes / elementSize == static_cast<size_t>(paddedNumElements),
            "Local gradient storage must match the padded reduction size; use makeLocalGradient()");
        // Expose the allocation's zeroed tail without copying or changing the logical gradient.
        paddedGradients[rank] = localGradient.as_strided({paddedNumElements}, {1});
        reducedShards[rank]   = paddedGradients[rank].narrow(0, rank * shardSize, shardSize);
    }

    // NCCL supports in-place reduce-scatter when each receive buffer is its rank's input slice.
    // Ranks with no logical elements still participate using their zero-filled padded slice.
    torch::cuda::nccl::reduce_scatter(paddedGradients, reducedShards);
}

// Call after queuing the reductions and waiting for output prefetching on the current streams.
// Keep the local gradient buffers alive until all output copies have been queued.
inline void
copyGradientShards(const std::vector<torch::Tensor> &localGradients,
                   torch::Tensor &outputGradient) {
    TORCH_CHECK(!localGradients.empty(), "Gradient copy requires at least one local gradient");
    TORCH_CHECK(outputGradient.is_contiguous(), "Gradient copy requires contiguous outputs");
    for (const auto &localGradient: localGradients) {
        TORCH_CHECK(localGradient.is_contiguous() &&
                        localGradient.sizes() == outputGradient.sizes() &&
                        localGradient.scalar_type() == outputGradient.scalar_type(),
                    "Gradient copy requires contiguous inputs with matching shapes and dtypes");
    }

    const int64_t numElements = outputGradient.numel();
    const size_t elementSize  = outputGradient.element_size();
    const auto rankCount      = localGradients.size();
    for (const auto rank: c10::irange(rankCount)) {
        const auto [shardOffset, shardSize] = deviceAlignedChunk(1, numElements, rank, rankCount);
        if (shardSize == 0) {
            continue;
        }

        const auto deviceId = localGradients[rank].get_device();
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        const size_t byteOffset = shardOffset * elementSize;
        C10_CUDA_CHECK(
            cudaMemcpyAsync(static_cast<char *>(outputGradient.data_ptr()) + byteOffset,
                            static_cast<const char *>(localGradients[rank].data_ptr()) + byteOffset,
                            shardSize * elementSize,
                            cudaMemcpyDeviceToDevice,
                            stream));
    }
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
