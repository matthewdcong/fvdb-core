// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/cuda/GradientReduction.h>
#include <fvdb/detail/utils/cuda/LocalGradient.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <torch/csrc/cuda/nccl.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <tuple>

namespace fvdb::detail {
namespace {

void
reduceFlattenedGradients(const std::vector<torch::Tensor> &localGradients) {
    const int64_t numElements = localGradients.front().numel();
    if (numElements == 0) {
        return;
    }

    const int64_t deviceCount       = c10::cuda::device_count();
    const int64_t shardSize         = localGradientShardSize(numElements, deviceCount);
    const int64_t paddedNumElements = shardSize * deviceCount;
    std::vector<torch::Tensor> paddedGradients(deviceCount);
    std::vector<torch::Tensor> reducedShards(deviceCount);
    for (const auto deviceId: c10::irange(deviceCount)) {
        const auto &localGradient = localGradients[deviceId];
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
        paddedGradients[deviceId] = localGradient.as_strided({paddedNumElements}, {1});
        reducedShards[deviceId] =
            paddedGradients[deviceId].narrow(0, deviceId * shardSize, shardSize);
    }

    // NCCL supports in-place reduce-scatter when each receive buffer is its rank's input slice.
    // Ranks with no logical elements still participate using their zero-filled padded slice.
    torch::cuda::nccl::reduce_scatter(paddedGradients, reducedShards);
}

void
reduceCameraGradients(const std::vector<torch::Tensor> &localGradients,
                      const DeviceLists &deviceLists) {
    const auto deviceCount = c10::cuda::device_count();
    TORCH_CHECK(std::ssize(localGradients) == deviceCount && !localGradients.empty(),
                "Camera gradients require one local buffer per CUDA device");
    const auto &first = localGradients.front();
    TORCH_CHECK(first.dim() >= 2 && std::ssize(deviceLists) == first.size(0),
                "Camera contributors must match the [C, N, ...] gradient shape");
    for (const auto &sources: deviceLists) {
        c10::DeviceIndex previous = -1;
        for (const auto source: sources) {
            TORCH_CHECK(source > previous && source < deviceCount,
                        "Camera contributors must be unique, sorted CUDA device IDs");
            previous = source;
        }
    }
    if (first.numel() == 0) {
        return;
    }

    const int64_t numGaussians = first.size(1);
    const int64_t elementsPerGaussian = first.numel() / first.size(0) / numGaussians;
    const int64_t shardSize = localGradientShardSize(numGaussians, deviceCount) *
                              elementsPerGaussian;
    const int64_t cameraSize = shardSize * deviceCount;
    for (const auto deviceId: c10::irange(deviceCount)) {
        const auto &gradient = localGradients[deviceId];
        TORCH_CHECK(gradient.is_cuda() && gradient.get_device() == deviceId &&
                        gradient.sizes() == first.sizes() &&
                        gradient.scalar_type() == first.scalar_type(),
                    "Camera gradients must have matching shapes and dtypes on their CUDA devices");
        TORCH_CHECK(gradient.storage_offset() == 0 && gradient.stride(0) == cameraSize &&
                        gradient.select(0, 0).is_contiguous() &&
                        gradient.storage().nbytes() / gradient.element_size() ==
                            static_cast<size_t>(first.size(0) * cameraSize),
                    "Camera gradient storage must use makeLocalGradient<false>()");
    }

    // Reuse the device-ordered communicator cache used by PyTorch's collectives. Resolve it
    // before grouping transfers because a cache miss initializes the communicators.
    const auto comms = torch::cuda::nccl::detail::get_communicators(localGradients);
    for (const auto cameraId: c10::irange(first.size(0))) {
        const auto &sources = deviceLists[cameraId];
        if (sources.empty()) {
            continue; // The zero-initialized local slices already hold this camera's result.
        }
        if (std::ssize(sources) == deviceCount) {
            std::vector<torch::Tensor> inputs(deviceCount), outputs(deviceCount);
            for (const auto deviceId: c10::irange(deviceCount)) {
                inputs[deviceId] = localGradients[deviceId].as_strided(
                    {cameraSize}, {1}, cameraId * cameraSize);
                outputs[deviceId] = inputs[deviceId].narrow(0, deviceId * shardSize, shardSize);
            }
            torch::cuda::nccl::reduce_scatter(inputs, outputs);
            continue;
        }

        std::vector<torch::Tensor> outputs(deviceCount), received(deviceCount);
        for (const auto deviceId: c10::irange(deviceCount)) {
            int64_t gaussianOffset, gaussianCount;
            std::tie(gaussianOffset, gaussianCount) = deviceChunk(numGaussians, deviceId);
            if (gaussianCount == 0) {
                continue;
            }
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            outputs[deviceId] = localGradients[deviceId]
                                    .select(0, cameraId)
                                    .narrow(0, gaussianOffset, gaussianCount)
                                    .view({-1});
            // One contribution is already local or received straight into the output slice.
            // Only the remaining contributions need separate receive buffers for summation.
            received[deviceId] = torch::empty(
                {std::ssize(sources) - 1, outputs[deviceId].numel()},
                localGradients[deviceId].options());
        }
        {
            torch::cuda::nccl::AutoNcclGroup group;
            for (const auto deviceId: c10::irange(deviceCount)) {
                if (!outputs[deviceId].defined()) {
                    continue;
                }
                int64_t gaussianOffset, gaussianCount;
                std::tie(gaussianOffset, gaussianCount) = deviceChunk(numGaussians, deviceId);
                bool hasContribution = std::binary_search(sources.begin(), sources.end(), deviceId);
                int64_t receiveIndex = 0;
                for (const auto source: sources) {
                    if (source == deviceId) {
                        continue;
                    }
                    auto input = localGradients[source]
                                     .select(0, cameraId)
                                     .narrow(0, gaussianOffset, gaussianCount)
                                     .view({-1});
                    auto output = hasContribution ? received[deviceId].select(0, receiveIndex++)
                                                  : outputs[deviceId];
                    hasContribution = true;
                    C10_CUDA_CHECK(cudaSetDevice(source));
                    auto stream = c10::cuda::getCurrentCUDAStream(source);
                    torch::cuda::nccl::send(input, comms[source], stream, deviceId);
                    C10_CUDA_CHECK(cudaSetDevice(deviceId));
                    stream = c10::cuda::getCurrentCUDAStream(deviceId);
                    torch::cuda::nccl::recv(output, comms[deviceId], stream, source);
                }
            }
        }
        // The group must finish enqueueing before CUDA consumers or receive-buffer releases.
        for (const auto deviceId: c10::irange(deviceCount)) {
            if (!outputs[deviceId].defined()) {
                continue;
            }
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            for (const auto receiveIndex: c10::irange(received[deviceId].size(0))) {
                outputs[deviceId].add_(received[deviceId].select(0, receiveIndex));
            }
        }
    }
}

} // namespace

DeviceLists
makeDeviceLists(int64_t numCameras,
                   int64_t tilesPerCamera,
                   int64_t tileAlignment,
                   int64_t deviceCount,
                   std::optional<int64_t> numActiveTiles) {
    TORCH_CHECK(numCameras >= 0 && tilesPerCamera >= 0 && tileAlignment > 0 && deviceCount > 0,
                "Invalid camera tile partition");
    TORCH_CHECK(!numActiveTiles || *numActiveTiles >= 0, "Active tile count must be nonnegative");
    TORCH_CHECK(tilesPerCamera == 0 ||
                    numCameras <= std::numeric_limits<int64_t>::max() / tilesPerCamera,
                "Camera tile count overflows int64_t");
    const int64_t tileCount = numActiveTiles.value_or(numCameras * tilesPerCamera);
    DeviceLists deviceLists(numCameras);
    for (const auto deviceId: c10::irange(deviceCount)) {
        int64_t tileOffset, deviceTileCount;
        std::tie(tileOffset, deviceTileCount) =
            deviceAlignedChunk(tileAlignment, tileCount, deviceId, deviceCount);
        if (deviceTileCount == 0) {
            continue;
        }
        for (const auto cameraId: c10::irange(numCameras)) {
            const int64_t cameraBegin = cameraId * tilesPerCamera;
            const int64_t cameraEnd   = cameraBegin + tilesPerCamera;
            if (numActiveTiles ||
                (tileOffset < cameraEnd && cameraBegin < tileOffset + deviceTileCount)) {
                deviceLists[cameraId].push_back(deviceId);
            }
        }
    }
    return deviceLists;
}

template <bool Flatten>
void
reduceLocalGradients(const std::vector<torch::Tensor> &localGradients,
                     const DeviceLists &deviceLists) {
    if constexpr (Flatten) {
        reduceFlattenedGradients(localGradients);
    } else {
        reduceCameraGradients(localGradients, deviceLists);
    }
}

template <bool Flatten>
void
copyLocalGradients(const std::vector<torch::Tensor> &localGradients,
                   torch::Tensor outputGradient) {
    if constexpr (!Flatten) {
        TORCH_CHECK(outputGradient.is_contiguous(), "Camera gradient outputs must be contiguous");
    }
    if (outputGradient.numel() == 0) {
        return;
    }
    const int64_t partitionSize =
        Flatten ? localGradients.front().numel() : outputGradient.size(1);
    const size_t elementSize = outputGradient.element_size();
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        int64_t shardOffset, shardSize;
        std::tie(shardOffset, shardSize) = deviceChunk(partitionSize, deviceId);
        if (shardSize == 0) {
            continue;
        }

        const auto &gradient = localGradients[deviceId];
        TORCH_CHECK(gradient.scalar_type() == outputGradient.scalar_type(),
                    "Local and output gradients must have matching dtypes");
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        auto *output = static_cast<char *>(outputGradient.data_ptr());
        const auto *input = static_cast<const char *>(gradient.data_ptr());
        if constexpr (Flatten) {
            C10_CUDA_CHECK(cudaMemcpyAsync(
                output + shardOffset * elementSize,
                input + shardOffset * elementSize,
                shardSize * elementSize,
                cudaMemcpyDeviceToDevice,
                stream));
        } else {
            const size_t outputPitch = outputGradient.stride(0) * elementSize;
            const size_t inputPitch = gradient.stride(0) * elementSize;
            const size_t width = shardSize * outputGradient.stride(1) * elementSize;
            output += shardOffset * outputGradient.stride(1) * elementSize;
            input += shardOffset * gradient.stride(1) * elementSize;
            for (const auto cameraId: c10::irange(outputGradient.size(0))) {
                C10_CUDA_CHECK(cudaMemcpyAsync(output + cameraId * outputPitch,
                                              input + cameraId * inputPitch,
                                              width,
                                              cudaMemcpyDeviceToDevice,
                                              stream));
            }
        }
    }
}

template void
reduceLocalGradients<true>(const std::vector<torch::Tensor> &, const DeviceLists &);
template void
reduceLocalGradients<false>(const std::vector<torch::Tensor> &, const DeviceLists &);
template void
copyLocalGradients<true>(const std::vector<torch::Tensor> &, torch::Tensor);
template void
copyLocalGradients<false>(const std::vector<torch::Tensor> &, torch::Tensor);

} // namespace fvdb::detail
