// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/cuda/GradientReduction.h>
#include <fvdb/detail/utils/cuda/LocalGradient.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <torch/csrc/cuda/nccl.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>

namespace fvdb::detail {
namespace {

void
reduceFlattenedGradients(const std::vector<torch::Tensor> &localGradients) {
    const int64_t numElements = localGradients.front().numel();
    if (numElements == 0) {
        return;
    }

    const int64_t shardSize = localGradientShardSize(numElements, c10::cuda::device_count());
    const int64_t paddedNumElements = shardSize * c10::cuda::device_count();
    std::vector<torch::Tensor> paddedGradients(c10::cuda::device_count());
    std::vector<torch::Tensor> reducedShards(c10::cuda::device_count());
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
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

struct GradientTransfer {
    c10::DeviceIndex source;
    c10::DeviceIndex destination;
    torch::Tensor input;
    torch::Tensor output;
};

struct GradientAccumulation {
    c10::DeviceIndex deviceId;
    torch::Tensor output;
    torch::Tensor received;
};

void
reduceCameraGradients(const std::vector<torch::Tensor> &localGradients,
                      const DeviceLists &deviceLists) {
    TORCH_CHECK(static_cast<int64_t>(localGradients.size()) == c10::cuda::device_count() &&
                    !localGradients.empty(),
                "Camera gradients require one local buffer per CUDA device");
    const auto &first = localGradients.front();
    TORCH_CHECK(first.dim() >= 2 && static_cast<int64_t>(deviceLists.size()) == first.size(0),
                "Camera contributors must match the [C, N, ...] gradient shape");
    for (const auto &sources: deviceLists) {
        c10::DeviceIndex previous = -1;
        for (const auto source: sources) {
            TORCH_CHECK(source > previous && source < c10::cuda::device_count(),
                        "Camera contributors must be unique, sorted CUDA device IDs");
            previous = source;
        }
    }
    if (first.numel() == 0) {
        return;
    }

    const int64_t numGaussians = first.size(1);
    const int64_t elementsPerGaussian = first.numel() / first.size(0) / numGaussians;
    const int64_t shardSize = localGradientShardSize(numGaussians, c10::cuda::device_count()) *
                              elementsPerGaussian;
    const int64_t cameraSize = shardSize * c10::cuda::device_count();
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
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
    std::vector<GradientTransfer> transfers;
    std::vector<GradientAccumulation> accumulations;

    // Prepare all point-to-point transfers and receive buffers before entering the NCCL group.
    // Cameras with contributions on every device use reduce-scatter directly.
    for (const auto cameraId: c10::irange(first.size(0))) {
        const auto &sources = deviceLists[cameraId];
        if (sources.empty()) {
            continue; // The zero-initialized local slices already hold this camera's result.
        }
        if (static_cast<int64_t>(sources.size()) == c10::cuda::device_count()) {
            std::vector<torch::Tensor> inputs(c10::cuda::device_count());
            std::vector<torch::Tensor> outputs(c10::cuda::device_count());
            for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
                inputs[deviceId] = localGradients[deviceId].as_strided(
                    {cameraSize}, {1}, cameraId * cameraSize);
                outputs[deviceId] = inputs[deviceId].narrow(0, deviceId * shardSize, shardSize);
            }
            torch::cuda::nccl::reduce_scatter(inputs, outputs);
            continue;
        }

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            int64_t gaussianOffset, gaussianCount;
            std::tie(gaussianOffset, gaussianCount) = deviceChunk(numGaussians, deviceId);
            if (gaussianCount == 0) {
                continue;
            }
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            auto output = localGradients[deviceId]
                              .select(0, cameraId)
                              .narrow(0, gaussianOffset, gaussianCount)
                              .view({-1});
            auto received = torch::empty(
                {static_cast<int64_t>(sources.size()) - 1, output.numel()},
                localGradients[deviceId].options());

            // Start with this device's own contribution when present; otherwise receive the
            // first source directly into output. Every other contribution goes into scratch.
            const auto initialSource =
                std::binary_search(sources.begin(), sources.end(), deviceId) ? deviceId
                                                                            : sources.front();
            int64_t receiveIndex = 0;
            for (const auto source: sources) {
                if (source == deviceId) {
                    continue; // The local contribution is already in output.
                }
                auto input = localGradients[source]
                                 .select(0, cameraId)
                                 .narrow(0, gaussianOffset, gaussianCount)
                                 .view({-1});
                auto receiveBuffer = source == initialSource
                                         ? output
                                         : received.select(0, receiveIndex++);
                transfers.push_back(
                    {source, deviceId, std::move(input), std::move(receiveBuffer)});
            }
            accumulations.push_back({deviceId, std::move(output), std::move(received)});
        }
    }

    if (transfers.empty()) {
        return;
    }

    {
        // Group every camera's point-to-point transfers for this gradient tensor so different
        // camera owners can send concurrently instead of launching one scatter per camera.
        torch::cuda::nccl::AutoNcclGroup group;
        for (auto &transfer: transfers) {
            C10_CUDA_CHECK(cudaSetDevice(transfer.source));
            auto stream = c10::cuda::getCurrentCUDAStream(transfer.source);
            torch::cuda::nccl::send(
                transfer.input, comms[transfer.source], stream, transfer.destination);
            C10_CUDA_CHECK(cudaSetDevice(transfer.destination));
            stream = c10::cuda::getCurrentCUDAStream(transfer.destination);
            torch::cuda::nccl::recv(
                transfer.output, comms[transfer.destination], stream, transfer.source);
        }
    }

    // Accumulate on each destination's current stream after NCCL has enqueued every transfer.
    // Keep all buffers alive until these additions have also been enqueued.
    for (auto &accumulation: accumulations) {
        C10_CUDA_CHECK(cudaSetDevice(accumulation.deviceId));
        for (const auto receiveIndex: c10::irange(accumulation.received.size(0))) {
            accumulation.output.add_(accumulation.received.select(0, receiveIndex));
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
