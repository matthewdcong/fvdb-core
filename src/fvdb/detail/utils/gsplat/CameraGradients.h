// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_GSPLAT_CAMERAGRADIENTS_H
#define FVDB_DETAIL_UTILS_GSPLAT_CAMERAGRADIENTS_H

#include <fvdb/detail/utils/cuda/GradientReduction.h>
#include <fvdb/detail/utils/cuda/Prefetch.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/gsplat/CameraTilePartition.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <utility>
#include <vector>

namespace fvdb::detail {

// Dense raster gradients follow the existing camera-major tile partition. A device's tile
// interval has at most two shared boundary cameras and one run of fully owned cameras.
// Only shared cameras need local buffers; fully owned cameras accumulate directly into outputs.
class CameraGradients {
  public:
    struct Segment : CameraTileSegment {
        std::vector<torch::Tensor> gradients;
    };

    CameraGradients(const std::vector<torch::Tensor> &outputs, uint32_t tilesPerCamera)
        : mOutputs(outputs), mSegments(c10::cuda::device_count()),
          mOutputPrefetchEvents(c10::cuda::device_count(), nullptr) {
        TORCH_CHECK(!outputs.empty() && tilesPerCamera > 0 && !mSegments.empty(),
                    "Camera gradients require outputs, tiles, and CUDA devices");
        const auto cameraCount   = outputs.front().size(0);
        const uint64_t tileCount = cameraCount * uint64_t{tilesPerCamera};
        TORCH_CHECK(tileCount <= std::numeric_limits<uint32_t>::max(),
                    "Too many tiles for camera gradients");
        for (const auto &output: outputs) {
            TORCH_CHECK(output.is_contiguous() && output.size(0) == cameraCount,
                        "Camera gradients require contiguous outputs with matching camera counts");
        }

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            const auto [offset, count] = deviceChunk(tileCount, deviceId);
            auto segments              = partitionCameraTiles(offset, count, tilesPerCamera);

            // Shared cameras can rasterize into local buffers while outputs are prefetched.
            std::partition(segments.begin(), segments.end(), [](const auto &segment) {
                return segment.shared;
            });

            for (const auto &segment: segments) {
                if (segment.shared) {
                    mSharedCameras[segment.cameraOffset].emplace_back(deviceId,
                                                                      mSegments[deviceId].size());
                }
                mSegments[deviceId].push_back({segment, {}});
            }
        }
    }

    CameraGradients(const CameraGradients &)            = delete;
    CameraGradients &operator=(const CameraGradients &) = delete;

    ~CameraGradients() {
        for (const auto event: mOutputPrefetchEvents) {
            if (event != nullptr) {
                C10_CUDA_CHECK_WARN(cudaEventDestroy(event));
            }
        }
    }

    // Initialize local gradients before the compute stream waits for input prefetching.
    void
    prepare(c10::DeviceIndex deviceId) {
        const auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        for (auto &segment: mSegments[deviceId]) {
            const size_t rankCount =
                segment.shared ? mSharedCameras.at(segment.cameraOffset).size() : 1;
            for (const auto &output: mOutputs) {
                const auto outputSlice =
                    output.narrow(0, segment.cameraOffset, segment.cameraCount);
                segment.gradients.push_back(
                    segment.shared ? makeLocalGradient(outputSlice, deviceId, stream, rankCount)
                                   : outputSlice);
            }
        }
    }

    // Call after recording input readiness on the prefetch stream. Queue output prefetches on
    // every device before launching rasterization; only output writers need to wait for them.
    void
    prefetchOutputs(c10::DeviceIndex deviceId, cudaStream_t prefetchStream) {
        std::vector<void *> prefetchPointers;
        std::vector<size_t> prefetchSizes;
        for (const auto &segment: mSegments[deviceId]) {
            size_t rank      = 0;
            size_t rankCount = 1;
            if (segment.shared) {
                // A camera's tile interval spans consecutive device chunks.
                const auto &owners = mSharedCameras.at(segment.cameraOffset);
                rank               = deviceId - owners.front().first;
                rankCount          = owners.size();
            }

            for (const auto &output: mOutputs) {
                const auto outputSlice =
                    output.narrow(0, segment.cameraOffset, segment.cameraCount);

                // Shared cameras publish one shard per contributor after reduce-scatter.
                const auto [elementOffset, elementCount] =
                    deviceAlignedChunk(1, outputSlice.numel(), rank, rankCount);
                if (elementCount > 0) {
                    prefetchPointers.emplace_back(static_cast<char *>(outputSlice.data_ptr()) +
                                                  elementOffset * outputSlice.element_size());
                    prefetchSizes.emplace_back(elementCount * outputSlice.element_size());
                }
            }
        }

        memDiscardAndPrefetchBatchAsync(prefetchPointers, prefetchSizes, deviceId, prefetchStream);
        C10_CUDA_CHECK(
            cudaEventCreateWithFlags(&mOutputPrefetchEvents[deviceId], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(mOutputPrefetchEvents[deviceId], prefetchStream));
    }

    const std::vector<Segment> &
    segments(c10::DeviceIndex deviceId) const {
        return mSegments[deviceId];
    }

    // Called before each segment's raster launch. Shared gradients were initialized in prepare();
    // fully owned gradients must wait for output prefetching before being zeroed.
    void
    prepareForRasterization(c10::DeviceIndex deviceId, const Segment &segment) const {
        if (segment.shared) {
            return;
        }

        waitForOutputs(deviceId);
        perCameraMemsetAsync(mOutputs,
                             segment.cameraOffset,
                             segment.cameraCount,
                             0,
                             c10::cuda::getCurrentCUDAStream(deviceId));
    }

    // Call after all raster launches. Queue all shared-camera reductions before waiting for
    // output prefetching and copying each contributor's shard.
    // Release buffers after queuing their last use, before the caller merges compute streams.
    void
    finalize() {
        std::vector<std::pair<torch::Tensor, std::vector<torch::Tensor>>> reductions;
        for (const auto &[camera, owners]: mSharedCameras) {
            for (const auto i: c10::irange(mOutputs.size())) {
                auto &[output, partials] = reductions.emplace_back(mOutputs[i].narrow(0, camera, 1),
                                                                   std::vector<torch::Tensor>{});
                for (const auto &[deviceId, segmentIndex]: owners) {
                    partials.push_back(mSegments[deviceId][segmentIndex].gradients[i]);
                }
                reduceGradientShards(partials);
            }
        }

        if (!reductions.empty()) {
            for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
                // Devices with a fully owned segment already waited before zeroing its outputs.
                if (!mSegments[deviceId].empty() && !hasOwnedCameras(deviceId)) {
                    waitForOutputs(deviceId);
                }
            }
        }

        for (auto &[output, partials]: reductions) {
            copyGradientShards(partials, output);
        }
        mSegments.clear();
    }

  private:
    bool
    hasOwnedCameras(c10::DeviceIndex deviceId) const {
        const auto &segments = mSegments[deviceId];
        return std::any_of(
            segments.begin(), segments.end(), [](const auto &segment) { return !segment.shared; });
    }

    void
    waitForOutputs(c10::DeviceIndex deviceId) const {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        const auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        // Adjacent camera slices and reduction shards can share pages. Finish every device's
        // discard/prefetch before any output writes, including zeroing fully owned gradients.
        for (const auto event: mOutputPrefetchEvents) {
            C10_CUDA_CHECK(cudaStreamWaitEvent(stream, event));
        }
    }

    std::vector<torch::Tensor> mOutputs;
    std::vector<std::vector<Segment>> mSegments;
    std::map<uint32_t, std::vector<std::pair<c10::DeviceIndex, size_t>>> mSharedCameras;
    std::vector<cudaEvent_t> mOutputPrefetchEvents;
};

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_GSPLAT_CAMERAGRADIENTS_H
