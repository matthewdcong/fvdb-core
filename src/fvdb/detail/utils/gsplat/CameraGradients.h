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
        : mOutputs(outputs), mSegments(c10::cuda::device_count()) {
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
            for (const auto &segment: partitionCameraTiles(offset, count, tilesPerCamera)) {
                if (segment.shared) {
                    mSharedCameras[segment.cameraOffset].emplace_back(deviceId,
                                                                      mSegments[deviceId].size());
                }
                mSegments[deviceId].push_back({segment, {}});
            }
        }
    }

    // Called on each device after its prefetch stream has waited for preceding compute work.
    // The caller must wait for this stream before launching rasterization on the current stream.
    void
    prepare(c10::DeviceIndex deviceId, cudaStream_t prefetchStream) {
        const auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        std::vector<void *> prefetchPointers;
        std::vector<size_t> prefetchSizes;
        for (auto &segment: mSegments[deviceId]) {
            size_t rank      = 0;
            size_t rankCount = 1;
            if (segment.shared) {
                const auto &owners = mSharedCameras.at(segment.cameraOffset);
                const auto ownerIter =
                    std::find_if(owners.begin(), owners.end(), [deviceId](const auto &owner) {
                        return owner.first == deviceId;
                    });
                rank      = ownerIter - owners.begin();
                rankCount = owners.size();
            }
            for (const auto &output: mOutputs) {
                auto outputSlice = output.narrow(0, segment.cameraOffset, segment.cameraCount);
                segment.gradients.push_back(
                    segment.shared ? makeLocalGradient(outputSlice, deviceId, stream, rankCount)
                                   : outputSlice);
                // Shared cameras publish one shard per contributor after reduce-scatter. Prefetch
                // before rasterization can start writing fully owned cameras in the same outputs.
                const auto [elementOffset, elementCount] =
                    deviceAlignedChunk(1, outputSlice.numel(), rank, rankCount);
                if (elementCount > 0) {
                    prefetchPointers.push_back(static_cast<char *>(outputSlice.data_ptr()) +
                                               elementOffset * outputSlice.element_size());
                    prefetchSizes.push_back(elementCount * outputSlice.element_size());
                }
            }
        }
        memDiscardAndPrefetchBatchAsync(prefetchPointers, prefetchSizes, deviceId, prefetchStream);
        for (const auto &segment: mSegments[deviceId]) {
            if (!segment.shared) {
                perCameraMemsetAsync(
                    mOutputs, segment.cameraOffset, segment.cameraCount, 0, prefetchStream);
            }
        }
    }

    const std::vector<Segment> &
    segments(c10::DeviceIndex deviceId) const {
        return mSegments[deviceId];
    }

    // All raster launches must be queued first. Reduce shared cameras over their contributors,
    // then publish each contributor's shard after all reductions have been queued.
    // Release buffers after queuing their last use, before the caller merges compute streams.
    void
    finalize() {
        std::vector<std::pair<torch::Tensor, std::vector<torch::Tensor>>> reductions;
        for (const auto &[camera, owners]: mSharedCameras) {
            for (size_t i = 0; i < mOutputs.size(); ++i) {
                auto &[output, partials] = reductions.emplace_back(mOutputs[i].narrow(0, camera, 1),
                                                                   std::vector<torch::Tensor>{});
                for (const auto &[deviceId, segmentIndex]: owners) {
                    partials.push_back(mSegments[deviceId][segmentIndex].gradients[i]);
                }
                reduceGradientShards(partials);
            }
        }
        for (auto &[output, partials]: reductions) {
            copyGradientShards(partials, output);
        }
        mSegments.clear();
    }

  private:
    std::vector<torch::Tensor> mOutputs;
    std::vector<std::vector<Segment>> mSegments;
    std::map<uint32_t, std::vector<std::pair<c10::DeviceIndex, size_t>>> mSharedCameras;
};

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_GSPLAT_CAMERAGRADIENTS_H
