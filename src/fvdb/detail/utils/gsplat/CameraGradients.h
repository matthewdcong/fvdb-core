// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_GSPLAT_CAMERAGRADIENTS_H
#define FVDB_DETAIL_UTILS_GSPLAT_CAMERAGRADIENTS_H

#include <fvdb/detail/utils/cuda/LocalGradient.h>
#include <fvdb/detail/utils/cuda/Prefetch.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <torch/csrc/cuda/nccl.h>

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
    struct Segment {
        uint32_t tileOffset;
        uint32_t tileCount;
        uint32_t cameraOffset;
        uint32_t cameraCount;
        bool shared;
        std::vector<torch::Tensor> gradients;
    };

    CameraGradients(const std::vector<torch::Tensor> &outputs, uint32_t tilesPerCamera)
        : mOutputs(outputs), mSegments(c10::cuda::device_count()) {
        TORCH_CHECK(!outputs.empty() && tilesPerCamera > 0 && !mSegments.empty(),
                    "Camera gradients require outputs, tiles, and CUDA devices");
        const auto cameraCount = outputs.front().size(0);
        const uint64_t tileCount = cameraCount * uint64_t{tilesPerCamera};
        TORCH_CHECK(tileCount <= std::numeric_limits<uint32_t>::max(),
                    "Too many tiles for camera gradients");
        for (const auto &output: outputs) {
            TORCH_CHECK(output.is_contiguous() && output.size(0) == cameraCount,
                        "Camera gradients require contiguous outputs with matching camera counts");
        }

        for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
            const auto [offset, count] = deviceChunk(tileCount, deviceId);
            const uint32_t end = offset + count;
            uint32_t begin = offset;
            while (begin < end) {
                const uint32_t camera = begin / tilesPerCamera;
                const uint32_t wholeCameras =
                    begin % tilesPerCamera == 0 ? (end - begin) / tilesPerCamera : 0;
                const bool shared = wholeCameras == 0;
                const uint32_t segmentEnd = shared
                                               ? std::min(end, (camera + 1) * tilesPerCamera)
                                               : begin + wholeCameras * tilesPerCamera;
                if (shared) {
                    mSharedCameras[camera].emplace_back(deviceId, mSegments[deviceId].size());
                }
                mSegments[deviceId].push_back({begin,
                                               segmentEnd - begin,
                                               camera,
                                               shared ? 1 : wholeCameras,
                                               shared,
                                               {}});
                begin = segmentEnd;
            }
        }
    }

    // Called on each device after its prefetch stream has waited for preceding compute work.
    // The caller must wait for this stream before launching rasterization on the current stream.
    void
    prepare(c10::DeviceIndex deviceId, cudaStream_t prefetchStream) {
        const auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        std::vector<void *> pointers;
        std::vector<size_t> sizes;
        for (auto &segment: mSegments[deviceId]) {
            for (const auto &output: mOutputs) {
                auto slice = output.narrow(0, segment.cameraOffset, segment.cameraCount);
                segment.gradients.push_back(segment.shared
                                                ? makeLocalGradient(slice, deviceId, stream)
                                                : slice);
            }
            // The first contributor publishes a shared camera after its reduction. Prefetch that
            // destination now as well, before any raster kernel starts writing adjacent cameras.
            if (!segment.shared || mSharedCameras.at(segment.cameraOffset).front().first == deviceId) {
                appendPerCameraPrefetchRanges(
                    pointers, sizes, mOutputs, segment.cameraOffset, segment.cameraCount);
            }
        }
        memDiscardAndPrefetchBatchAsync(pointers, sizes, deviceId, prefetchStream);
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

    // All raster launches must be queued first. NCCL uses only the contributing devices, in
    // device order, and reduces into the first contributor's local buffer on its current stream.
    // Release buffers after queuing their last use, before the caller merges compute streams.
    void
    finalize() {
        for (const auto &[camera, owners]: mSharedCameras) {
            for (size_t i = 0; i < mOutputs.size(); ++i) {
                auto output = mOutputs[i].narrow(0, camera, 1);
                if (output.numel() == 0) {
                    continue;
                }
                std::vector<torch::Tensor> partials;
                for (const auto &[deviceId, segmentIndex]: owners) {
                    partials.push_back(mSegments[deviceId][segmentIndex].gradients[i]);
                }
                torch::cuda::nccl::reduce(partials);
                const auto deviceId = owners.front().first;
                C10_CUDA_CHECK(cudaSetDevice(deviceId));
                C10_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(),
                                               partials.front().data_ptr(),
                                               output.numel() * output.element_size(),
                                               cudaMemcpyDeviceToDevice,
                                               c10::cuda::getCurrentCUDAStream(deviceId)));
            }
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
