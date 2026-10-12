// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/utils/cuda/Utils.cuh>

#include <c10/util/CallOnce.h>

#include <vector>

namespace fvdb::detail {

namespace {

constexpr int kMergeDeviceId = 0;
c10::once_flag gMergeEventsInitFlag;
cudaEvent_t gMergeEvent = nullptr;
std::vector<cudaEvent_t> gPeerEvents;

void
initMergeEvents() {
    // Keep events for the process lifetime, as in the DGX allocator.
    C10_CUDA_CHECK(cudaSetDevice(kMergeDeviceId));
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&gMergeEvent, cudaEventDisableTiming));

    gPeerEvents.resize(c10::cuda::device_count());
    for (const auto deviceId: c10::irange(1, c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&gPeerEvents[deviceId], cudaEventDisableTiming));
    }
}

} // namespace

void
mergeStreams() {
    c10::call_once(gMergeEventsInitFlag, initMergeEvents);

    // Calls are issued from one host thread. CUDA waits capture the most recent recording,
    // so events can be reused on the next call while GPU work is pending.

    // Record peer streams; the merge stream's preceding work is already ordered.
    for (const auto deviceId: c10::irange(1, c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaEventRecord(gPeerEvents[deviceId], stream));
    }

    C10_CUDA_CHECK(cudaSetDevice(kMergeDeviceId));
    auto mergeStream = c10::cuda::getCurrentCUDAStream(kMergeDeviceId);
    for (const auto deviceId: c10::irange(1, c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaStreamWaitEvent(mergeStream, gPeerEvents[deviceId]));
    }
    C10_CUDA_CHECK(cudaEventRecord(gMergeEvent, mergeStream));

    // Only peers wait on the merge event; its own stream is already ordered.
    for (const auto deviceId: c10::irange(1, c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, gMergeEvent));
    }
}

} // namespace fvdb::detail
