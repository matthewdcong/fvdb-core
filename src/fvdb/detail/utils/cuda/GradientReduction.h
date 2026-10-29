// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
#define FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H

#include <torch/types.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace fvdb::detail {

// Lists of unique, sorted local CUDA device IDs, which also serve as NCCL ranks;
// the caller determines what each list represents.
using DeviceLists = std::vector<std::vector<c10::DeviceIndex>>;

// Match rasterization's aligned, flattened tile partition. Sparse camera membership is not
// available on the host: conservatively include every device with assigned active tiles.
DeviceLists
makeDeviceLists(int64_t numCameras,
                   int64_t tilesPerCamera,
                   int64_t tileAlignment,
                   int64_t deviceCount,
                   std::optional<int64_t> numActiveTiles = std::nullopt);

// Reduce in place into each device's owned slice of its local gradient buffer.
// Inputs must come from makeLocalGradient<Flatten>(). Flatten uses deviceChunk(numel, rank);
// otherwise each camera uses deviceChunk(N, rank) and requires its sorted contributor list.
// Per-camera reduction uses NCCL reduce-scatter for all-device contributors and grouped send/recv
// otherwise. Noncontributors must have zero local gradients for that camera.
// Uses current CUDA streams; keep them unchanged until the local buffers have been released.
// Calls sharing PyTorch's NCCL communicator cache must be serialized on the host.
template <bool Flatten = true>
void
reduceLocalGradients(const std::vector<torch::Tensor> &localGradients,
                     const DeviceLists &deviceLists = {});

// Publish each device's reduced slices using the same Flatten value as makeLocalGradient().
// Call after queuing reductions and waiting for output discard/prefetch on the current streams.
// Keep local gradient buffers alive until all output copies have been queued.
template <bool Flatten = true>
void
copyLocalGradients(const std::vector<torch::Tensor> &localGradients,
                   torch::Tensor outputGradient);

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_CUDA_GRADIENTREDUCTION_H
