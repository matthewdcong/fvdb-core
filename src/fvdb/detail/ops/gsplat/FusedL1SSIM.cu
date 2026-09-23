// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/FusedImageLossKernels.cuh>
#include <fvdb/detail/ops/gsplat/FusedL1SSIM.h>
#include <fvdb/detail/utils/cuda/Prefetch.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/gsplat/ImagePartition.cuh>

#include <ATen/native/Lerp.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/cuda/nccl.h>
#include <torch/types.h>

#include <cub/cub.cuh>

#include <cooperative_groups.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace fvdb {

namespace detail {

namespace ops {

namespace {

namespace cg = cooperative_groups;

// ------------------------------------------
// Block Dimensions (shared with FusedSSIM)
// ------------------------------------------
#define BLOCK_X kImageBlockWidth
#define BLOCK_Y kImageBlockHeight

// ------------------------------------------
// Forward Kernel: Blend L1 and SSIM Maps
//  - Reads both per-pixel maps with the same spatial ownership.
//  - Applies lerp per pixel, then reduces over each tile and all channels.
//  - Writes one aggregate tile contribution normalized by the full image batch size.
// ------------------------------------------
__global__ void
fusedLerpKernel(int localToGlobalOffset,
                int B,
                int H,
                int W,
                int CH,
                float ssim_weight,
                const float *__restrict__ ssim_map,
                const float *__restrict__ l1_map,
                float *__restrict__ loss_tiles) {
    auto block = cg::this_thread_block();

    auto globalLinearGroupIndex = block.group_index().x + localToGlobalOffset;
    dim3 globalGroupDim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 globalGroupIndex(globalLinearGroupIndex % globalGroupDim.x,
                          (globalLinearGroupIndex / globalGroupDim.x) % globalGroupDim.y,
                          (globalLinearGroupIndex / (globalGroupDim.x * globalGroupDim.y)));

    const int bIdx            = globalGroupIndex.z;
    const int pix_y           = globalGroupIndex.y * BLOCK_Y + block.thread_index().y;
    const int pix_x           = globalGroupIndex.x * BLOCK_X + block.thread_index().x;
    const int64_t pix_id      = static_cast<int64_t>(pix_y) * W + pix_x;
    const int64_t num_pix     = static_cast<int64_t>(H) * W;
    const float normalization = 1.0f / (static_cast<int64_t>(B) * CH * num_pix);

    float loss = 0.0f;
    if (pix_y < H && pix_x < W) {
        for (int c = 0; c < CH; ++c) {
            const int64_t global_idx = (static_cast<int64_t>(bIdx) * CH + c) * num_pix + pix_id;
            const float l1_value     = l1_map[global_idx];
            const float ssim_value   = 1.0f - ssim_map[global_idx];
            loss += at::native::lerp(l1_value, ssim_value, ssim_weight) * normalization;
        }
    }

    // Padding contributes zero, but every thread participates in the reduction.
    using BlockReduce =
        cub::BlockReduce<float, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y>;
    __shared__ typename BlockReduce::TempStorage reduction_storage;
    const int tile        = block.group_index().x;
    const float loss_tile = BlockReduce(reduction_storage).Sum(loss);
    if (block.thread_rank() == 0) {
        loss_tiles[tile] = loss_tile;
    }
}

// ------------------------------------------
// Backward Kernel: Initialize SSIM Map Gradient
//  - Expands the upstream scalar with mean normalization in SSIM's spatial tiles.
// ------------------------------------------
__global__ void
fusedSSIMMapGradientKernel(int localToGlobalOffset,
                           int B,
                           int H,
                           int W,
                           int CH,
                           const float *__restrict__ grad_loss,
                           float *__restrict__ grad_map) {
    auto block = cg::this_thread_block();

    auto globalLinearGroupIndex = block.group_index().x + localToGlobalOffset;
    dim3 globalGroupDim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 globalGroupIndex(globalLinearGroupIndex % globalGroupDim.x,
                          (globalLinearGroupIndex / globalGroupDim.x) % globalGroupDim.y,
                          (globalLinearGroupIndex / (globalGroupDim.x * globalGroupDim.y)));

    const int bIdx        = globalGroupIndex.z;
    const int pix_y       = globalGroupIndex.y * BLOCK_Y + block.thread_index().y;
    const int pix_x       = globalGroupIndex.x * BLOCK_X + block.thread_index().x;
    const int64_t pix_id  = static_cast<int64_t>(pix_y) * W + pix_x;
    const int64_t num_pix = static_cast<int64_t>(H) * W;

    if (pix_y < H && pix_x < W) {
        const float gradient =
            *grad_loss / static_cast<float>(static_cast<int64_t>(B) * CH * num_pix);
        for (int c = 0; c < CH; ++c) {
            const int64_t global_idx = (static_cast<int64_t>(bIdx) * CH + c) * num_pix + pix_id;
            grad_map[global_idx]     = gradient;
        }
    }
}

// ------------------------------------------
// Backward Kernel: Weighted Gradient Sum
//  - Both existing backwards include the upstream scalar and mean normalization.
//  - Applies the blend weights and 1 - SSIM sign in the same spatial tiles.
//  - Reuses the L1 gradient buffer for the final image gradient.
// ------------------------------------------
__global__ void
fusedAddKernel(int localToGlobalOffset,
               int B,
               int H,
               int W,
               int CH,
               float ssim_weight,
               float *__restrict__ grad_l1,
               const float *__restrict__ grad_ssim) {
    auto block = cg::this_thread_block();

    auto globalLinearGroupIndex = block.group_index().x + localToGlobalOffset;
    dim3 globalGroupDim((W + BLOCK_X - 1) / BLOCK_X, (H + BLOCK_Y - 1) / BLOCK_Y, B);
    dim3 globalGroupIndex(globalLinearGroupIndex % globalGroupDim.x,
                          (globalLinearGroupIndex / globalGroupDim.x) % globalGroupDim.y,
                          (globalLinearGroupIndex / (globalGroupDim.x * globalGroupDim.y)));

    const int bIdx        = globalGroupIndex.z;
    const int pix_y       = globalGroupIndex.y * BLOCK_Y + block.thread_index().y;
    const int pix_x       = globalGroupIndex.x * BLOCK_X + block.thread_index().x;
    const int64_t pix_id  = static_cast<int64_t>(pix_y) * W + pix_x;
    const int64_t num_pix = static_cast<int64_t>(H) * W;

    if (pix_y < H && pix_x < W) {
        for (int c = 0; c < CH; ++c) {
            const int64_t global_idx = (static_cast<int64_t>(bIdx) * CH + c) * num_pix + pix_id;
            grad_l1[global_idx] =
                (1.0f - ssim_weight) * grad_l1[global_idx] - ssim_weight * grad_ssim[global_idx];
        }
    }
}

void
checkImages(const torch::Tensor &img1, const torch::Tensor &img2) {
    TORCH_CHECK_VALUE(img1.dim() == 4 && img2.dim() == 4,
                      "Fused L1/SSIM expects NCHW image batches");
    TORCH_CHECK_VALUE(img1.sizes() == img2.sizes(),
                      "Fused L1/SSIM requires identical image shapes");
    TORCH_CHECK_VALUE(img1.device() == img2.device(),
                      "Fused L1/SSIM requires images on the same device");
    TORCH_CHECK_VALUE(img1.scalar_type() == torch::kFloat && img2.scalar_type() == torch::kFloat,
                      "Fused L1/SSIM only supports float32 images");
    TORCH_CHECK_VALUE(img1.is_contiguous() && img2.is_contiguous(),
                      "Fused L1/SSIM expects contiguous NCHW storage");
    TORCH_CHECK_VALUE(img1.numel(), "Fused L1/SSIM requires nonempty image batches");
    // The existing SSIM kernels index the complete image batch with signed ints.
    TORCH_CHECK_VALUE(img1.numel() <= std::numeric_limits<int>::max(),
                      "Fused L1/SSIM image batch is too large");
    for (const auto size: img1.sizes()) {
        TORCH_CHECK_VALUE(size <= std::numeric_limits<int>::max() - BLOCK_Y,
                          "Fused L1/SSIM image dimension is too large");
    }
}

void
checkBackwardInputs(const torch::Tensor &img1,
                    const torch::Tensor &grad_loss,
                    const torch::TensorList &derivatives) {
    TORCH_CHECK_VALUE(
        grad_loss.dim() == 0 && grad_loss.device() == img1.device() &&
            grad_loss.scalar_type() == img1.scalar_type(),
        "Fused L1/SSIM requires a scalar loss gradient with the images' device and dtype");
    for (const auto &derivative: derivatives) {
        TORCH_CHECK_VALUE(
            derivative.sizes() == img1.sizes() && derivative.device() == img1.device() &&
                derivative.scalar_type() == img1.scalar_type() && derivative.is_contiguous(),
            "Fused L1/SSIM requires contiguous saved derivatives matching the images");
    }
}

// Prefetch all images and intermediates once per pass, on the same tile rows.
std::vector<cudaEvent_t>
prefetchImages(const torch::TensorList &images) {
    const auto &image = images.front();
    const int B       = image.size(0);
    const int CH      = image.size(1);
    const int H       = image.size(2);
    const int W       = image.size(3);
    std::vector<cudaEvent_t> events(c10::cuda::device_count());
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&events[deviceId], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getStreamFromPool(false, deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
        const auto chunk = imageBlockChunk(B, H, W, deviceId);
        if (chunk.blockCount) {
            std::vector<void *> pointers;
            std::vector<size_t> sizes;
            appendImagePrefetchRanges(
                pointers, sizes, images, chunk.tileRowOffset, chunk.tileRowCount, B, CH, H, W);
            memPrefetchBatchAsync(pointers, sizes, deviceId, stream);
        }
        C10_CUDA_CHECK(cudaEventRecord(events[deviceId], stream));
    }
    return events;
}

void
forwardTiles(const torch::Tensor &img1,
             const torch::Tensor &img2,
             const torch::Tensor &ssim_map,
             const torch::Tensor &l1_map,
             const torch::Tensor &dm_dmu1,
             const torch::Tensor &dm_dsigma1_sq,
             const torch::Tensor &dm_dsigma12,
             float ssim_weight,
             float C1,
             float C2,
             int blockOffset,
             int blockCount,
             float *loss,
             cudaStream_t stream) {
    if (!blockCount) {
        C10_CUDA_CHECK(cudaMemsetAsync(loss, 0, sizeof(float), stream));
        return;
    }
    const int B  = img1.size(0);
    const int CH = img1.size(1);
    const int H  = img1.size(2);
    const int W  = img1.size(3);
    // Keep tile contributions on the GPU that produced them.
    auto tile_losses = torch::empty(
        {blockCount}, img1.options().device(torch::kCUDA, c10::cuda::current_device()));
    auto *loss_tiles = tile_losses.data_ptr<float>();

    launchFusedSSIM(blockOffset,
                    blockCount,
                    B,
                    H,
                    W,
                    CH,
                    C1,
                    C2,
                    img1.const_data_ptr<float>(),
                    img2.const_data_ptr<float>(),
                    ssim_map.data_ptr<float>(),
                    dm_dmu1.numel() ? dm_dmu1.data_ptr<float>() : nullptr,
                    dm_dsigma1_sq.numel() ? dm_dsigma1_sq.data_ptr<float>() : nullptr,
                    dm_dsigma12.numel() ? dm_dsigma12.data_ptr<float>() : nullptr,
                    stream);
    launchFusedL1(blockOffset,
                  blockCount,
                  B,
                  H,
                  W,
                  CH,
                  img1.const_data_ptr<float>(),
                  img2.const_data_ptr<float>(),
                  l1_map.data_ptr<float>(),
                  stream);

    dim3 grid(blockCount);
    dim3 block(BLOCK_X, BLOCK_Y);
    fusedLerpKernel<<<grid, block, 0, stream>>>(blockOffset,
                                                B,
                                                H,
                                                W,
                                                CH,
                                                ssim_weight,
                                                ssim_map.const_data_ptr<float>(),
                                                l1_map.const_data_ptr<float>(),
                                                loss_tiles);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    FVDB_CUB_WRAPPER_ASYNC(stream, cub::DeviceReduce::Sum, loss_tiles, loss, blockCount);
}

void
initializeSSIMMapGradient(const torch::Tensor &grad_loss,
                          const torch::Tensor &grad_map,
                          int blockOffset,
                          int blockCount,
                          cudaStream_t stream) {
    if (!blockCount) {
        return;
    }
    dim3 grid(blockCount);
    dim3 block(BLOCK_X, BLOCK_Y);
    fusedSSIMMapGradientKernel<<<grid, block, 0, stream>>>(blockOffset,
                                                           grad_map.size(0),
                                                           grad_map.size(2),
                                                           grad_map.size(3),
                                                           grad_map.size(1),
                                                           grad_loss.const_data_ptr<float>(),
                                                           grad_map.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void
backwardTiles(const torch::Tensor &img1,
              const torch::Tensor &img2,
              const torch::Tensor &grad_loss,
              const torch::Tensor &grad_map,
              const torch::Tensor &dm_dmu1,
              const torch::Tensor &dm_dsigma1_sq,
              const torch::Tensor &dm_dsigma12,
              const torch::Tensor &grad_ssim,
              const torch::Tensor &grad_l1,
              float ssim_weight,
              float C1,
              float C2,
              int blockOffset,
              int blockCount,
              cudaStream_t stream) {
    if (!blockCount) {
        return;
    }
    const int B  = img1.size(0);
    const int CH = img1.size(1);
    const int H  = img1.size(2);
    const int W  = img1.size(3);
    launchFusedSSIMBackward(blockOffset,
                            blockCount,
                            B,
                            H,
                            W,
                            CH,
                            C1,
                            C2,
                            img1.const_data_ptr<float>(),
                            img2.const_data_ptr<float>(),
                            grad_map.const_data_ptr<float>(),
                            grad_ssim.data_ptr<float>(),
                            dm_dmu1.const_data_ptr<float>(),
                            dm_dsigma1_sq.const_data_ptr<float>(),
                            dm_dsigma12.const_data_ptr<float>(),
                            stream);
    launchFusedL1Backward(blockOffset,
                          blockCount,
                          B,
                          H,
                          W,
                          CH,
                          img1.const_data_ptr<float>(),
                          img2.const_data_ptr<float>(),
                          grad_loss.const_data_ptr<float>(),
                          grad_l1.data_ptr<float>(),
                          stream);

    dim3 grid(blockCount);
    dim3 block(BLOCK_X, BLOCK_Y);
    fusedAddKernel<<<grid, block, 0, stream>>>(blockOffset,
                                               B,
                                               H,
                                               W,
                                               CH,
                                               ssim_weight,
                                               grad_l1.data_ptr<float>(),
                                               grad_ssim.const_data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

// ------------------------------------------
// PyTorch Interfaces (Forward)
// ------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedL1SSIMCUDA(const torch::Tensor &img1,
                const torch::Tensor &img2,
                double ssim_weight,
                double C1,
                double C2,
                bool train) {
    checkImages(img1, img2);
    const c10::cuda::CUDAGuard device_guard(img1.device().index());
    const int B        = img1.size(0);
    const int H        = img1.size(2);
    const int W        = img1.size(3);
    auto ssim_map      = torch::empty_like(img1);
    auto l1_map        = torch::empty_like(img1);
    auto loss          = torch::empty({}, img1.options());
    auto dm_dmu1       = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());

    auto stream          = c10::cuda::getCurrentCUDAStream(img1.device().index());
    const int blockCount = B * ((H + BLOCK_Y - 1) / BLOCK_Y) * ((W + BLOCK_X - 1) / BLOCK_X);
    forwardTiles(img1,
                 img2,
                 ssim_map,
                 l1_map,
                 dm_dmu1,
                 dm_dsigma1_sq,
                 dm_dsigma12,
                 static_cast<float>(ssim_weight),
                 static_cast<float>(C1),
                 static_cast<float>(C2),
                 0,
                 blockCount,
                 loss.data_ptr<float>(),
                 stream);
    return std::make_tuple(loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
fusedL1SSIMPrivateUse1(const torch::Tensor &img1,
                       const torch::Tensor &img2,
                       double ssim_weight,
                       double C1,
                       double C2,
                       bool train) {
    checkImages(img1, img2);
    const int B        = img1.size(0);
    const int H        = img1.size(2);
    const int W        = img1.size(3);
    auto ssim_map      = torch::empty_like(img1);
    auto l1_map        = torch::empty_like(img1);
    auto loss          = torch::empty({}, img1.options());
    auto dm_dmu1       = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma1_sq = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());
    auto dm_dsigma12   = train ? torch::empty_like(img1) : torch::empty({0}, img1.options());

    std::vector<torch::Tensor> images = {img1, img2, ssim_map, l1_map};
    if (train) {
        images.insert(images.end(), {dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
    }
    auto events = prefetchImages(images);
    std::vector<torch::Tensor> partial_losses;
    partial_losses.reserve(c10::cuda::device_count());
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
        C10_CUDA_CHECK(cudaEventDestroy(events[deviceId]));
        const auto chunk = imageBlockChunk(B, H, W, deviceId);
        partial_losses.push_back(torch::empty({1}, img1.options().device(torch::kCUDA, deviceId)));
        forwardTiles(img1,
                     img2,
                     ssim_map,
                     l1_map,
                     dm_dmu1,
                     dm_dsigma1_sq,
                     dm_dsigma12,
                     static_cast<float>(ssim_weight),
                     static_cast<float>(C1),
                     static_cast<float>(C2),
                     chunk.blockOffset,
                     chunk.blockCount,
                     partial_losses.back().data_ptr<float>(),
                     stream);
    }
    auto nccl_loss =
        torch::from_blob(loss.data_ptr<float>(), {1}, partial_losses.front().options());
    torch::cuda::nccl::reduce(partial_losses, nccl_loss);
    mergeStreams();
    return std::make_tuple(loss, dm_dmu1, dm_dsigma1_sq, dm_dsigma12);
}

// ------------------------------------------
// PyTorch Interfaces (Backward)
// ------------------------------------------
torch::Tensor
fusedL1SSIMBackwardCUDA(const torch::Tensor &img1,
                        const torch::Tensor &img2,
                        const torch::Tensor &grad_loss,
                        const torch::Tensor &dm_dmu1,
                        const torch::Tensor &dm_dsigma1_sq,
                        const torch::Tensor &dm_dsigma12,
                        double ssim_weight,
                        double C1,
                        double C2) {
    checkImages(img1, img2);
    checkBackwardInputs(img1, grad_loss, {dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
    const c10::cuda::CUDAGuard device_guard(img1.device().index());
    const int B    = img1.size(0);
    const int H    = img1.size(2);
    const int W    = img1.size(3);
    auto grad_ssim = torch::empty_like(img1);
    auto grad_l1   = torch::empty_like(img1);
    auto grad_map  = torch::empty_like(img1);

    auto stream          = c10::cuda::getCurrentCUDAStream(img1.device().index());
    const int blockCount = B * ((H + BLOCK_Y - 1) / BLOCK_Y) * ((W + BLOCK_X - 1) / BLOCK_X);
    initializeSSIMMapGradient(grad_loss, grad_map, 0, blockCount, stream);
    backwardTiles(img1,
                  img2,
                  grad_loss,
                  grad_map,
                  dm_dmu1,
                  dm_dsigma1_sq,
                  dm_dsigma12,
                  grad_ssim,
                  grad_l1,
                  static_cast<float>(ssim_weight),
                  static_cast<float>(C1),
                  static_cast<float>(C2),
                  0,
                  blockCount,
                  stream);
    return grad_l1;
}

torch::Tensor
fusedL1SSIMBackwardPrivateUse1(const torch::Tensor &img1,
                               const torch::Tensor &img2,
                               const torch::Tensor &grad_loss,
                               const torch::Tensor &dm_dmu1,
                               const torch::Tensor &dm_dsigma1_sq,
                               const torch::Tensor &dm_dsigma12,
                               double ssim_weight,
                               double C1,
                               double C2) {
    checkImages(img1, img2);
    checkBackwardInputs(img1, grad_loss, {dm_dmu1, dm_dsigma1_sq, dm_dsigma12});
    const int B    = img1.size(0);
    const int H    = img1.size(2);
    const int W    = img1.size(3);
    auto grad_ssim = torch::empty_like(img1);
    auto grad_l1   = torch::empty_like(img1);
    auto grad_map  = torch::empty_like(img1);

    auto events = prefetchImages(
        {img1, img2, dm_dmu1, dm_dsigma1_sq, dm_dsigma12, grad_ssim, grad_l1, grad_map});
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, events[deviceId]));
        C10_CUDA_CHECK(cudaEventDestroy(events[deviceId]));
        const auto chunk = imageBlockChunk(B, H, W, deviceId);
        initializeSSIMMapGradient(grad_loss, grad_map, chunk.blockOffset, chunk.blockCount, stream);
    }
    // SSIM backward reads halo pixels from neighboring devices' gradient-map rows.
    // Finish initializing every row before any device starts those reads.
    mergeStreams();
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream      = c10::cuda::getCurrentCUDAStream(deviceId);
        const auto chunk = imageBlockChunk(B, H, W, deviceId);
        backwardTiles(img1,
                      img2,
                      grad_loss,
                      grad_map,
                      dm_dmu1,
                      dm_dsigma1_sq,
                      dm_dsigma12,
                      grad_ssim,
                      grad_l1,
                      static_cast<float>(ssim_weight),
                      static_cast<float>(C1),
                      static_cast<float>(C2),
                      chunk.blockOffset,
                      chunk.blockCount,
                      stream);
    }
    mergeStreams();
    return grad_l1;
}

} // namespace ops

} // namespace detail

} // namespace fvdb
