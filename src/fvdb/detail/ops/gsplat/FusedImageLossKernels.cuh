// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_GSPLAT_FUSEDIMAGELOSSKERNELS_CUH
#define FVDB_DETAIL_OPS_GSPLAT_FUSEDIMAGELOSSKERNELS_CUH

#include <cuda_runtime_api.h>

namespace fvdb {
namespace detail {
namespace ops {

// Launch the existing kernels into caller-owned storage on the supplied stream.
// Offsets and counts are measured in 16x16 image tiles. Callers skip empty chunks.
// L1 writes an unreduced NCHW map of absolute differences.
void launchFusedL1(int blockOffset,
                   int blockCount,
                   int B,
                   int H,
                   int W,
                   int CH,
                   const float *img1,
                   const float *img2,
                   float *l1_map,
                   cudaStream_t stream);

void launchFusedL1Backward(int blockOffset,
                           int blockCount,
                           int B,
                           int H,
                           int W,
                           int CH,
                           const float *img1,
                           const float *img2,
                           const float *grad_loss,
                           float *grad_img1,
                           cudaStream_t stream);

void launchFusedSSIM(int blockOffset,
                     int blockCount,
                     int B,
                     int H,
                     int W,
                     int CH,
                     float C1,
                     float C2,
                     const float *img1,
                     const float *img2,
                     float *ssim_map,
                     float *dm_dmu1,
                     float *dm_dsigma1_sq,
                     float *dm_dsigma12,
                     cudaStream_t stream);

// Use the existing backward kernel with a caller-provided per-pixel gradient map.
void launchFusedSSIMBackward(int blockOffset,
                             int blockCount,
                             int B,
                             int H,
                             int W,
                             int CH,
                             float C1,
                             float C2,
                             const float *img1,
                             const float *img2,
                             const float *grad_map,
                             float *grad_img1,
                             const float *dm_dmu1,
                             const float *dm_dsigma1_sq,
                             const float *dm_dsigma12,
                             cudaStream_t stream);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_GSPLAT_FUSEDIMAGELOSSKERNELS_CUH
