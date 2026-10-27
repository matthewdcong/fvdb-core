// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/gsplat/RasterizeScreenSpaceGaussiansBackward.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/Nvtx.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/GridDim.h>
#include <fvdb/detail/utils/cuda/Prefetch.h>
#include <fvdb/detail/utils/cuda/Utils.cuh>
#include <fvdb/detail/utils/cuda/WarpReduce.cuh>
#include <fvdb/detail/utils/gsplat/Gaussian2D.cuh>
#include <fvdb/detail/utils/gsplat/GaussianRasterize.cuh>

#include <ATen/EmptyTensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/ops/from_blob.h>
#include <c10/core/DeviceType.h>
#include <c10/cuda/CUDAGuard.h>

#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda/cmath>
#include <thrust/iterator/transform_iterator.h>

#include <cooperative_groups.h>

#include <algorithm>
#include <limits>

namespace fvdb::detail::ops {
namespace {

// All threads in the tile participate, with inactive pixels contributing -1. Reuse each warp's
// maximum so only one thread per warp updates the tile-wide cutoff.
template <typename WarpT>
inline __device__ int32_t
tileLastIntersectionOffset(const int32_t lastIntersectionOffsetInWarp, const WarpT &warp) {
    __shared__ int32_t lastIntersectionOffsetInTile;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        lastIntersectionOffsetInTile = -1;
    }
    __syncthreads();
    if (warp.thread_rank() == 0) {
        atomicMax(&lastIntersectionOffsetInTile, lastIntersectionOffsetInWarp);
    }
    __syncthreads();
    return lastIntersectionOffsetInTile;
}

template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
struct RasterizeBackwardArgs {
    using CommonArgs = RasterizeCommonArgs<ScalarType, NUM_CHANNELS, IS_PACKED>;
    CommonArgs commonArgs;

    using vec2t          = typename CommonArgs::vec2t;
    using vec3t          = typename CommonArgs::vec3t;
    using VectorAccessor = typename CommonArgs::VectorAccessor;
    using ScalarAccessor = typename CommonArgs::ScalarAccessor;

    constexpr static bool IS_CHUNKED = (NUM_CHANNELS != NUM_SHARED_CHANNELS);

    // Preserve full unrolling for small feature vectors while bounding compiler work for the
    // 512/513-channel kernels, as in the world-space rasterizer.
    enum : size_t {
        CHANNEL_UNROLL = NUM_CHANNELS <= 32 ? NUM_CHANNELS : 4,
        SHARED_CHANNEL_UNROLL =
            NUM_CHANNELS <= 32 || NUM_SHARED_CHANNELS <= 32 ? NUM_SHARED_CHANNELS : 4
    };

    bool mAbsGrad;

    // Pixel inputs are flattened across cameras. P is C * H * W in dense mode, or the total
    // number of active pixels in sparse mode.
    fvdb::TorchRAcc64<ScalarType, 2> mRenderedAlphas;         // [P, 1]
    fvdb::TorchRAcc64<int32_t, 1> mLastGaussianIds;           // [P]
    fvdb::TorchRAcc64<ScalarType, 2> mDLossDRenderedFeatures; // [P, NUM_CHANNELS]
    fvdb::TorchRAcc64<ScalarType, 2> mDLossDRenderedAlphas;   // [P, 1]

    VectorAccessor mOutDLossDMeans2dAbs;                      // [C, N, 2] or [nnz, 2]
    VectorAccessor mOutDLossDMeans2d;                         // [C, N, 2] or [nnz, 2]
    VectorAccessor mOutDLossDConics;                          // [C, N, 3] or [nnz, 3]
    VectorAccessor mOutDLossDFeatures;  // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
    ScalarAccessor mOutDLossDOpacities; // [C, N] or [nnz]

    // Intersection lists retain global IDs. When present, this map converts them to the local
    // packed indices shared by the input tensors and gradient outputs.
    const int32_t *mGlobalToLocal = nullptr;

    RasterizeBackwardArgs(
        const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
        const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
        const torch::Tensor &opacities, // [C, N] or [nnz]
        const torch::Tensor &features,  // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const at::optional<torch::Tensor> &backgrounds, // [C, NUM_CHANNELS]
        const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
        const RenderWindow2D &renderWindow,
        const uint32_t tileSize,
        const uint32_t blockOffset,
        const torch::Tensor &tileOffsets,            // [C, numTilesH, numTilesW]
        const torch::Tensor &tileGaussianIds,        // [totalIntersections]
        const torch::Tensor &renderedAlphas,         // [P, 1]
        const torch::Tensor &lastGaussianIds,        // [P]
        const torch::Tensor &dLossDRenderedFeatures, // [P, NUM_CHANNELS]
        const torch::Tensor &dLossDRenderedAlphas,   // [P, 1]
        const torch::Tensor &outDLossDMeans2d,       // [C, N, 2] or [nnz, 2]
        const torch::Tensor &outDLossDConics,        // [C, N, 3] or [nnz, 3]
        const torch::Tensor &outDLossDFeatures,      // [C, N, NUM_CHANNELS] or [nnz, NUM_CHANNELS]
        const torch::Tensor &outDLossDOpacities,     // [C, N] or [nnz]
        const std::optional<torch::Tensor> &outDLossDMeans2dAbs,        // [C, N, 2] or [nnz, 2]
        const std::optional<torch::Tensor> &activeTiles = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &tilePixelMask =
            std::nullopt, // [AT, wordsPerTileBitmask] e.g. [AT, 4]
        const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt, // [AT]
        const std::optional<torch::Tensor> &pixelMap        = std::nullopt)        // [AP]
        : commonArgs(means2d,
                     conics,
                     opacities,
                     features,
                     backgrounds,
                     masks,
                     renderWindow,
                     tileSize,
                     blockOffset,
                     tileOffsets,
                     tileGaussianIds,
                     activeTiles,
                     tilePixelMask,
                     tilePixelCumsum,
                     pixelMap),
          mAbsGrad(outDLossDMeans2dAbs.has_value()),
          mRenderedAlphas(initAccessor<ScalarType, 2>(renderedAlphas, "renderedAlphas")),
          mLastGaussianIds(initAccessor<int32_t, 1>(lastGaussianIds, "lastGaussianIds")),
          mDLossDRenderedFeatures(
              initAccessor<ScalarType, 2>(dLossDRenderedFeatures, "dLossDRenderedFeatures")),
          mDLossDRenderedAlphas(
              initAccessor<ScalarType, 2>(dLossDRenderedAlphas, "dLossDRenderedAlphas")),
          mOutDLossDMeans2dAbs(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS + 1>(
              outDLossDMeans2dAbs, means2d.options(), "outDLossDMeans2dAbs")),
          mOutDLossDMeans2d(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS + 1>(
              outDLossDMeans2d, "outDLossDMeans2d")),
          mOutDLossDFeatures(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS + 1>(
              outDLossDFeatures, "outDLossDFeatures")),
          mOutDLossDConics(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS + 1>(
              outDLossDConics, "outDLossDConics")),
          mOutDLossDOpacities(initAccessor<ScalarType, CommonArgs::NUM_OUTER_DIMS>(
              outDLossDOpacities, "outDLossDOpacities")) {
        checkInputShapes();
    }

    // Check that the input tensor shapes are valid
    void
    checkInputShapes() {
        const int64_t totalGaussians = IS_PACKED ? commonArgs.mMeans2d.size(0) : 0;

        TORCH_CHECK_VALUE(2 == mOutDLossDMeans2d.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDMeans2d");
        TORCH_CHECK_VALUE(3 == mOutDLossDConics.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDConics");
        TORCH_CHECK_VALUE(NUM_CHANNELS == mOutDLossDFeatures.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDFeatures");
        TORCH_CHECK_VALUE(2 == mOutDLossDMeans2d.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDMeans2d");
        TORCH_CHECK_VALUE(3 == mOutDLossDConics.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDConics");
        TORCH_CHECK_VALUE(NUM_CHANNELS == mOutDLossDFeatures.size(CommonArgs::NUM_OUTER_DIMS),
                          "Bad size for outDLossDFeatures");

        if constexpr (IS_PACKED) {
            if (mAbsGrad) {
                TORCH_CHECK_VALUE(totalGaussians == mOutDLossDMeans2dAbs.size(0),
                                  "Bad size for outDLossDMeans2dAbs");
                TORCH_CHECK_VALUE(2 == mOutDLossDMeans2dAbs.size(CommonArgs::NUM_OUTER_DIMS),
                                  "Bad size for outDLossDMeans2dAbs");
            }
            TORCH_CHECK_VALUE(totalGaussians == mOutDLossDMeans2d.size(0),
                              "Bad size for outDLossDMeans2d");
            TORCH_CHECK_VALUE(totalGaussians == mOutDLossDConics.size(0),
                              "Bad size for outDLossDConics");
            TORCH_CHECK_VALUE(totalGaussians == mOutDLossDFeatures.size(0),
                              "Bad size for outDLossDFeatures");
            TORCH_CHECK_VALUE(totalGaussians == mOutDLossDOpacities.size(0),
                              "Bad size for outDLossDOpacities");
        } else {
            if (mAbsGrad) {
                TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDMeans2dAbs.size(0),
                                  "Bad size for outDLossDMeans2dAbs");
                TORCH_CHECK_VALUE(2 == mOutDLossDMeans2dAbs.size(CommonArgs::NUM_OUTER_DIMS),
                                  "Bad size for outDLossDMeans2dAbs");
            }

            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDMeans2d.size(0),
                              "Bad size for outDLossDMeans2d");
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDConics.size(0),
                              "Bad size for outDLossDConics");
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDFeatures.size(0),
                              "Bad size for outDLossDFeatures");
            TORCH_CHECK_VALUE(commonArgs.mNumCameras == mOutDLossDOpacities.size(0),
                              "Bad size for outDLossDOpacities");
            TORCH_CHECK_VALUE(commonArgs.mNumGaussiansPerCamera == mOutDLossDOpacities.size(1),
                              "Bad size for outDLossDOpacities");
        }

        const int64_t numPixels =
            commonArgs.mIsSparse
                ? commonArgs.mPixelMap.size(0)
                : commonArgs.mNumCameras * commonArgs.renderHeight() * commonArgs.renderWidth();
        TORCH_CHECK_VALUE(numPixels == mRenderedAlphas.size(0), "Bad size for renderedAlphas");
        TORCH_CHECK_VALUE(1 == mRenderedAlphas.size(1), "Bad size for renderedAlphas");
        TORCH_CHECK_VALUE(numPixels == mLastGaussianIds.size(0), "Bad size for lastGaussianIds");
        TORCH_CHECK_VALUE(numPixels == mDLossDRenderedFeatures.size(0),
                          "Bad size for dLossDRenderedFeatures");
        TORCH_CHECK_VALUE(NUM_CHANNELS == mDLossDRenderedFeatures.size(1),
                          "Bad size for dLossDRenderedFeatures");
        TORCH_CHECK_VALUE(numPixels == mDLossDRenderedAlphas.size(0),
                          "Bad size for dLossDRenderedAlphas");
        TORCH_CHECK_VALUE(1 == mDLossDRenderedAlphas.size(1), "Bad size for dLossDRenderedAlphas");
    }

    /// @brief Read the alpha value for a pixel
    /// @param pixelIndex The index of the pixel
    /// @return The alpha value for the pixel
    __device__ ScalarType
    readAlpha(uint64_t pixelIndex) {
        return mRenderedAlphas[pixelIndex][0];
    }

    /// @brief Read the last ID for a pixel
    /// @param pixelIndex The index of the pixel
    /// @return The last ID for the pixel
    __device__ int32_t
    readLastId(uint64_t pixelIndex) {
        return mLastGaussianIds[pixelIndex];
    }

    /// @brief Read the gradient of the loss with respect to the rendered alpha for a pixel
    /// @param pixelIndex The index of the pixel
    /// @return The gradient of the loss with respect to the rendered alpha for the pixel
    __device__ ScalarType
    readDLossDRenderedAlpha(uint64_t pixelIndex) {
        return mDLossDRenderedAlphas[pixelIndex][0];
    }

    /// @brief Read the gradient of the loss with respect to the rendered features for a pixel
    /// @param pixelIndex The index of the pixel
    /// @return The gradient of the loss with respect to the rendered features for the pixel
    __device__ ScalarType
    readDLossDRenderedFeature(uint64_t pixelIndex, uint32_t channel) {
        return mDLossDRenderedFeatures[pixelIndex][channel];
    }

    // g is a local Gaussian ID when mGlobalToLocal is present, otherwise a global Gaussian ID.
    inline __device__ void
    fetchGaussianFeatureIntoSharedMemory(const int32_t g,
                                         const size_t channelStart,
                                         const size_t numChannels,
                                         ScalarType *outFeatures) {
        if constexpr (IS_PACKED) {
            const auto featureAccessor = commonArgs.mFeatures[g];
            for (uint32_t k = 0; k < numChannels; ++k) {
                outFeatures[k] = featureAccessor[k + channelStart];
            }
        } else {
            // colors: [C, N, NUM_CHANNELS]
            // colors[c, n, k] = [c * N * NUM_CHANNELS + n * NUM_CHANNELS + k]
            // g = c * N + n
            const int32_t cid          = g / commonArgs.mNumGaussiansPerCamera;
            const int32_t gid          = g % commonArgs.mNumGaussiansPerCamera;
            const auto featureAccessor = commonArgs.mFeatures[cid][gid];
            if constexpr (IS_CHUNKED) {
                for (auto k = 0; k < numChannels; ++k) {
                    outFeatures[k] = featureAccessor[k + channelStart];
                }
            } else {
#pragma unroll CHANNEL_UNROLL
                for (auto k = 0; k < NUM_CHANNELS; ++k) {
                    outFeatures[k] = featureAccessor[k];
                }
            }
        }
    }

    /// @brief Get a pointer to per-Gaussian data from a tensor accessor (used for device atomics)
    ///
    /// This assumes that the tensor accessor dimension is at least 2.
    /// @tparam T The type of the data to access
    /// @tparam Accessor The type of the accessor
    /// @param accessor The accessor to the tensor
    /// @param g The index of the Gaussian
    /// @return A pointer to the data for the Gaussian
    template <typename T, typename Accessor>
    inline __device__ T *
    getAccessorPointer(const Accessor &accessor, const int32_t g) const {
        if constexpr (IS_PACKED) {
            return reinterpret_cast<T *>(accessor[g].data());
        } else {
            auto cid = g / commonArgs.mNumGaussiansPerCamera;
            auto gid = g % commonArgs.mNumGaussiansPerCamera;
            return reinterpret_cast<T *>(accessor[cid][gid].data());
        }
    }

    /// @brief Get a pointer to per-Gaussian data from a 1D tensor accessor (used for device
    /// atomics)
    /// @tparam T The type of the data to access
    /// @tparam Accessor The type of the accessor
    /// @param accessor The accessor to the tensor
    /// @param g The index of the Gaussian
    /// @return A pointer to the data for the Gaussian
    template <typename T, typename Accessor>
    inline __device__ T *
    get1DAccessorPointer(const Accessor &accessor, const int32_t g) const {
        // Because GenericPackedTensorAccessor<...,N=1,...>::operator[] returns a reference,
        // we can't take the address of the specific offset. So in the 1D accessor case, we
        // have to use some pointer arithmetic to get the address of the specific offset.
        if constexpr (IS_PACKED) {
            return reinterpret_cast<T *>(accessor.data()) + g;
        } else {
            auto cid = g / commonArgs.mNumGaussiansPerCamera;
            auto gid = g % commonArgs.mNumGaussiansPerCamera;
            return reinterpret_cast<T *>(accessor[cid].data()) + gid;
        }
    }

    /// @brief Atomically accumulate the gradient contribution from this pixel to the gradient of
    /// the loss with respect to the features of this Gaussian
    /// @param g The index of the Gaussian
    /// @param featureGradientContribution The gradient contribution from this pixel to the gradient
    /// for the features of this Gaussian
    /// @param channelStart The starting channel index for the features
    /// @param numChannels The number of channels to accumulate
    inline __device__ void
    atomicAddFeatureGradientContributions(const int32_t g,
                                          const ScalarType *featureGradientContribution,
                                          const size_t channelStart,
                                          const size_t numChannels) {
        auto dLossDFeaturesGaussianPtr = getAccessorPointer<ScalarType>(mOutDLossDFeatures, g);
        if constexpr (IS_CHUNKED) {
            for (uint32_t k = 0; k < numChannels; ++k) {
                atomicAdd(dLossDFeaturesGaussianPtr + channelStart + k,
                          featureGradientContribution[k]);
            }
        } else {
#pragma unroll CHANNEL_UNROLL
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                atomicAdd(dLossDFeaturesGaussianPtr + k, featureGradientContribution[k]);
            }
        }
    }

    /// @brief Atomically accumulate the gradient contribution from this pixel to the gradient of
    /// the loss with respect to the means, conics, and opacities of this Gaussian
    /// @param g The index of the Gaussian
    /// @param pixelConicGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the conic of this Gaussian
    /// @param pixelMean2dGradientContribution The gradient contribution from this pixel to the
    inline __device__ void
    atomicAddMeans2dConicsAndOpacitiesGradientContributions(
        const int32_t g,
        const vec3t &pixelConicGradientContribution,
        const vec2t &pixelMean2dGradientContribution,
        const vec2t &pixelMean2dAbsGradientContribution,
        const ScalarType pixelOpacityGradientContribution) {
        auto *dLossDConicsGaussianPtr = getAccessorPointer<vec3t>(mOutDLossDConics, g);
        atomicAdd(&dLossDConicsGaussianPtr->operator[](0), pixelConicGradientContribution[0]);
        atomicAdd(&dLossDConicsGaussianPtr->operator[](1), pixelConicGradientContribution[1]);
        atomicAdd(&dLossDConicsGaussianPtr->operator[](2), pixelConicGradientContribution[2]);

        auto *dLossDMeans2dGaussianPtr = getAccessorPointer<vec2t>(mOutDLossDMeans2d, g);
        atomicAdd(&dLossDMeans2dGaussianPtr->operator[](0), pixelMean2dGradientContribution[0]);
        atomicAdd(&dLossDMeans2dGaussianPtr->operator[](1), pixelMean2dGradientContribution[1]);

        if (mAbsGrad) {
            auto *dLossDMeans2dAbsGaussianPtr = getAccessorPointer<vec2t>(mOutDLossDMeans2dAbs, g);
            atomicAdd(&dLossDMeans2dAbsGaussianPtr->operator[](0),
                      pixelMean2dAbsGradientContribution[0]);
            atomicAdd(&dLossDMeans2dAbsGaussianPtr->operator[](1),
                      pixelMean2dAbsGradientContribution[1]);
        }

        auto *dLossDOpacitiesGaussianPtr = get1DAccessorPointer<ScalarType>(mOutDLossDOpacities, g);
        atomicAdd(dLossDOpacitiesGaussianPtr, pixelOpacityGradientContribution);
    }

    /// @brief Accumulate the features of this Gaussian into the accumulated features
    /// @param gaussianFeatures The features of this Gaussian
    /// @param fac The factor to multiply the features by
    /// @param numChannels The number of channels to accumulate
    /// @param outAccumFeatures The accumulated features
    inline __device__ void
    accumulateFeaturesStep(const ScalarType *gaussianFeatures,
                           const ScalarType fac,
                           const size_t numChannels,
                           ScalarType *outAccumFeatures) const {
        if constexpr (IS_CHUNKED) {
            for (uint32_t k = 0; k < numChannels; ++k) {
                outAccumFeatures[k] += gaussianFeatures[k] * fac;
            }
        } else {
#pragma unroll CHANNEL_UNROLL
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                outAccumFeatures[k] += gaussianFeatures[k] * fac;
            }
        }
    }

    /// @brief Calculate the gradient contribution from this pixel to the gradient of the loss with
    /// respect to the features of this Gaussian
    /// @param fac The factor to multiply the features by
    /// @param dLossDRenderedFeatures The gradient of the loss with respect to the rendered features
    /// @param outFeatureGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the features of this Gaussian
    inline __device__ void
    calculateFeatureGradientContribution(const ScalarType fac,
                                         const ScalarType *dLossDRenderedFeatures,
                                         ScalarType *outFeatureGradientContribution) const {
#pragma unroll SHARED_CHANNEL_UNROLL
        for (uint32_t k = 0; k < NUM_SHARED_CHANNELS; ++k) {
            outFeatureGradientContribution[k] = fac * dLossDRenderedFeatures[k];
        }
    }

    /// @brief Calculate the gradient contribution from this pixel to the gradient of the loss with
    /// respect to the alpha value of this Gaussian
    /// @param cameraId The ID of the camera
    /// @param finalTransmittance The final transmittance for the current pixel
    /// @param oneOverOneMinusAlpha The inverse of (1 - alpha)
    /// @param accumTransmittance The accumulated transmittance for the current pixel
    inline __device__ ScalarType
    calculateAlphaGradientContribution(const uint32_t cameraId,
                                       const ScalarType finalTransmittance,
                                       const ScalarType oneOverOneMinusAlpha,
                                       const ScalarType accumTransmittance,
                                       const ScalarType *accumFeature,
                                       const ScalarType *gaussianFeature,
                                       const ScalarType *dLossDRenderedFeature,
                                       const ScalarType dLossDRenderedAlpha,
                                       const size_t numChannels,
                                       const bool includeLastTerm) const {
        ScalarType alphaGradientContribution = ScalarType{0};
        if constexpr (IS_CHUNKED) {
            for (uint32_t k = 0; k < numChannels; ++k) {
                alphaGradientContribution += (gaussianFeature[k] * accumTransmittance -
                                              accumFeature[k] * oneOverOneMinusAlpha) *
                                             dLossDRenderedFeature[k];
            }
        } else {
#pragma unroll CHANNEL_UNROLL
            for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                alphaGradientContribution += (gaussianFeature[k] * accumTransmittance -
                                              accumFeature[k] * oneOverOneMinusAlpha) *
                                             dLossDRenderedFeature[k];
            }
        }

        if (includeLastTerm) {
            alphaGradientContribution +=
                finalTransmittance * oneOverOneMinusAlpha * dLossDRenderedAlpha;
        }

        // Factor in the contribution from the background to this pixel
        if (commonArgs.mHasBackgrounds) {
            ScalarType accum = ScalarType{0};
            if constexpr (IS_CHUNKED) {
                for (uint32_t k = 0; k < numChannels; ++k) {
                    accum += commonArgs.mBackgrounds[cameraId][k] * dLossDRenderedFeature[k];
                }
            } else {
#pragma unroll CHANNEL_UNROLL
                for (uint32_t k = 0; k < NUM_CHANNELS; ++k) {
                    accum += commonArgs.mBackgrounds[cameraId][k] * dLossDRenderedFeature[k];
                }
            }
            if (includeLastTerm) {
                alphaGradientContribution += -finalTransmittance * oneOverOneMinusAlpha * accum;
            }
        }

        return alphaGradientContribution;
    }

    /// @brief Calculate the gradient contribution from this pixel to the gradients of the loss with
    /// respect to the means, conics, and opacities of this Gaussian
    /// @param opac The opacity of this Gaussian
    /// @param vis The visibility of this Gaussian
    /// @param alphaGradientContribution The gradient contribution from this pixel to the gradient
    /// of the loss with respect to the alpha value of this Gaussian
    inline __device__ void
    calculateMeansConicsAndOpacitiesGradientContribution(
        ScalarType opac,
        ScalarType expMinusSigma,
        ScalarType alphaGradientContribution,
        const vec3t &conic,
        const vec2t &delta,
        vec3t &outConicGradientContribution,
        vec2t &outMean2dGradientContribution,
        vec2t &outMean2dAbsGradientContribution,
        ScalarType &outOpacityGradientContribution) const {
        // Contribution from this pixel to sigma for this Gaussian
        const ScalarType sigmaGradientContribution =
            -opac * expMinusSigma * alphaGradientContribution;
        outConicGradientContribution = {
            ScalarType{0.5} * sigmaGradientContribution * delta[0] * delta[0],
            sigmaGradientContribution * delta[0] * delta[1],
            ScalarType{0.5} * sigmaGradientContribution * delta[1] * delta[1]};
        outMean2dGradientContribution = {
            sigmaGradientContribution * (conic[0] * delta[0] + conic[1] * delta[1]),
            sigmaGradientContribution * (conic[1] * delta[0] + conic[2] * delta[1])};
        if (mAbsGrad) {
            outMean2dAbsGradientContribution = {abs(outMean2dGradientContribution[0]),
                                                abs(outMean2dGradientContribution[1])};
        }
        outOpacityGradientContribution = expMinusSigma * alphaGradientContribution;
    }

    /// @brief Calculate the gradient contribution from this pixel to the gradients of the loss with
    /// respect to the features, means, conics, and opacities of this Gaussian
    /// @param cameraId The ID of the camera
    /// @param gaussian The Gaussian
    /// @param gaussianFeature The features of this Gaussian
    /// @param dLossDRenderedFeature The gradient of the loss with respect to the rendered features
    /// @param dLossDRenderedAlpha The gradient of the loss with respect to the alpha value of this
    /// Gaussian
    /// @param px The x coordinate of the pixel
    /// @param py The y coordinate of the pixel
    /// @param finalTransmittance The final transmittance for the current pixel
    /// @param numChannels The number of channels to accumulate
    /// @param calculateMeansConicsAndOpacitiesGradient Whether to calculate the gradient
    /// contribution from this pixel to the gradients of the loss with respect to the means, conics,
    /// and opacities of this Gaussian
    /// @param accumTransmittance The accumulated transmittance for the current pixel
    /// @param accumFeature The accumulated features for the current pixel
    /// @param outFeatureGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the features of this Gaussian
    /// @param outConicGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the conic of this Gaussian
    /// @param outMean2dGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the 2d mean of this Gaussian
    /// @param outMean2dAbsGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the absolute value of the 2d mean of this Gaussian
    /// @param outOpacityGradientContribution The gradient contribution from this pixel to the
    /// gradient of the loss with respect to the opacity of this Gaussian
    /// @return Whether this Gaussian contributes to this pixel
    inline __device__ void
    calculateGradientContributions(const uint32_t cameraId,
                                   const Gaussian2D<ScalarType> &gaussian,
                                   const vec2t &delta,
                                   const ScalarType expMinusSigma,
                                   const ScalarType alpha,
                                   const ScalarType *gaussianFeature,
                                   const ScalarType *dLossDRenderedFeature,
                                   const ScalarType dLossDRenderedAlpha,
                                   const ScalarType px,
                                   const ScalarType py,
                                   const ScalarType finalTransmittance,
                                   const size_t numChannels,
                                   const bool calculateMeansConicsAndOpacitiesGradient,
                                   ScalarType &accumTransmittance,
                                   ScalarType *accumFeature,
                                   ScalarType *outFeatureGradientContribution,
                                   vec3t &outConicGradientContribution,
                                   vec2t &outMean2dGradientContribution,
                                   vec2t &outMean2dAbsGradientContribution,
                                   ScalarType &outOpacityGradientContribution) const {
        const vec3t conic     = gaussian.conic;
        const ScalarType opac = gaussian.opacity;

        // Compute the transmittance for the current gaussian
        const ScalarType oneOverOneMinusAlpha = ScalarType{1} / (ScalarType{1} - alpha);
        accumTransmittance *= oneOverOneMinusAlpha;

        // Update the contribution of this pixel to the color gradient of the
        // Gaussian
        const ScalarType fac = alpha * accumTransmittance;
        calculateFeatureGradientContribution(
            fac, dLossDRenderedFeature, outFeatureGradientContribution);

        // Contribution from this pixel to the alpha value for this Gaussian
        const ScalarType alphaGradientContribution =
            calculateAlphaGradientContribution(cameraId,
                                               finalTransmittance,
                                               oneOverOneMinusAlpha,
                                               accumTransmittance,
                                               accumFeature,
                                               gaussianFeature,
                                               dLossDRenderedFeature,
                                               dLossDRenderedAlpha,
                                               numChannels,
                                               calculateMeansConicsAndOpacitiesGradient);

        if (opac * expMinusSigma <= commonArgs.ALPHA_THRESHOLD) {
            calculateMeansConicsAndOpacitiesGradientContribution(opac,
                                                                 expMinusSigma,
                                                                 alphaGradientContribution,
                                                                 conic,
                                                                 delta,
                                                                 outConicGradientContribution,
                                                                 outMean2dGradientContribution,
                                                                 outMean2dAbsGradientContribution,
                                                                 outOpacityGradientContribution);
        }

        accumulateFeaturesStep(gaussianFeature, fac, numChannels, accumFeature);
    }

    /// @brief Compute the gradient of the loss with respect to the parameters of the Gaussians
    /// that contribute to this pixel.
    /// @param warp The warp of threads
    /// @param cameraId The ID of the camera
    /// @param i The row index of the pixel
    /// @param j The column index of the pixel
    /// @param firstGaussianIdInBlock The ID of the first Gaussian in the block
    template <size_t WARP_TILE_SIZE>
    inline __device__ void
    volumeRenderTileBackward( // const cooperative_groups::thread_block_tile<WARP_TILE_SIZE> &warp,
        const uint32_t cameraId,
        const uint32_t row,
        const uint32_t col,
        const int64_t firstGaussianIdInBlock,
        const int64_t lastGaussianIdInBlock,
        const uint32_t blockSize,
        const bool pixelIsActive,
        const uint32_t activePixelIndex) {
        alignas(Gaussian2D<ScalarType>) extern __shared__ char s[];

        Gaussian2D<ScalarType> *sharedGaussians =
            reinterpret_cast<Gaussian2D<ScalarType> *>(s);               // [blockSize]
        ScalarType *sharedGaussianFeatures =
            reinterpret_cast<ScalarType *>(&sharedGaussians[blockSize]); // [blockSize]

        // To protect against out of bounds access, we clamp the coordinates to the image bounds
        const auto rowClamped = min(row, commonArgs.renderHeight() - 1);
        const auto colClamped = min(col, commonArgs.renderWidth() - 1);

        const auto pixIdx = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);

        // Only access memory if the pixel is active (within image bounds)
        ScalarType finalTransmittance  = ScalarType{1};
        ScalarType dLossDRenderedAlpha = ScalarType{0};
        int32_t lastIntersectionOffset = -1;

        if (pixelIsActive) {
            // this is the T AFTER the last gaussian in this pixel
            finalTransmittance = ScalarType{1} - readAlpha(pixIdx);

            // Gradient of the loss with respect to the alpha output of the forward pass at this
            // pixel
            dLossDRenderedAlpha = readDLossDRenderedAlpha(pixIdx);

            // Tile-relative offset of the last Gaussian intersection to contribute to this pixel.
            lastIntersectionOffset = readLastId(pixIdx);
        }

        namespace cg = cooperative_groups;
        auto block   = cg::this_thread_block();
        const cg::thread_block_tile<WARP_TILE_SIZE> warp =
            cg::tiled_partition<WARP_TILE_SIZE>(block);

        const int32_t lastIntersectionOffsetInWarp = warpMax(lastIntersectionOffset, warp);
        const int32_t lastIntersectionOffsetInTile =
            tileLastIntersectionOffset(lastIntersectionOffsetInWarp, warp);
        if (lastIntersectionOffsetInTile < 0) {
            return;
        }

        // Load only the prefix that can contribute to an active pixel. The marking pass uses
        // the same cutoff, so only Gaussians in this prefix have local entries.
        // Intersection offsets remain relative to the original tile list.
        const uint32_t tidx            = threadIdx.y * blockDim.x + threadIdx.x;
        const int64_t numIntersections = min(lastGaussianIdInBlock - firstGaussianIdInBlock,
                                             int64_t{lastIntersectionOffsetInTile} + 1);
        const int64_t numBatches       = (numIntersections + blockSize - 1) / blockSize;

        constexpr size_t NUM_CHUNKS =
            (NUM_CHANNELS + NUM_SHARED_CHANNELS - 1) / NUM_SHARED_CHANNELS;
        for (size_t chunk = 0; chunk < NUM_CHUNKS; chunk += 1) {
            const size_t channelStart = chunk * NUM_SHARED_CHANNELS;
            const size_t numChannels  = min(NUM_CHANNELS - channelStart, NUM_SHARED_CHANNELS);
            const bool isLastChunk    = chunk == (NUM_CHUNKS - 1);

            ScalarType accumTransmittance = finalTransmittance;

            // the contribution from gaussians behind the current one
            ScalarType accumFeature[NUM_SHARED_CHANNELS] = {ScalarType(0)};

            // Gradient of the loss with respect to the color output of the forward pass at this
            // pixel
            ScalarType dLossDRenderedFeature[NUM_SHARED_CHANNELS];
            if (pixelIsActive) {
                for (auto k = 0; k < numChannels; ++k) {
                    dLossDRenderedFeature[k] = readDLossDRenderedFeature(pixIdx, channelStart + k);
                }
            } else {
                for (auto k = 0; k < numChannels; ++k) {
                    dLossDRenderedFeature[k] = ScalarType{0};
                }
            }
            for (auto k = numChannels; k < NUM_SHARED_CHANNELS; ++k) {
                dLossDRenderedFeature[k] = ScalarType{0};
            }

            for (int64_t b = 0; b < numBatches; ++b) {
                // resync all threads before writing next batch of shared mem
                __syncthreads();

                // Each thread fetches one gaussian into shared memory.
                // Gaussians are stored in shared memory locations in order of decreasing
                // distance from the camera. Gaussians are processed in batches of size
                // blockSize (i.e. one Gaussian per thread in the block). batchEndOffset is relative
                // to the tile's first intersection. NOTE: These values can be negative so must be
                // signed instead of unsigned.
                const int64_t batchEndOffset = numIntersections - 1 - blockSize * b;
                const int64_t idx            = firstGaussianIdInBlock + batchEndOffset - tidx;
                if (idx >= firstGaussianIdInBlock) {
                    const int32_t g =
                        commonArgs.mTileGaussianIds[idx]; // Gaussian index in [C * N] or [nnz]
                    const int32_t inputId = mGlobalToLocal != nullptr ? mGlobalToLocal[g] : g;
                    // getGaussian preserves inputId in Gaussian2D::id, so gradient atomics use
                    // the same local slot as attribute loads.
                    sharedGaussians[tidx] = commonArgs.getGaussian(inputId);
                    // Only load features if the Gaussian could be valid for any pixel.
                    // If opacity < 1/255, alpha < 1/255 for all pixels regardless of
                    // position, so gaussianIsValid will be false and features are never
                    // read in the rendering loop.
                    if (sharedGaussians[tidx].opacity >= 1.f / 255.f) {
                        ScalarType *feature = &sharedGaussianFeatures[tidx * NUM_SHARED_CHANNELS];
                        fetchGaussianFeatureIntoSharedMemory(
                            inputId, channelStart, numChannels, feature);
                    }
                }

                // Sync threads so all gaussians for this batch are loaded in shared memory
                __syncthreads();

                // process gaussians in the current batch for this pixel
                // 0 index is the furthest back gaussian in the batch
                // For each Gaussian which contributes to this pixel, compute this pixel's
                // gradient contribution to that Gaussian
                const int64_t remaining = batchEndOffset + 1;
                const uint32_t batchSize =
                    static_cast<uint32_t>(remaining < blockSize ? remaining : blockSize);
                const int64_t firstT64 = batchEndOffset > lastIntersectionOffsetInWarp
                                             ? batchEndOffset - lastIntersectionOffsetInWarp
                                             : 0;
                const uint32_t firstT =
                    firstT64 < batchSize ? static_cast<uint32_t>(firstT64) : batchSize;
                for (uint32_t t = firstT; t < batchSize; ++t) {
                    // (row, col) coordinates are relative to the specified image origin which may
                    // be a crop so we need to add the origin to get the absolute pixel coordinates
                    const ScalarType px =
                        col + ScalarType(commonArgs.renderOriginX()) + ScalarType{0.5};
                    const ScalarType py =
                        row + ScalarType(commonArgs.renderOriginY()) + ScalarType{0.5};

                    bool valid = pixelIsActive && (batchEndOffset - t <= lastIntersectionOffset);

                    const auto [gaussianIsValid, delta, expMinusSigma, alpha] = [&]() {
                        if (valid) {
                            return commonArgs.evalGaussian(sharedGaussians[t], px, py);
                        }
                        return std::make_tuple(false, vec2t{}, ScalarType{0}, ScalarType{0});
                    }();

                    valid = valid && gaussianIsValid;

                    // if there are no active thread in this warp, skip this loop
                    if (!warp.any(valid)) {
                        continue;
                    }

                    // How much each pixel contributes to the gradient of the parameters for
                    // this gaussian Initialize to 0 and only set if this pixel is valid
                    ScalarType featureGradientContribution[NUM_SHARED_CHANNELS] = {ScalarType{0}};
                    vec3t conicGradientContribution = {ScalarType{0}, ScalarType{0}, ScalarType{0}};
                    vec2t mean2dGradientContribution       = {ScalarType{0}, ScalarType{0}};
                    vec2t mean2dAbsGradientContribution    = {ScalarType{0}, ScalarType{0}};
                    ScalarType opacityGradientContribution = ScalarType{0};

                    if (valid) {
                        calculateGradientContributions(
                            cameraId,
                            sharedGaussians[t],
                            delta,
                            expMinusSigma,
                            alpha,
                            &sharedGaussianFeatures[t * NUM_SHARED_CHANNELS],
                            dLossDRenderedFeature,
                            dLossDRenderedAlpha,
                            px,
                            py,
                            finalTransmittance,
                            numChannels,
                            isLastChunk,
                            accumTransmittance,
                            accumFeature,
                            featureGradientContribution,
                            conicGradientContribution,
                            mean2dGradientContribution,
                            mean2dAbsGradientContribution,
                            opacityGradientContribution);
                    }

                    // Accumulate the gradient contribution to this Gaussian from every
                    // pixel in the block
                    if constexpr (IS_CHUNKED || NUM_SHARED_CHANNELS > 32) {
                        warpSumMut<decltype(warp), ScalarType>(
                            featureGradientContribution, numChannels, warp);
                    } else {
                        warpSumMut<NUM_SHARED_CHANNELS, decltype(warp), ScalarType>(
                            featureGradientContribution, warp);
                    }

                    warpSumMut<decltype(warp), ScalarType>(conicGradientContribution, warp);
                    warpSumMut<decltype(warp), ScalarType>(mean2dGradientContribution, warp);
                    if (mAbsGrad) {
                        warpSumMut<decltype(warp), ScalarType>(mean2dAbsGradientContribution, warp);
                    }
                    warpSumMut<decltype(warp), ScalarType>(opacityGradientContribution, warp);

                    // The first thread in the block accumulates the gradient
                    // contribution from the whole block into the global gradient of
                    // this Gaussian
                    if (warp.thread_rank() == 0) {
                        atomicAddFeatureGradientContributions(sharedGaussians[t].id,
                                                              featureGradientContribution,
                                                              channelStart,
                                                              numChannels);
                        atomicAddMeans2dConicsAndOpacitiesGradientContributions(
                            sharedGaussians[t].id,
                            conicGradientContribution,
                            mean2dGradientContribution,
                            mean2dAbsGradientContribution,
                            opacityGradientContribution);
                    }
                }
            }
        }
    }
};

/// @brief Compute the gradient of the loss with respect to the parameters of the Gaussians that
/// contribute to each pixel in the image.
/// @param args The arguments for the backward pass
/// @param NUM_CHANNELS The number of channels in the features
/// @param NUM_SHARED_CHANNELS The number of channels to chunk in shared memory
/// @param IS_PACKED Whether the features are packed (i.e. linearized across the outer dimensions)
template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
__global__ void
rasterizeGaussiansBackward(
    RasterizeBackwardArgs<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED> args) {
    auto &commonArgs = args.commonArgs;

    int32_t cameraId;
    int32_t tileRow;
    int32_t tileCol;
    uint32_t row;
    uint32_t col;

    cuda::std::tie(cameraId, tileRow, tileCol, row, col) =
        commonArgs.mIsSparse ? commonArgs.sparseCoordinates() : commonArgs.denseCoordinates();

    // NOTE: We keep threads which correspond to pixels outside the image bounds around
    //       to load gaussians from global memory, but they do not contribute to the output.

    // pixelInImage: Whether this pixel is inside the image bounds.
    // activePixelIndex: Index of this pixel in the output for the block if it is active (sparse
    // mode only).
    bool pixelInImage{false};
    uint32_t activePixelIndex{0};
    cuda::std::tie(pixelInImage, activePixelIndex) = commonArgs.activePixelIndex(row, col);

    // If the caller provides a per-tile mask and this tile is masked, do nothing and return
    if (commonArgs.mHasMasks && !commonArgs.mMasks[cameraId][tileRow][tileCol]) {
        return;
    }

    int64_t firstGaussianIdInBlock;
    int64_t lastGaussianIdInBlock;
    cuda::std::tie(firstGaussianIdInBlock, lastGaussianIdInBlock) =
        commonArgs.tileGaussianRange(cameraId, tileRow, tileCol);

    // Compute the backward pass for the current tile starting at pixel (i, j)
    // and containing Gaussians with ids in [firstGaussianIdInBlock, lastGaussianIdInBlock)
    constexpr uint32_t WARP_TILE_SIZE = 32; // TODO (fwilliams): Tune this value
    args.template volumeRenderTileBackward<WARP_TILE_SIZE>(cameraId,
                                                           row,
                                                           col,
                                                           firstGaussianIdInBlock,
                                                           lastGaussianIdInBlock,
                                                           blockDim.x * blockDim.y,
                                                           pixelInImage,
                                                           activePixelIndex);
}

/// @brief Get the shared memory requirements for the backward pass kernel
/// @param numColorChannels The number of color channels
/// @param tileSize The size of the tile
/// @return The shared memory required in bytes
template <typename ScalarType>
size_t
getSharedMemRequirements(const size_t numColorChannels, const size_t tileSize) {
    return tileSize * tileSize *
           (sizeof(Gaussian2D<ScalarType>) + numColorChannels * sizeof(ScalarType));
}

template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
callRasterizeBackwardWithTemplatedSharedChannels(
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &features,                  // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
    const RenderWindow2D &renderWindow,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,               // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,           // [totalIntersections]
    const torch::Tensor &renderedAlphas,            // [C, H, W, 1] or [P, 1]
    const torch::Tensor &lastGaussianIds,           // [C, H, W] or [P]
    const torch::Tensor &dLossDRenderedFeatures,    // [C, H, W, NUM_CHANNELS] or [P, NUM_CHANNELS]
    const torch::Tensor &dLossDRenderedAlphas,      // [C, H, W, 1] or [P, 1]
    bool absGrad, // True if we are computing the gradient of the absolute gradient of the means2d
    at::cuda::CUDAStream stream,
    const std::optional<torch::Tensor> &activeTiles     = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask   = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap        = std::nullopt) {
    TORCH_CHECK(tileSize > 0, "Tile size must be greater than 0");

    torch::Tensor outDLossDMeans2d   = torch::zeros_like(means2d);
    torch::Tensor outDLossDConics    = torch::zeros_like(conics);
    torch::Tensor outDLossDFeatures  = torch::zeros_like(features);
    torch::Tensor outDLossDOpacities = torch::zeros_like(opacities);
    torch::Tensor outDLossDMeans2dAbs;
    if (absGrad) {
        outDLossDMeans2dAbs = torch::zeros_like(means2d);
    }

    // Just return empty tensors if there are no gaussians, cameras, or intersections
    if (means2d.numel() == 0 || tileGaussianIds.numel() == 0) {
        return std::make_tuple(outDLossDMeans2dAbs,
                               outDLossDMeans2d,
                               outDLossDConics,
                               outDLossDFeatures,
                               outDLossDOpacities);
    }

    // Keep flattening after the empty-work return; sparse inputs are already flat.
    const auto reshapedRenderedAlphas =
        activeTiles.has_value() ? renderedAlphas : renderedAlphas.view({-1, 1});
    const auto reshapedLastGaussianIds =
        activeTiles.has_value() ? lastGaussianIds : lastGaussianIds.view({-1});
    const auto reshapedDLossDRenderedFeatures =
        activeTiles.has_value() ? dLossDRenderedFeatures
                                : dLossDRenderedFeatures.view({-1, NUM_CHANNELS});
    const auto reshapedDLossDRenderedAlphas =
        activeTiles.has_value() ? dLossDRenderedAlphas : dLossDRenderedAlphas.view({-1, 1});

    RasterizeBackwardArgs<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED> args(
        means2d,
        conics,
        opacities,
        features,
        backgrounds,
        masks,
        renderWindow,
        tileSize,
        0,
        tileOffsets,
        tileGaussianIds,
        reshapedRenderedAlphas,
        reshapedLastGaussianIds,
        reshapedDLossDRenderedFeatures,
        reshapedDLossDRenderedAlphas,
        outDLossDMeans2d,
        outDLossDConics,
        outDLossDFeatures,
        outDLossDOpacities,
        absGrad ? std::make_optional(outDLossDMeans2dAbs) : std::nullopt,
        activeTiles,
        tilePixelMask,
        tilePixelCumsum,
        pixelMap);

    const size_t numChannels =
        (NUM_SHARED_CHANNELS == NUM_CHANNELS) ? NUM_CHANNELS : NUM_SHARED_CHANNELS + 1;
    const size_t sharedMemSize = getSharedMemRequirements<ScalarType>(numChannels, tileSize);

    if (cudaFuncSetAttribute(
            rasterizeGaussiansBackward<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedMemSize) != cudaSuccess) {
        AT_ERROR("Failed to set maximum shared memory size (requested ",
                 sharedMemSize,
                 " bytes), try lowering tileSize.");
    }
    rasterizeGaussiansBackward<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED>
        <<<args.commonArgs.getGridDim(), args.commonArgs.getBlockDim(), sharedMemSize, stream>>>(
            args);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(outDLossDMeans2dAbs,
                           outDLossDMeans2d,
                           outDLossDConics,
                           outDLossDFeatures,
                           outDLossDOpacities);
}

template <typename ScalarType, size_t NUM_CHANNELS, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
callRasterizeBackwardWithCorrectSharedChannels(
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &features,                  // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
    const RenderWindow2D &renderWindow,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,               // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,           // [totalIntersections]
    const torch::Tensor &renderedAlphas,            // [C, H, W, 1] or [P, 1]
    const torch::Tensor &lastGaussianIds,           // [C, H, W] or [P]
    const torch::Tensor &dLossDRenderedFeatures,    // [C, H, W, NUM_CHANNELS] or [P, NUM_CHANNELS]
    const torch::Tensor &dLossDRenderedAlphas,      // [C, H, W, 1] or [P, 1]
    const bool absGrad,
    const int64_t numSharedSharedChannelsOverride,
    const std::optional<torch::Tensor> &activeTiles     = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask   = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap        = std::nullopt) {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    auto callWithSharedChannels = [&](size_t numSharedChannels) {
        if (numSharedChannels == NUM_CHANNELS) {
            return callRasterizeBackwardWithTemplatedSharedChannels<ScalarType,
                                                                    NUM_CHANNELS,
                                                                    NUM_CHANNELS,
                                                                    IS_PACKED>(
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                masks,
                renderWindow,
                tileSize,
                tileOffsets,
                tileGaussianIds,
                renderedAlphas,
                lastGaussianIds,
                dLossDRenderedFeatures,
                dLossDRenderedAlphas,
                absGrad,
                stream,
                activeTiles,
                tilePixelMask,
                tilePixelCumsum,
                pixelMap);
        } else if (numSharedChannels == 64) {
            return callRasterizeBackwardWithTemplatedSharedChannels<ScalarType,
                                                                    NUM_CHANNELS,
                                                                    64,
                                                                    IS_PACKED>(
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                masks,
                renderWindow,
                tileSize,
                tileOffsets,
                tileGaussianIds,
                renderedAlphas,
                lastGaussianIds,
                dLossDRenderedFeatures,
                dLossDRenderedAlphas,
                absGrad,
                stream,
                activeTiles,
                tilePixelMask,
                tilePixelCumsum,
                pixelMap);
        } else if (numSharedChannels == 32) {
            return callRasterizeBackwardWithTemplatedSharedChannels<ScalarType,
                                                                    NUM_CHANNELS,
                                                                    32,
                                                                    IS_PACKED>(
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                masks,
                renderWindow,
                tileSize,
                tileOffsets,
                tileGaussianIds,
                renderedAlphas,
                lastGaussianIds,
                dLossDRenderedFeatures,
                dLossDRenderedAlphas,
                absGrad,
                stream,
                activeTiles,
                tilePixelMask,
                tilePixelCumsum,
                pixelMap);
        } else if (numSharedChannels == 16) {
            return callRasterizeBackwardWithTemplatedSharedChannels<ScalarType,
                                                                    NUM_CHANNELS,
                                                                    16,
                                                                    IS_PACKED>(
                means2d,
                conics,
                features,
                opacities,
                backgrounds,
                masks,
                renderWindow,
                tileSize,
                tileOffsets,
                tileGaussianIds,
                renderedAlphas,
                lastGaussianIds,
                dLossDRenderedFeatures,
                dLossDRenderedAlphas,
                absGrad,
                stream,
                activeTiles,
                tilePixelMask,
                tilePixelCumsum,
                pixelMap);
        } else {
            if (numSharedSharedChannelsOverride > 0) {
                AT_ERROR("Invalid numSharedChannelsOverride. Must be 64, 32, or 16.");
            } else {
                AT_ERROR("Failed to set maximum shared memory size");
            }
        }
    };

    if (numSharedSharedChannelsOverride > 0) {
        return callWithSharedChannels(numSharedSharedChannelsOverride);
    } else {
        const size_t maxSharedMemory = fvdb::detail::getMaxSharedMemory(stream.device_index());

        const size_t sharedMemChannelOptions[4] = {NUM_CHANNELS, 64, 32, 16};
        for (size_t i = 0; i < 4; ++i) {
            const size_t numSharedChannels = sharedMemChannelOptions[i];
            if (getSharedMemRequirements<ScalarType>(numSharedChannels, tileSize) <=
                maxSharedMemory) {
                return callWithSharedChannels(numSharedChannels);
            }
        }
        AT_ERROR("Failed to set maximum shared memory size");
    }
}

// Allocate device-local storage and free it on the same stream. Callers must order any remote
// reads before destruction, as the final mergeStreams does for local gradients.
torch::Tensor
makeDeviceBuffer(at::IntArrayRef sizes,
                 const torch::TensorOptions &options,
                 c10::DeviceIndex deviceId,
                 cudaStream_t stream) {
    const c10::Device device(c10::kCUDA, deviceId);

    const size_t numBytes = at::detail::computeStorageNbytesContiguous(
        sizes, c10::elementSize(options.dtype().toScalarType()));
    void *ptr = nullptr;
    if (numBytes != 0) {
        C10_CUDA_CHECK(cudaMallocAsync(&ptr, numBytes, stream));
    }
    return at::from_blob(
        ptr,
        sizes,
        [deviceId, stream](void *data) noexcept {
            if (data == nullptr) {
                return;
            }
            C10_CUDA_CHECK(cudaSetDevice(deviceId));
            C10_CUDA_CHECK(cudaFreeAsync(data, stream));
        },
        options.device(device),
        device);
}

// Replace the source's Gaussian dimensions with the number requested on this device.
template <bool IS_PACKED>
torch::Tensor
makeLocalGaussianBuffer(const torch::Tensor &source,
                        int64_t numRequested,
                        c10::DeviceIndex deviceId,
                        cudaStream_t stream) {
    std::vector<int64_t> shape{numRequested};
    if (source.dim() == (IS_PACKED ? 2 : 3)) {
        shape.push_back(source.size(-1));
    }
    return makeDeviceBuffer(shape, source.options(), deviceId, stream);
}

template <bool IS_PACKED>
std::vector<torch::Tensor>
makeLocalGaussianGradients(const std::vector<torch::Tensor> &globalGradients,
                           int64_t numRequested,
                           c10::DeviceIndex deviceId,
                           cudaStream_t stream) {
    std::vector<torch::Tensor> gradients;
    gradients.reserve(globalGradients.size());
    for (const auto &output: globalGradients) {
        auto gradient = makeLocalGaussianBuffer<IS_PACKED>(output, numRequested, deviceId, stream);
        C10_CUDA_CHECK(cudaMemsetAsync(
            gradient.data_ptr(), 0, gradient.numel() * gradient.element_size(), stream));
        gradients.push_back(std::move(gradient));
    }
    return gradients;
}

// All per-device buffers use the same local ordering. Only the global IDs and gradients
// survive rasterization; reduction needs them to restore the global output indices.
struct LocalGaussians {
    enum Gradient { MEANS2D, CONICS, FEATURES, OPACITIES, MEANS2D_ABS };

    torch::Tensor requestMask;            // [G + 1], including the scan sentinel
    torch::Tensor globalToLocal;          // [G + 1], exclusive prefix counts
    torch::Tensor localToGlobal;          // [U], sorted global IDs
    std::vector<torch::Tensor> gradients; // Indexed by Gradient, each [U] or [U, K]
};

// A contiguous part of one global gradient owned by one GPU.
struct GradientShard {
    size_t tensorIndex;
    c10::DeviceIndex ownerId;
    int64_t firstElement;
    int64_t numElements;
    int64_t numComponents;

    int64_t
    firstGaussian() const {
        return firstElement / numComponents;
    }

    int64_t
    endGaussian() const {
        return cuda::ceil_div(firstElement + numElements, numComponents);
    }
};

// Preserve global gradient placement: dense gradients are sharded by Gaussian per camera; packed
// gradients retain their flat element shards, which can split a Gaussian's component vector.
template <bool IS_PACKED>
std::vector<GradientShard>
makeGradientShards(const std::vector<torch::Tensor> &globalGradients, int64_t numGaussians) {
    std::vector<GradientShard> shards;
    for (size_t tensorIndex = 0; tensorIndex < globalGradients.size(); ++tensorIndex) {
        const auto &output          = globalGradients[tensorIndex];
        const int64_t numComponents = output.numel() / numGaussians;
        for (const auto ownerId: c10::irange(c10::cuda::device_count())) {
            if constexpr (IS_PACKED) {
                int64_t offset, count;
                std::tie(offset, count) = deviceChunk(output.numel(), ownerId);
                if (count > 0) {
                    shards.push_back({tensorIndex, ownerId, offset, count, numComponents});
                }
            } else {
                int64_t offset, count;
                std::tie(offset, count) = deviceChunk(output.size(1), ownerId);
                if (count > 0) {
                    for (int64_t cameraId = 0; cameraId < output.size(0); ++cameraId) {
                        shards.push_back({tensorIndex,
                                          ownerId,
                                          (cameraId * output.size(1) + offset) * numComponents,
                                          count * numComponents,
                                          numComponents});
                    }
                }
            }
        }
    }
    return shards;
}

template <typename ScalarType>
__global__ void
scatterAddLocalGradients(const int32_t *__restrict__ localToGlobal,
                         const ScalarType *__restrict__ localGradients,
                         ScalarType *__restrict__ globalGradient,
                         const int64_t numValues,
                         const int64_t numComponents,
                         const int64_t firstElement,
                         const int64_t endElement) {
    const int64_t index = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (index >= numValues) {
        return;
    }
    const int64_t outputIndex =
        int64_t{localToGlobal[index / numComponents]} * numComponents + index % numComponents;
    // Packed output ownership can split a Gaussian across devices. Each owner writes only its
    // own components, even though the transferred local rows contain complete Gaussians.
    if (outputIndex >= firstElement && outputIndex < endElement) {
        globalGradient[outputIndex] += localGradients[index];
    }
}

// Every source list contains unique sorted IDs. Sources are accumulated sequentially on the
// owner's stream, so scatter-add needs no atomics. The caller orders all source producers and
// each owner's output discard-prefetch before this function, and remote reads before source frees.
template <typename ScalarType>
void
reduceScatterGradients(const std::vector<GradientShard> &shards,
                       const torch::TensorAccessor<int32_t, 3> shardOffsets,
                       const std::vector<LocalGaussians> &localGaussians,
                       const std::vector<torch::Tensor> &globalGradients) {
    for (size_t shardIndex = 0; shardIndex < shards.size(); ++shardIndex) {
        const auto &shard  = shards[shardIndex];
        const auto &output = globalGradients[shard.tensorIndex];
        C10_CUDA_CHECK(cudaSetDevice(shard.ownerId));
        auto stream     = c10::cuda::getCurrentCUDAStream(shard.ownerId);
        auto *outputPtr = output.data_ptr<ScalarType>();
        C10_CUDA_CHECK(cudaMemsetAsync(
            outputPtr + shard.firstElement, 0, shard.numElements * sizeof(ScalarType), stream));

        int64_t maxRemoteCount = 0;
        for (const auto sourceId: c10::irange(c10::cuda::device_count())) {
            if (sourceId != shard.ownerId) {
                const auto offsets = shardOffsets[sourceId][shardIndex];
                maxRemoteCount     = std::max(maxRemoteCount, int64_t{offsets[1]} - offsets[0]);
            }
        }
        torch::Tensor receivedIds;
        torch::Tensor receivedGradients;
        if (maxRemoteCount > 0) {
            receivedIds = makeDeviceBuffer(
                {maxRemoteCount}, output.options().dtype(torch::kInt32), shard.ownerId, stream);
            receivedGradients = makeDeviceBuffer(
                {maxRemoteCount, shard.numComponents}, output.options(), shard.ownerId, stream);
        }

        for (const auto sourceId: c10::irange(c10::cuda::device_count())) {
            const auto offsets  = shardOffsets[sourceId][shardIndex];
            const int64_t begin = offsets[0];
            const int64_t count = offsets[1] - begin;
            if (count == 0) {
                continue;
            }
            const auto &source      = localGaussians[sourceId];
            const int64_t numValues = count * shard.numComponents;
            const auto *ids         = source.localToGlobal.const_data_ptr<int32_t>() + begin;
            const auto *values = source.gradients[shard.tensorIndex].const_data_ptr<ScalarType>() +
                                 begin * shard.numComponents;
            if (sourceId != shard.ownerId) {
                C10_CUDA_CHECK(cudaMemcpyPeerAsync(receivedIds.data_ptr<int32_t>(),
                                                   shard.ownerId,
                                                   ids,
                                                   sourceId,
                                                   count * sizeof(int32_t),
                                                   stream));
                C10_CUDA_CHECK(cudaMemcpyPeerAsync(receivedGradients.data_ptr<ScalarType>(),
                                                   shard.ownerId,
                                                   values,
                                                   sourceId,
                                                   numValues * sizeof(ScalarType),
                                                   stream));
                ids    = receivedIds.const_data_ptr<int32_t>();
                values = receivedGradients.const_data_ptr<ScalarType>();
            }
            scatterAddLocalGradients<ScalarType><<<GET_BLOCKS(numValues, 256), 256, 0, stream>>>(
                ids,
                values,
                outputPtr,
                numValues,
                shard.numComponents,
                shard.firstElement,
                shard.firstElement + shard.numElements);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
    }
}

// Mark each assigned tile's prefix through the last intersection contributing to any active pixel.
// Match the rasterizer's block shape and pixel mapping, including sparse masks and image edges.
template <typename ScalarType, size_t NUM_CHANNELS, bool IS_PACKED>
__global__ void
markRequestedGaussians(RasterizeCommonArgs<ScalarType, NUM_CHANNELS, IS_PACKED> commonArgs,
                       fvdb::TorchRAcc64<int32_t, 1> lastGaussianIds,
                       bool *requestMask) {
    const auto [cameraId, tileRow, tileCol, row, col] =
        commonArgs.mIsSparse ? commonArgs.sparseCoordinates() : commonArgs.denseCoordinates();
    if (commonArgs.mHasMasks && !commonArgs.mMasks[cameraId][tileRow][tileCol]) {
        return;
    }

    const auto [pixelIsActive, activePixelIndex] = commonArgs.activePixelIndex(row, col);
    int32_t lastIntersectionOffset               = -1;
    if (pixelIsActive) {
        const auto pixelIndex  = commonArgs.pixelIndex(cameraId, row, col, activePixelIndex);
        lastIntersectionOffset = lastGaussianIds[pixelIndex];
    }

    namespace cg                               = cooperative_groups;
    auto block                                 = cg::this_thread_block();
    auto warp                                  = cg::tiled_partition<32>(block);
    const int32_t lastIntersectionOffsetInWarp = warpMax(lastIntersectionOffset, warp);
    const int32_t lastIntersectionOffsetInTile =
        tileLastIntersectionOffset(lastIntersectionOffsetInWarp, warp);
    if (lastIntersectionOffsetInTile < 0) {
        return;
    }

    const auto [firstIntersection, lastIntersection] =
        commonArgs.tileGaussianRange(cameraId, tileRow, tileCol);
    const int64_t contributingEnd =
        min(lastIntersection, firstIntersection + int64_t{lastIntersectionOffsetInTile} + 1);
    const uint32_t tidx      = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t blockSize = blockDim.x * blockDim.y;
    for (int64_t intersection = firstIntersection + tidx; intersection < contributingEnd;
         intersection += blockSize) {
        const int32_t g = commonArgs.mTileGaussianIds[intersection];
        // Multiple tile lists can contain the same Gaussian. The next kernel on this stream
        // observes all marks, so relaxed device-scope atomic stores are sufficient.
        cuda::atomic_ref<bool, cuda::thread_scope_device> requested(requestMask[g]);
        requested.store(true, cuda::std::memory_order_relaxed);
    }
}

struct GaussianRequestFlag {
    __host__ __device__ int32_t
    operator()(bool requested) const {
        return requested;
    }
};

// Replace each metadata index with its scan value: the request count followed by shard bounds.
__global__ void
gatherGaussianRequestMetadata(const int32_t *__restrict__ globalToLocal,
                              int32_t *__restrict__ metadata,
                              const int64_t numValues) {
    const int64_t index = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (index < numValues) {
        metadata[index] = globalToLocal[metadata[index]];
    }
}

// Scan bool flags as int32 counts, then read the count and all shard bounds in one D2H copy.
// metadataIndices selects globalToLocal[G] and the scan values at each shard's boundaries.
void
scanGaussianRequests(LocalGaussians &local,
                     const torch::Tensor &metadataIndices,
                     int32_t *hostMetadata,
                     cudaStream_t stream) {
    const auto metadata        = makeDeviceBuffer(metadataIndices.sizes(),
                                           local.globalToLocal.options(),
                                           local.globalToLocal.get_device(),
                                           stream);
    const size_t metadataBytes = metadataIndices.numel() * sizeof(int32_t);
    C10_CUDA_CHECK(cudaMemcpyAsync(metadata.data_ptr<int32_t>(),
                                   metadataIndices.const_data_ptr<int32_t>(),
                                   metadataBytes,
                                   cudaMemcpyHostToDevice,
                                   stream));

    const int64_t numGaussians = local.requestMask.numel() - 1;
    auto *globalToLocal        = local.globalToLocal.data_ptr<int32_t>();
    const auto requestFlags =
        thrust::make_transform_iterator(local.requestMask.data_ptr<bool>(), GaussianRequestFlag{});
    FVDB_CUB_WRAPPER_ASYNC(
        stream, cub::DeviceScan::ExclusiveSum, requestFlags, globalToLocal, numGaussians + 1);
    gatherGaussianRequestMetadata<<<GET_BLOCKS(metadata.numel(), 256), 256, 0, stream>>>(
        globalToLocal, metadata.data_ptr<int32_t>(), metadata.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    C10_CUDA_CHECK(cudaMemcpyAsync(hostMetadata,
                                   metadata.const_data_ptr<int32_t>(),
                                   metadataBytes,
                                   cudaMemcpyDeviceToHost,
                                   stream));
}

// The exclusive scan assigns local IDs in ascending global-ID order. Only marked entries of
// globalToLocal are local IDs, so only they are inverted.
__global__ void
invertGlobalToLocal(const bool *__restrict__ requestMask,
                    const int32_t *__restrict__ globalToLocal,
                    int32_t *__restrict__ localToGlobal,
                    const int64_t numGaussians) {
    const int64_t g = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g < numGaussians && requestMask[g]) {
        localToGlobal[globalToLocal[g]] = static_cast<int32_t>(g);
    }
}

struct GaussianInputLayout {
    int64_t gaussiansPerCamera;
    int64_t cameraStride;
    int64_t gaussianStride;
    int64_t componentStride;
};

template <typename ScalarType, bool IS_PACKED>
__global__ void
gatherLocalGaussianInputKernel(const ScalarType *__restrict__ source,
                               ScalarType *__restrict__ destination,
                               const int32_t *__restrict__ localToGlobal,
                               const GaussianInputLayout layout,
                               const int64_t numComponents,
                               const int64_t numElements) {
    const int64_t index = int64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (index >= numElements) {
        return;
    }
    const int64_t g         = localToGlobal[index / numComponents];
    const int64_t component = index % numComponents;
    int64_t sourceOffset;
    if constexpr (IS_PACKED) {
        sourceOffset = g * layout.gaussianStride;
    } else {
        sourceOffset = (g / layout.gaussiansPerCamera) * layout.cameraStride +
                       (g % layout.gaussiansPerCamera) * layout.gaussianStride;
    }
    destination[index] = source[sourceOffset + component * layout.componentStride];
}

// Visit only selected Gaussians, in global-ID order. Respect source strides while writing
// contiguous local storage, including broadcast or permuted source views.
template <typename ScalarType, bool IS_PACKED>
torch::Tensor
gatherLocalGaussianInput(const torch::Tensor &source,
                         const torch::Tensor &localToGlobal,
                         c10::DeviceIndex deviceId,
                         cudaStream_t stream) {
    torch::Tensor destination =
        makeLocalGaussianBuffer<IS_PACKED>(source, localToGlobal.numel(), deviceId, stream);
    if (destination.numel() == 0) {
        return destination;
    }
    constexpr int NUM_OUTER_DIMS = IS_PACKED ? 1 : 2;
    const int64_t numComponents  = source.dim() == NUM_OUTER_DIMS ? 1 : source.size(-1);
    const GaussianInputLayout layout{
        IS_PACKED ? 0 : source.size(1),
        IS_PACKED ? 0 : source.stride(0),
        source.stride(NUM_OUTER_DIMS - 1),
        source.dim() == NUM_OUTER_DIMS ? 0 : source.stride(NUM_OUTER_DIMS)};
    gatherLocalGaussianInputKernel<ScalarType, IS_PACKED>
        <<<GET_BLOCKS(destination.numel(), 256), 256, 0, stream>>>(
            source.const_data_ptr<ScalarType>(),
            destination.data_ptr<ScalarType>(),
            localToGlobal.const_data_ptr<int32_t>(),
            layout,
            numComponents,
            destination.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return destination;
}

// Each GPU gathers local copies of the Gaussians its tiles request. Each GPU then runs the same
// backward rasterization kernel on its local Gaussians, accumulating local gradients. The local
// gradients from every GPU are reduce-scattered into the global gradients, with each GPU reducing
// the shard it owns.
template <typename ScalarType, size_t NUM_CHANNELS, size_t NUM_SHARED_CHANNELS, bool IS_PACKED>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
callRasterizeBackwardPrivateUse1(
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &features,                  // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks,       // [C, numTilesH, numTilesW]
    const RenderWindow2D &renderWindow,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,               // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,           // [totalIntersections]
    const torch::Tensor &renderedAlphas,            // [C, H, W, 1] or [P, 1]
    const torch::Tensor &lastGaussianIds,           // [C, H, W] or [P]
    const torch::Tensor &dLossDRenderedFeatures,    // [C, H, W, NUM_CHANNELS] or [P, NUM_CHANNELS]
    const torch::Tensor &dLossDRenderedAlphas,      // [C, H, W, 1] or [P, 1]
    bool absGrad,
    const std::optional<torch::Tensor> &activeTiles     = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelMask   = std::nullopt,
    const std::optional<torch::Tensor> &tilePixelCumsum = std::nullopt,
    const std::optional<torch::Tensor> &pixelMap        = std::nullopt) {
    TORCH_CHECK(tileSize > 0, "Tile size must be greater than 0");

    // Just return empty tensors if there are no gaussians, cameras, or intersections
    if (means2d.numel() == 0 || tileGaussianIds.numel() == 0) {
        return std::make_tuple(absGrad ? torch::zeros_like(means2d) : torch::Tensor(),
                               torch::zeros_like(means2d),
                               torch::zeros_like(conics),
                               torch::zeros_like(features),
                               torch::zeros_like(opacities));
    }

    // Keep flattening after the empty-work return; sparse inputs are already flat.
    const auto reshapedRenderedAlphas =
        activeTiles.has_value() ? renderedAlphas : renderedAlphas.view({-1, 1});
    const auto reshapedLastGaussianIds =
        activeTiles.has_value() ? lastGaussianIds : lastGaussianIds.view({-1});
    const auto reshapedDLossDRenderedFeatures =
        activeTiles.has_value() ? dLossDRenderedFeatures
                                : dLossDRenderedFeatures.view({-1, NUM_CHANNELS});
    const auto reshapedDLossDRenderedAlphas =
        activeTiles.has_value() ? dLossDRenderedAlphas : dLossDRenderedAlphas.view({-1, 1});

    // Local gradients are scattered into the global gradient layout after owner-directed
    // transfers. Dense Gaussian ownership and packed flat-element ownership stay unchanged.
    torch::Tensor outDLossDMeans2d   = torch::empty(means2d.sizes(), means2d.options());
    torch::Tensor outDLossDConics    = torch::empty(conics.sizes(), conics.options());
    torch::Tensor outDLossDFeatures  = torch::empty(features.sizes(), features.options());
    torch::Tensor outDLossDOpacities = torch::empty(opacities.sizes(), opacities.options());
    torch::Tensor outDLossDMeans2dAbs;
    if (absGrad) {
        outDLossDMeans2dAbs = torch::empty(means2d.sizes(), means2d.options());
    }

    // tileOffsets can be 3D (dense) or 1D (sparse).
    const bool tileOffsetsAreSparse = tileOffsets.dim() == 1;
    const uint32_t C                = tileOffsetsAreSparse ? 0 : tileOffsets.size(0);
    const int64_t tileExtentH       = renderWindow.tileExtentH(tileSize);
    const int64_t tileExtentW       = renderWindow.tileExtentW(tileSize);
    const int64_t tilesPerCamera    = tileExtentH * tileExtentW;
    // Compute tile count: for sparse mode use activeTiles, for dense mode compute from image dims.
    const int64_t tileCount =
        activeTiles.has_value() ? activeTiles.value().size(0) : C * tilesPerCamera;
    // Use the forward pass's full-width tile rows for both input prefetch and rasterization.
    // Sparse active-tile lists retain their flat partition; empty images need nonzero alignment.
    const int64_t tileAlignment = activeTiles.has_value() || tileCount == 0 ? 1 : tileExtentW;
    std::vector<torch::Tensor> tileTensors = {dLossDRenderedFeatures, dLossDRenderedAlphas};

    // Keep global and local gradient slots in LocalGaussians::Gradient order.
    std::vector<torch::Tensor> globalGradients = {
        outDLossDMeans2d, outDLossDConics, outDLossDFeatures, outDLossDOpacities};
    if (absGrad) {
        globalGradients.emplace_back(outDLossDMeans2dAbs);
    }

    std::vector<cudaStream_t> prefetchStreams(c10::cuda::device_count());
    // Keep each device's current compute stream unchanged through allocation, rasterization,
    // reduction, and release: the owning tensors enqueue their frees on their allocation streams.
    std::vector<LocalGaussians> localGaussians(c10::cuda::device_count());

    // Prefetch inputs after prior work, then make the current streams wait for those inputs.
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream               = c10::cuda::getCurrentCUDAStream(deviceId);
        prefetchStreams[deviceId] = c10::cuda::getStreamFromPool(false, deviceId);

        cudaEvent_t prefetchEvent;
        C10_CUDA_CHECK(cudaEventCreateWithFlags(&prefetchEvent, cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(prefetchEvent, stream));
        C10_CUDA_CHECK(cudaStreamWaitEvent(prefetchStreams[deviceId], prefetchEvent));

        int64_t deviceTileOffset, deviceTileCount;
        std::tie(deviceTileOffset, deviceTileCount) =
            deviceAlignedChunk(tileAlignment, tileCount, deviceId);
        if (deviceTileCount) {
            std::vector<void *> prefetchPointers;
            std::vector<size_t> prefetchSizes;
            const TilePrefetchRange tileRange{deviceTileOffset,
                                              deviceTileCount,
                                              tileExtentH,
                                              tileExtentW,
                                              renderWindow.height,
                                              renderWindow.width,
                                              tileSize};
            appendPerTilePrefetchRanges(prefetchPointers, prefetchSizes, tileTensors, tileRange);

            memPrefetchBatchAsync(
                prefetchPointers, prefetchSizes, deviceId, prefetchStreams[deviceId]);
        }
        C10_CUDA_CHECK(cudaEventRecord(prefetchEvent, prefetchStreams[deviceId]));
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, prefetchEvent));
        C10_CUDA_CHECK(cudaEventDestroy(prefetchEvent));
    }

    const int64_t numGaussians = opacities.numel();
    TORCH_CHECK_VALUE(numGaussians > 0 && numGaussians <= std::numeric_limits<int32_t>::max(),
                      "Local Gaussians require a positive Gaussian count that fits in int32");
    const std::vector<GradientShard> gradientShards =
        makeGradientShards<IS_PACKED>(globalGradients, numGaussians);
    const int64_t numGradientShards = gradientShards.size();
    const int64_t numMetadataValues = 1 + 2 * numGradientShards;
    const auto metadataOptions =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    // Every device reads the same scan positions: total count, then [begin, end) for each shard.
    // Keep this pinned index table alive until all H2D copies have completed.
    auto metadataIndices = torch::empty({numMetadataValues}, metadataOptions);
    auto *metadataIndicesPtr = metadataIndices.data_ptr<int32_t>();
    metadataIndicesPtr[0] = static_cast<int32_t>(numGaussians);
    for (size_t shardIndex = 0; shardIndex < gradientShards.size(); ++shardIndex) {
        metadataIndicesPtr[1 + 2 * shardIndex] =
            static_cast<int32_t>(gradientShards[shardIndex].firstGaussian());
        metadataIndicesPtr[2 + 2 * shardIndex] =
            static_cast<int32_t>(gradientShards[shardIndex].endGaussian());
    }
    // One contiguous pinned destination per device; devices without tiles retain zero metadata.
    auto hostMetadata =
        torch::zeros({c10::cuda::device_count(), numMetadataValues}, metadataOptions);
    auto requestMetadata = hostMetadata.accessor<int32_t, 2>();
    const torch::Tensor hostShardOffsets =
        hostMetadata.narrow(1, 1, 2 * numGradientShards)
            .view({c10::cuda::device_count(), numGradientShards, 2});
    auto shardOffsets = hostShardOffsets.accessor<int32_t, 3>();
    // Enqueue every device's scan and readback before waiting for any allocation counts.
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        uint32_t deviceTileOffset, deviceTileCount;
        std::tie(deviceTileOffset, deviceTileCount) =
            deviceAlignedChunk(tileAlignment, tileCount, deviceId);
        if (deviceTileCount == 0) {
            continue;
        }

        // Validate pixel inputs before marking reads lastGaussianIds. Output tensors are only
        // shape metadata here; the marking kernel receives no gradient accessors.
        RasterizeBackwardArgs<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, IS_PACKED> args(
            means2d,
            conics,
            opacities,
            features,
            backgrounds,
            masks,
            renderWindow,
            tileSize,
            deviceTileOffset,
            tileOffsets,
            tileGaussianIds,
            reshapedRenderedAlphas,
            reshapedLastGaussianIds,
            reshapedDLossDRenderedFeatures,
            reshapedDLossDRenderedAlphas,
            outDLossDMeans2d,
            outDLossDConics,
            outDLossDFeatures,
            outDLossDOpacities,
            absGrad ? std::make_optional(outDLossDMeans2dAbs) : std::nullopt,
            activeTiles,
            tilePixelMask,
            tilePixelCumsum,
            pixelMap);
        auto &local       = localGaussians[deviceId];
        local.requestMask = makeDeviceBuffer(
            {numGaussians + 1}, opacities.options().dtype(torch::kBool), deviceId, stream);
        local.globalToLocal = makeDeviceBuffer(
            {numGaussians + 1}, opacities.options().dtype(torch::kInt32), deviceId, stream);
        auto *requestMask = local.requestMask.data_ptr<bool>();
        C10_CUDA_CHECK(cudaMemsetAsync(requestMask, 0, (numGaussians + 1) * sizeof(bool), stream));

        const dim3 blockDim = {tileSize, tileSize, 1};
        const dim3 gridDim  = {deviceTileCount, 1, 1};
        markRequestedGaussians<ScalarType, NUM_CHANNELS, IS_PACKED>
            <<<gridDim, blockDim, 0, stream>>>(args.commonArgs, args.mLastGaussianIds, requestMask);
        C10_CUDA_KERNEL_LAUNCH_CHECK();

        scanGaussianRequests(local, metadataIndices, requestMetadata[deviceId].data(), stream);
    }

    // Submit every output prefetch before the count readback can block the host launch thread.
    // Reuse the input prefetch streams to preserve their ordering. Rasterization writes local
    // gradients, so only the reduction needs to wait for output placement.
    std::vector<cudaEvent_t> outputPrefetchEvents(c10::cuda::device_count());
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto prefetchStream = prefetchStreams[deviceId];
        std::vector<void *> prefetchPointers;
        std::vector<size_t> prefetchSizes;
        for (const auto &shard: gradientShards) {
            if (shard.ownerId == deviceId) {
                prefetchPointers.emplace_back(
                    globalGradients[shard.tensorIndex].data_ptr<ScalarType>() + shard.firstElement);
                prefetchSizes.emplace_back(shard.numElements * sizeof(ScalarType));
            }
        }
        memDiscardAndPrefetchBatchAsync(prefetchPointers, prefetchSizes, deviceId, prefetchStream);

        C10_CUDA_CHECK(
            cudaEventCreateWithFlags(&outputPrefetchEvents[deviceId], cudaEventDisableTiming));
        C10_CUDA_CHECK(cudaEventRecord(outputPrefetchEvents[deviceId], prefetchStream));
    }

    // Counts must reach the host before exact-size allocation. All scans and count copies above
    // are already enqueued, so one device can gather and rasterize while another finishes its scan.
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);

        uint32_t deviceTileOffset, deviceTileCount;
        std::tie(deviceTileOffset, deviceTileCount) =
            deviceAlignedChunk(tileAlignment, tileCount, deviceId);

        if (deviceTileCount == 0) {
            continue;
        }
        auto &local = localGaussians[deviceId];
        C10_CUDA_CHECK(cudaStreamSynchronize(stream));
        const int64_t numRequested = requestMetadata[deviceId][0];
        if (numRequested == 0) {
            local.requestMask   = torch::Tensor();
            local.globalToLocal = torch::Tensor();
            continue;
        }

        // Build the inverse map for gathering inputs and scattering gradients to their owners.
        local.localToGlobal = makeDeviceBuffer(
            {numRequested}, opacities.options().dtype(torch::kInt32), deviceId, stream);
        const auto *globalToLocal = local.globalToLocal.const_data_ptr<int32_t>();
        invertGlobalToLocal<<<GET_BLOCKS(numGaussians, 256), 256, 0, stream>>>(
            local.requestMask.const_data_ptr<bool>(),
            globalToLocal,
            local.localToGlobal.data_ptr<int32_t>(),
            numGaussians);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        local.requestMask = torch::Tensor();

        const auto localMeans2d = gatherLocalGaussianInput<ScalarType, IS_PACKED>(
            means2d, local.localToGlobal, deviceId, stream);
        const auto localConics = gatherLocalGaussianInput<ScalarType, IS_PACKED>(
            conics, local.localToGlobal, deviceId, stream);
        const auto localOpacities = gatherLocalGaussianInput<ScalarType, IS_PACKED>(
            opacities, local.localToGlobal, deviceId, stream);
        const auto localFeatures = gatherLocalGaussianInput<ScalarType, IS_PACKED>(
            features, local.localToGlobal, deviceId, stream);
        local.gradients =
            makeLocalGaussianGradients<IS_PACKED>(globalGradients, numRequested, deviceId, stream);

        // Local input and gradient tensors share the existing packed layout. The global
        // source layout is needed only for gathering attributes and assigning output owners.
        // Backgrounds and masks were validated before request marking. Attach them below;
        // packed sparse inputs have no camera count for repeating their shape checks.
        RasterizeBackwardArgs<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, true> args(
            localMeans2d,
            localConics,
            localOpacities,
            localFeatures,
            std::nullopt,
            std::nullopt,
            renderWindow,
            tileSize,
            deviceTileOffset,
            tileOffsets,
            tileGaussianIds,
            reshapedRenderedAlphas,
            reshapedLastGaussianIds,
            reshapedDLossDRenderedFeatures,
            reshapedDLossDRenderedAlphas,
            local.gradients[LocalGaussians::MEANS2D],
            local.gradients[LocalGaussians::CONICS],
            local.gradients[LocalGaussians::FEATURES],
            local.gradients[LocalGaussians::OPACITIES],
            absGrad ? std::make_optional(local.gradients[LocalGaussians::MEANS2D_ABS])
                    : std::nullopt,
            activeTiles,
            tilePixelMask,
            tilePixelCumsum,
            pixelMap);
        args.mGlobalToLocal = globalToLocal;
        if (backgrounds.has_value()) {
            args.commonArgs.mBackgrounds =
                initAccessor<ScalarType, 2>(backgrounds.value(), "backgrounds");
            args.commonArgs.mHasBackgrounds = true;
        }
        if (masks.has_value()) {
            args.commonArgs.mMasks    = initAccessor<bool, 3>(masks.value(), "masks");
            args.commonArgs.mHasMasks = true;
        }

        const size_t numChannels =
            (NUM_SHARED_CHANNELS == NUM_CHANNELS) ? NUM_CHANNELS : NUM_SHARED_CHANNELS + 1;
        const size_t sharedMemSize = getSharedMemRequirements<ScalarType>(numChannels, tileSize);

        if (cudaFuncSetAttribute(
                rasterizeGaussiansBackward<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                sharedMemSize) != cudaSuccess) {
            AT_ERROR("Failed to set maximum shared memory size (requested ",
                     sharedMemSize,
                     " bytes), try lowering tileSize.");
        }

        const dim3 blockDim = {tileSize, tileSize, 1};
        const dim3 gridDim  = {deviceTileCount, 1, 1};

        // Marking, inversion, gathering, rasterization, and frees use the same stream.
        // The tile cutoff ensures every loaded global ID has a populated local slot.
        rasterizeGaussiansBackward<ScalarType, NUM_CHANNELS, NUM_SHARED_CHANNELS, true>
            <<<gridDim, blockDim, sharedMemSize, stream>>>(args);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        local.globalToLocal = torch::Tensor();
    }

    // Scatter-add reductions wait for output prefetching after rasterization has been submitted.
    for (const auto deviceId: c10::irange(c10::cuda::device_count())) {
        C10_CUDA_CHECK(cudaSetDevice(deviceId));
        auto stream = c10::cuda::getCurrentCUDAStream(deviceId);
        C10_CUDA_CHECK(cudaStreamWaitEvent(stream, outputPrefetchEvents[deviceId]));
        C10_CUDA_CHECK(cudaEventDestroy(outputPrefetchEvents[deviceId]));
    }

    // Every local gradient producer must precede remote copies. Owners need only their own output
    // prefetch, which the loop above already ordered.
    mergeStreams();

    reduceScatterGradients<ScalarType>(
        gradientShards, shardOffsets, localGaussians, globalGradients);

    // Remote reads run on destination streams. Order each source stream after those reads
    // before its tensors enqueue frees, and expose completed output writes to every device.
    mergeStreams();
    localGaussians.clear();

    return std::make_tuple(outDLossDMeans2dAbs,
                           outDLossDMeans2d,
                           outDLossDConics,
                           outDLossDFeatures,
                           outDLossDOpacities);
}

} // namespace

template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeScreenSpaceGaussiansBwd(const torch::Tensor &means2d,
                                         const torch::Tensor &conics,
                                         const torch::Tensor &features,
                                         const torch::Tensor &opacities,
                                         const RenderWindow2D &renderWindow,
                                         const uint32_t tileSize,
                                         const torch::Tensor &tileOffsets,
                                         const torch::Tensor &tileGaussianIds,
                                         const torch::Tensor &renderedAlphas,
                                         const torch::Tensor &lastIds,
                                         const torch::Tensor &dLossDRenderedFeatures,
                                         const torch::Tensor &dLossDRenderedAlphas,
                                         const bool absGrad,
                                         const int64_t numSharedChannelsOverride,
                                         const at::optional<torch::Tensor> &backgrounds,
                                         const at::optional<torch::Tensor> &masks);

template <torch::DeviceType>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeScreenSpaceGaussiansSparseBwd(const fvdb::JaggedTensor &pixelsToRender,
                                               const torch::Tensor &means2d,
                                               const torch::Tensor &conics,
                                               const torch::Tensor &features,
                                               const torch::Tensor &opacities,
                                               const RenderWindow2D &renderWindow,
                                               const uint32_t tileSize,
                                               const torch::Tensor &tileOffsets,
                                               const torch::Tensor &tileGaussianIds,
                                               const fvdb::JaggedTensor &renderedAlphas,
                                               const fvdb::JaggedTensor &lastIds,
                                               const fvdb::JaggedTensor &dLossDRenderedFeatures,
                                               const fvdb::JaggedTensor &dLossDRenderedAlphas,
                                               const torch::Tensor &activeTiles,
                                               const torch::Tensor &tilePixelMask,
                                               const torch::Tensor &tilePixelCumsum,
                                               const torch::Tensor &pixelMap,
                                               const bool absGrad,
                                               const int64_t numSharedChannelsOverride,
                                               const at::optional<torch::Tensor> &backgrounds,
                                               const at::optional<torch::Tensor> &masks);

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeScreenSpaceGaussiansBwd<torch::kCUDA>(
    const torch::Tensor &means2d,                // [C, N, 2]
    const torch::Tensor &conics,                 // [C, N, 3]
    const torch::Tensor &features,               // [C, N, 3]
    const torch::Tensor &opacities,              // [N]
    const RenderWindow2D &renderWindow,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,            // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,        // [totalIntersections]
    const torch::Tensor &renderedAlphas,         // [C, renderHeight, renderWidth, 1]
    const torch::Tensor &lastGaussianIds,        // [C, renderHeight, renderWidth]
    const torch::Tensor &dLossDRenderedFeatures, // [C, renderHeight, renderWidth, 3]
    const torch::Tensor &dLossDRenderedAlphas,   // [C, renderHeight, renderWidth, 1]
    const bool absGrad,
    const int64_t numSharedChannelsOverride,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(means2d));
    uint32_t colorDim   = features.size(-1);
    const bool isPacked = means2d.dim() == 2;

#define CALL_BWD_CUDA(N)                                                            \
    case N: {                                                                       \
        if (isPacked) {                                                             \
            return callRasterizeBackwardWithCorrectSharedChannels<float, N, true>(  \
                means2d,                                                            \
                conics,                                                             \
                features,                                                           \
                opacities,                                                          \
                backgrounds,                                                        \
                masks,                                                              \
                renderWindow,                                                       \
                tileSize,                                                           \
                tileOffsets,                                                        \
                tileGaussianIds,                                                    \
                renderedAlphas,                                                     \
                lastGaussianIds,                                                    \
                dLossDRenderedFeatures,                                             \
                dLossDRenderedAlphas,                                               \
                absGrad,                                                            \
                numSharedChannelsOverride);                                         \
        } else {                                                                    \
            return callRasterizeBackwardWithCorrectSharedChannels<float, N, false>( \
                means2d,                                                            \
                conics,                                                             \
                features,                                                           \
                opacities,                                                          \
                backgrounds,                                                        \
                masks,                                                              \
                renderWindow,                                                       \
                tileSize,                                                           \
                tileOffsets,                                                        \
                tileGaussianIds,                                                    \
                renderedAlphas,                                                     \
                lastGaussianIds,                                                    \
                dLossDRenderedFeatures,                                             \
                dLossDRenderedAlphas,                                               \
                absGrad,                                                            \
                numSharedChannelsOverride);                                         \
        }                                                                           \
    }

    switch (colorDim) {
        CALL_BWD_CUDA(1)
        CALL_BWD_CUDA(2)
        CALL_BWD_CUDA(3)
        CALL_BWD_CUDA(4)
        CALL_BWD_CUDA(5)
        CALL_BWD_CUDA(8)
        CALL_BWD_CUDA(9)
        CALL_BWD_CUDA(16)
        CALL_BWD_CUDA(17)
        CALL_BWD_CUDA(32)
        CALL_BWD_CUDA(33)
        CALL_BWD_CUDA(47) // TODO, is this only here to support a gtest?
        CALL_BWD_CUDA(64)
        CALL_BWD_CUDA(65)
        CALL_BWD_CUDA(128)
        CALL_BWD_CUDA(129)
        CALL_BWD_CUDA(192)
        CALL_BWD_CUDA(193)
        CALL_BWD_CUDA(256)
        CALL_BWD_CUDA(257)
        CALL_BWD_CUDA(512)
        CALL_BWD_CUDA(513)
    default: AT_ERROR("Unsupported number of channels: ", colorDim);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeScreenSpaceGaussiansBwd<torch::kPrivateUse1>(
    const torch::Tensor &means2d,                // [C, N, 2]
    const torch::Tensor &conics,                 // [C, N, 3]
    const torch::Tensor &features,               // [C, N, 3]
    const torch::Tensor &opacities,              // [N]
    const RenderWindow2D &renderWindow,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,            // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,        // [totalIntersections]
    const torch::Tensor &renderedAlphas,         // [C, renderHeight, renderWidth, 1]
    const torch::Tensor &lastGaussianIds,        // [C, renderHeight, renderWidth]
    const torch::Tensor &dLossDRenderedFeatures, // [C, renderHeight, renderWidth, 3]
    const torch::Tensor &dLossDRenderedAlphas,   // [C, renderHeight, renderWidth, 1]
    const bool absGrad,
    const int64_t numSharedChannelsOverride,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    TORCH_CHECK(numSharedChannelsOverride == -1,
                "PrivateUse1 implementation does not support shared channels override");
    uint32_t colorDim   = features.size(-1);
    const bool isPacked = means2d.dim() == 2;

#define CALL_BWD_PRIVATEUSE1(N)                                                                 \
    case N: {                                                                                   \
        if (isPacked) {                                                                         \
            return callRasterizeBackwardPrivateUse1<float, N, N, true>(means2d,                 \
                                                                       conics,                  \
                                                                       features,                \
                                                                       opacities,               \
                                                                       backgrounds,             \
                                                                       masks,                   \
                                                                       renderWindow,            \
                                                                       tileSize,                \
                                                                       tileOffsets,             \
                                                                       tileGaussianIds,         \
                                                                       renderedAlphas,          \
                                                                       lastGaussianIds,         \
                                                                       dLossDRenderedFeatures,  \
                                                                       dLossDRenderedAlphas,    \
                                                                       absGrad);                \
        } else {                                                                                \
            return callRasterizeBackwardPrivateUse1<float, N, N, false>(means2d,                \
                                                                        conics,                 \
                                                                        features,               \
                                                                        opacities,              \
                                                                        backgrounds,            \
                                                                        masks,                  \
                                                                        renderWindow,           \
                                                                        tileSize,               \
                                                                        tileOffsets,            \
                                                                        tileGaussianIds,        \
                                                                        renderedAlphas,         \
                                                                        lastGaussianIds,        \
                                                                        dLossDRenderedFeatures, \
                                                                        dLossDRenderedAlphas,   \
                                                                        absGrad);               \
        }                                                                                       \
    }

    switch (colorDim) {
        CALL_BWD_PRIVATEUSE1(1)
        CALL_BWD_PRIVATEUSE1(2)
        CALL_BWD_PRIVATEUSE1(3)
        CALL_BWD_PRIVATEUSE1(4)
        CALL_BWD_PRIVATEUSE1(5)
        CALL_BWD_PRIVATEUSE1(8)
        CALL_BWD_PRIVATEUSE1(9)
        CALL_BWD_PRIVATEUSE1(16)
        CALL_BWD_PRIVATEUSE1(17)
        CALL_BWD_PRIVATEUSE1(32)
        CALL_BWD_PRIVATEUSE1(33)
        CALL_BWD_PRIVATEUSE1(47) // TODO, is this only here to support a gtest?
        CALL_BWD_PRIVATEUSE1(64)
        CALL_BWD_PRIVATEUSE1(65)
        CALL_BWD_PRIVATEUSE1(128)
        CALL_BWD_PRIVATEUSE1(129)
        CALL_BWD_PRIVATEUSE1(192)
        CALL_BWD_PRIVATEUSE1(193)
        CALL_BWD_PRIVATEUSE1(256)
        CALL_BWD_PRIVATEUSE1(257)
        CALL_BWD_PRIVATEUSE1(512)
        CALL_BWD_PRIVATEUSE1(513)
    default: AT_ERROR("Unsupported number of channels: ", colorDim);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeScreenSpaceGaussiansBwd<torch::kCPU>(
    const torch::Tensor &means2d,                // [C, N, 2]
    const torch::Tensor &conics,                 // [C, N, 3]
    const torch::Tensor &features,               // [C, N, 3]
    const torch::Tensor &opacities,              // [N]
    const RenderWindow2D &renderWindow,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets,            // [C, numTilesH, numTilesW]
    const torch::Tensor &tileGaussianIds,        // [totalIntersections]
    const torch::Tensor &renderedAlphas,         // [C, renderHeight, renderWidth, 1]
    const torch::Tensor &lastGaussianIds,        // [C, renderHeight, renderWidth]
    const torch::Tensor &dLossDRenderedFeatures, // [C, renderHeight, renderWidth, 3]
    const torch::Tensor &dLossDRenderedAlphas,   // [C, renderHeight, renderWidth, 1]
    const bool absGrad,
    const int64_t numSharedChannelsOverride,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeScreenSpaceGaussiansSparseBwd<torch::kCUDA>(
    const fvdb::JaggedTensor &pixelsToRender, // [C, NumPixels, 2]
    const torch::Tensor &means2d,             // [C, N, 2]
    const torch::Tensor &conics,              // [C, N, 3]
    const torch::Tensor &features,            // [C, N, D]
    const torch::Tensor &opacities,           // [N]
    const RenderWindow2D &renderWindow,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets, // [C, tile_height, tile_width] (dense) or [AT + 1] (sparse)
    const torch::Tensor &tileGaussianIds,             // [n_isects]
    const fvdb::JaggedTensor &renderedAlphas,         // [C lists: varying sizes, each element [1]]
    const fvdb::JaggedTensor &lastGaussianIds,        // [C lists: varying sizes]
    const fvdb::JaggedTensor &dLossDRenderedFeatures, // [C lists: varying sizes, each element [D]]
    const fvdb::JaggedTensor &dLossDRenderedAlphas,   // [C lists: varying sizes, each element [1]]
    const torch::Tensor &activeTiles,                 // [AT]
    const torch::Tensor &tilePixelMask,               // [AT, wordsPerTile]
    const torch::Tensor &tilePixelCumsum,             // [AT]
    const torch::Tensor &pixelMap,                    // [AP]
    const bool absGrad,
    const int64_t numSharedChannelsOverride,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    uint32_t colorDim   = features.size(-1);
    const bool isPacked = means2d.dim() == 2;

    // Validate the jagged camera structure before passing only the flat pixel data to the
    // launch helpers. Packed Gaussians do not encode the camera count, so use renderedAlphas.
    if (means2d.numel() != 0 && tileGaussianIds.numel() != 0) {
        const auto numSparseCameras = renderedAlphas.num_tensors();
        if (!isPacked) {
            TORCH_CHECK_VALUE(means2d.size(0) == numSparseCameras, "Bad size for renderedAlphas");
        }
        TORCH_CHECK_VALUE(numSparseCameras == lastGaussianIds.num_tensors(),
                          "Bad size for lastGaussianIds");
        TORCH_CHECK_VALUE(numSparseCameras == dLossDRenderedFeatures.num_tensors(),
                          "Bad size for dLossDRenderedFeatures");
        TORCH_CHECK_VALUE(numSparseCameras == dLossDRenderedAlphas.num_tensors(),
                          "Bad size for dLossDRenderedAlphas");
    }

#define CALL_BWD_SPARSE_CUDA(N)                                                     \
    case N: {                                                                       \
        if (isPacked) {                                                             \
            return callRasterizeBackwardWithCorrectSharedChannels<float, N, true>(  \
                means2d,                                                            \
                conics,                                                             \
                features,                                                           \
                opacities,                                                          \
                backgrounds,                                                        \
                masks,                                                              \
                renderWindow,                                                       \
                tileSize,                                                           \
                tileOffsets,                                                        \
                tileGaussianIds,                                                    \
                renderedAlphas.jdata(),                                             \
                lastGaussianIds.jdata(),                                            \
                dLossDRenderedFeatures.jdata(),                                     \
                dLossDRenderedAlphas.jdata(),                                       \
                absGrad,                                                            \
                numSharedChannelsOverride,                                          \
                activeTiles,                                                        \
                tilePixelMask,                                                      \
                tilePixelCumsum,                                                    \
                pixelMap);                                                          \
        } else {                                                                    \
            return callRasterizeBackwardWithCorrectSharedChannels<float, N, false>( \
                means2d,                                                            \
                conics,                                                             \
                features,                                                           \
                opacities,                                                          \
                backgrounds,                                                        \
                masks,                                                              \
                renderWindow,                                                       \
                tileSize,                                                           \
                tileOffsets,                                                        \
                tileGaussianIds,                                                    \
                renderedAlphas.jdata(),                                             \
                lastGaussianIds.jdata(),                                            \
                dLossDRenderedFeatures.jdata(),                                     \
                dLossDRenderedAlphas.jdata(),                                       \
                absGrad,                                                            \
                numSharedChannelsOverride,                                          \
                activeTiles,                                                        \
                tilePixelMask,                                                      \
                tilePixelCumsum,                                                    \
                pixelMap);                                                          \
        }                                                                           \
    }

    switch (colorDim) {
        CALL_BWD_SPARSE_CUDA(1)
        CALL_BWD_SPARSE_CUDA(2)
        CALL_BWD_SPARSE_CUDA(3)
        CALL_BWD_SPARSE_CUDA(4)
        CALL_BWD_SPARSE_CUDA(5)
        CALL_BWD_SPARSE_CUDA(8)
        CALL_BWD_SPARSE_CUDA(9)
        CALL_BWD_SPARSE_CUDA(16)
        CALL_BWD_SPARSE_CUDA(17)
        CALL_BWD_SPARSE_CUDA(32)
        CALL_BWD_SPARSE_CUDA(33)
        CALL_BWD_SPARSE_CUDA(47)
        CALL_BWD_SPARSE_CUDA(64)
        CALL_BWD_SPARSE_CUDA(65)
        CALL_BWD_SPARSE_CUDA(128)
        CALL_BWD_SPARSE_CUDA(129)
        CALL_BWD_SPARSE_CUDA(192)
        CALL_BWD_SPARSE_CUDA(193)
        CALL_BWD_SPARSE_CUDA(256)
        CALL_BWD_SPARSE_CUDA(257)
        CALL_BWD_SPARSE_CUDA(512)
        CALL_BWD_SPARSE_CUDA(513)
    default: AT_ERROR("Unsupported number of channels: ", colorDim);
    }
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeScreenSpaceGaussiansSparseBwd<torch::kPrivateUse1>(
    const fvdb::JaggedTensor &pixelsToRender, // [C, NumPixels, 2]
    const torch::Tensor &means2d,             // [C, N, 2]
    const torch::Tensor &conics,              // [C, N, 3]
    const torch::Tensor &features,            // [C, N, D]
    const torch::Tensor &opacities,           // [N]
    const RenderWindow2D &renderWindow,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets, // [C, tile_height, tile_width] (dense) or [AT + 1] (sparse)
    const torch::Tensor &tileGaussianIds,             // [n_isects]
    const fvdb::JaggedTensor &renderedAlphas,         // [C lists: varying sizes, each element [1]]
    const fvdb::JaggedTensor &lastIds,                // [C lists: varying sizes]
    const fvdb::JaggedTensor &dLossDRenderedFeatures, // [C lists: varying sizes, each element [D]]
    const fvdb::JaggedTensor &dLossDRenderedAlphas,   // [C lists: varying sizes, each element [1]]
    const torch::Tensor &activeTiles,                 // [AT]
    const torch::Tensor &tilePixelMask,               // [AT, wordsPerTile]
    const torch::Tensor &tilePixelCumsum,             // [AT]
    const torch::Tensor &pixelMap,                    // [AP]
    const bool absGrad,
    const int64_t numSharedChannelsOverride,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "PrivateUse1 implementation not available");
}

template <>
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
dispatchRasterizeScreenSpaceGaussiansSparseBwd<torch::kCPU>(
    const fvdb::JaggedTensor &pixelsToRender, // [C, NumPixels, 2]
    const torch::Tensor &means2d,             // [C, N, 2]
    const torch::Tensor &conics,              // [C, N, 3]
    const torch::Tensor &features,            // [C, N, D]
    const torch::Tensor &opacities,           // [N]
    const RenderWindow2D &renderWindow,
    const uint32_t tileSize,
    const torch::Tensor &tileOffsets, // [C, tile_height, tile_width] (dense) or [AT + 1] (sparse)
    const torch::Tensor &tileGaussianIds,             // [n_isects]
    const fvdb::JaggedTensor &renderedAlphas,         // [C lists: varying sizes, each element [1]]
    const fvdb::JaggedTensor &lastIds,                // [C lists: varying sizes]
    const fvdb::JaggedTensor &dLossDRenderedFeatures, // [C lists: varying sizes, each element [D]]
    const fvdb::JaggedTensor &dLossDRenderedAlphas,   // [C lists: varying sizes, each element [1]]
    const torch::Tensor &activeTiles,                 // [AT]
    const torch::Tensor &tilePixelMask,               // [AT, wordsPerTile]
    const torch::Tensor &tilePixelCumsum,             // [AT]
    const torch::Tensor &pixelMap,                    // [AP]
    const bool absGrad,
    const int64_t numSharedChannelsOverride,
    const at::optional<torch::Tensor> &backgrounds,
    const at::optional<torch::Tensor> &masks) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, "CPU implementation not available");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeScreenSpaceGaussiansBwd(const torch::Tensor &means2d,
                                 const torch::Tensor &conics,
                                 const torch::Tensor &features,
                                 const torch::Tensor &opacities,
                                 const uint32_t imageWidth,
                                 const uint32_t imageHeight,
                                 const uint32_t imageOriginW,
                                 const uint32_t imageOriginH,
                                 const uint32_t tileSize,
                                 const torch::Tensor &tileOffsets,
                                 const torch::Tensor &tileGaussianIds,
                                 const torch::Tensor &renderedAlphas,
                                 const torch::Tensor &lastIds,
                                 const torch::Tensor &dLossDRenderedFeatures,
                                 const torch::Tensor &dLossDRenderedAlphas,
                                 const bool absGrad,
                                 const int64_t numSharedChannelsOverride,
                                 const at::optional<torch::Tensor> &backgrounds,
                                 const at::optional<torch::Tensor> &masks) {
    TORCH_CHECK_VALUE(tileOffsets.scalar_type() == torch::kInt64,
                      "tileOffsets must have dtype int64");
    TORCH_CHECK_VALUE(lastIds.scalar_type() == torch::kInt32, "lastIds must have dtype int32");
    const RenderWindow2D renderWindow{imageWidth, imageHeight, imageOriginW, imageOriginH};
    return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return dispatchRasterizeScreenSpaceGaussiansBwd<DeviceTag>(means2d,
                                                                   conics,
                                                                   features,
                                                                   opacities,
                                                                   renderWindow,
                                                                   tileSize,
                                                                   tileOffsets,
                                                                   tileGaussianIds,
                                                                   renderedAlphas,
                                                                   lastIds,
                                                                   dLossDRenderedFeatures,
                                                                   dLossDRenderedAlphas,
                                                                   absGrad,
                                                                   numSharedChannelsOverride,
                                                                   backgrounds,
                                                                   masks);
    });
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterizeScreenSpaceGaussiansSparseBwd(const fvdb::JaggedTensor &pixelsToRender,
                                       const torch::Tensor &means2d,
                                       const torch::Tensor &conics,
                                       const torch::Tensor &features,
                                       const torch::Tensor &opacities,
                                       const uint32_t imageWidth,
                                       const uint32_t imageHeight,
                                       const uint32_t imageOriginW,
                                       const uint32_t imageOriginH,
                                       const uint32_t tileSize,
                                       const torch::Tensor &tileOffsets,
                                       const torch::Tensor &tileGaussianIds,
                                       const fvdb::JaggedTensor &renderedAlphas,
                                       const fvdb::JaggedTensor &lastIds,
                                       const fvdb::JaggedTensor &dLossDRenderedFeatures,
                                       const fvdb::JaggedTensor &dLossDRenderedAlphas,
                                       const torch::Tensor &activeTiles,
                                       const torch::Tensor &tilePixelMask,
                                       const torch::Tensor &tilePixelCumsum,
                                       const torch::Tensor &pixelMap,
                                       const bool absGrad,
                                       const int64_t numSharedChannelsOverride,
                                       const at::optional<torch::Tensor> &backgrounds,
                                       const at::optional<torch::Tensor> &masks) {
    TORCH_CHECK_VALUE(tileOffsets.scalar_type() == torch::kInt64,
                      "tileOffsets must have dtype int64");
    TORCH_CHECK_VALUE(lastIds.jdata().scalar_type() == torch::kInt32,
                      "lastIds must have dtype int32");
    const RenderWindow2D renderWindow{imageWidth, imageHeight, imageOriginW, imageOriginH};
    return FVDB_DISPATCH_KERNEL(means2d.device(), [&]() {
        return dispatchRasterizeScreenSpaceGaussiansSparseBwd<DeviceTag>(pixelsToRender,
                                                                         means2d,
                                                                         conics,
                                                                         features,
                                                                         opacities,
                                                                         renderWindow,
                                                                         tileSize,
                                                                         tileOffsets,
                                                                         tileGaussianIds,
                                                                         renderedAlphas,
                                                                         lastIds,
                                                                         dLossDRenderedFeatures,
                                                                         dLossDRenderedAlphas,
                                                                         activeTiles,
                                                                         tilePixelMask,
                                                                         tilePixelCumsum,
                                                                         pixelMap,
                                                                         absGrad,
                                                                         numSharedChannelsOverride,
                                                                         backgrounds,
                                                                         masks);
    });
}

} // namespace fvdb::detail::ops
