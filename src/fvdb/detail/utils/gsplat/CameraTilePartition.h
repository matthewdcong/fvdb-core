// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_UTILS_GSPLAT_CAMERATILEPARTITION_H
#define FVDB_DETAIL_UTILS_GSPLAT_CAMERATILEPARTITION_H

#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace fvdb::detail {

struct CameraTileSegment {
    uint32_t tileOffset;
    uint32_t tileCount;
    uint32_t cameraOffset;
    uint32_t cameraCount;
    bool shared;
};

// Split a device's camera-major tile interval into at most two shared boundary cameras
// and one run of fully owned cameras. The input interval comes from the forward tile partition.
inline c10::SmallVector<CameraTileSegment, 3>
partitionCameraTiles(uint32_t tileOffset, uint32_t tileCount, uint32_t tilesPerCamera) {
    TORCH_CHECK(tilesPerCamera > 0, "Camera tile partition requires tiles per camera > 0");
    TORCH_CHECK(uint64_t{tileOffset} + tileCount <= std::numeric_limits<uint32_t>::max(),
                "Camera tile interval exceeds uint32_t range");
    c10::SmallVector<CameraTileSegment, 3> segments;
    const uint32_t end = tileOffset + tileCount;
    uint32_t begin     = tileOffset;
    while (begin < end) {
        const uint32_t camera = begin / tilesPerCamera;
        const uint32_t wholeCameras =
            begin % tilesPerCamera == 0 ? (end - begin) / tilesPerCamera : 0;
        const bool shared    = wholeCameras == 0;
        const uint32_t count = shared
                                   ? std::min(end - begin, tilesPerCamera - begin % tilesPerCamera)
                                   : wholeCameras * tilesPerCamera;
        segments.push_back({begin, count, camera, shared ? 1 : wholeCameras, shared});
        begin += count;
    }
    return segments;
}

} // namespace fvdb::detail

#endif // FVDB_DETAIL_UTILS_GSPLAT_CAMERATILEPARTITION_H
