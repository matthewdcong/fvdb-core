// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0

#include <fvdb/detail/utils/gsplat/CameraTilePartition.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

struct PartitionCase {
    uint32_t tilesPerCamera;
    uint32_t deviceCount;
    std::vector<std::vector<uint32_t>> cameraOwners;
};

class CameraTilePartitionTest : public ::testing::TestWithParam<PartitionCase> {};

TEST_P(CameraTilePartitionTest, CoversTilesWithExpectedCameraOwners) {
    const auto &[tilesPerCamera, deviceCount, expectedOwners] = GetParam();
    const uint32_t tileCount      = expectedOwners.size() * tilesPerCamera;
    const uint32_t tilesPerDevice = (tileCount + deviceCount - 1) / deviceCount;
    std::vector<std::vector<uint32_t>> actualOwners(expectedOwners.size());

    for (uint32_t deviceId = 0; deviceId < deviceCount; ++deviceId) {
        SCOPED_TRACE(deviceId);
        // Forward assigns ceil(total tiles / devices) consecutive tiles to each device.
        const uint32_t begin = std::min(deviceId * tilesPerDevice, tileCount);
        const uint32_t count = std::min(tilesPerDevice, tileCount - begin);
        const auto segments  = fvdb::detail::partitionCameraTiles(begin, count, tilesPerCamera);
        ASSERT_LE(segments.size(), 3u);
        uint32_t nextTile = begin;
        for (const auto &segment: segments) {
            EXPECT_EQ(segment.tileOffset, nextTile);
            ASSERT_GT(segment.tileCount, 0u);
            ASSERT_GT(segment.cameraCount, 0u);
            ASSERT_LE(segment.cameraOffset + segment.cameraCount, expectedOwners.size());
            EXPECT_EQ(segment.cameraOffset, segment.tileOffset / tilesPerCamera);
            EXPECT_EQ(segment.cameraOffset + segment.cameraCount - 1,
                      (segment.tileOffset + segment.tileCount - 1) / tilesPerCamera);
            if (segment.shared) {
                EXPECT_EQ(segment.cameraCount, 1u);
                EXPECT_LT(segment.tileCount, tilesPerCamera);
            } else {
                EXPECT_EQ(segment.tileOffset, segment.cameraOffset * tilesPerCamera);
                EXPECT_EQ(segment.tileCount, segment.cameraCount * tilesPerCamera);
            }
            for (uint32_t camera = segment.cameraOffset;
                 camera < segment.cameraOffset + segment.cameraCount;
                 ++camera) {
                EXPECT_EQ(segment.shared, expectedOwners[camera].size() > 1);
                actualOwners[camera].push_back(deviceId);
            }
            nextTile += segment.tileCount;
        }
        EXPECT_EQ(nextTile, begin + count);
    }
    EXPECT_EQ(actualOwners, expectedOwners);
}

INSTANTIATE_TEST_SUITE_P(
    CameraOwnership,
    CameraTilePartitionTest,
    ::testing::Values(PartitionCase{4, 2, {{0, 1}}},
                      PartitionCase{4, 4, {{0, 1}, {2, 3}}},
                      PartitionCase{4, 4, {{0}, {0, 1}, {1}, {2}, {2, 3}, {3}}},
                      PartitionCase{4, 4, {{0}, {1}, {2}, {3}}},
                      PartitionCase{2, 4, {{0, 1}}},
                      PartitionCase{1, 4, {{0}}},
                      PartitionCase{4, 3, {{0}, {0}, {0, 1}, {1}, {1}, {1, 2}, {2}, {2}}},
                      PartitionCase{5, 4, {{0, 1}, {1, 2}, {2, 3}}},
                      PartitionCase{4, 2, {}}));

TEST(CameraTilePartitionValidationTest, RejectsInvalidTileRanges) {
    EXPECT_THROW(fvdb::detail::partitionCameraTiles(0, 1, 0), c10::Error);
    EXPECT_THROW(fvdb::detail::partitionCameraTiles(std::numeric_limits<uint32_t>::max(), 1, 4),
                 c10::Error);
}

} // namespace
