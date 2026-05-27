# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0

# Download test data from github
CPMAddPackage(
  NAME fvdb_test_data
  GITHUB_REPOSITORY voxel-foundation/fvdb-test-data
  GIT_TAG 1735a8b7713b8a23a32248b3a4c681697bcec0ba
  DOWNLOAD_ONLY YES
)

if(fvdb_test_data_ADDED)
  set(FVDB_TEST_DATA_DIR "${CPM_PACKAGE_fvdb_test_data_SOURCE_DIR}")
  message(STATUS "Downloaded test data to: ${FVDB_TEST_DATA_DIR}")
else()
  unset(FVDB_TEST_DATA_DIR)
  message(FATAL_ERROR "Failed to download test data")
endif()
