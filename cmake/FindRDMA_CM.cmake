# Copyright (c)      2014 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#
# - Try to find RDMA CM
# Once done this will define
#  Rdma_CM_FOUND - System has RDMA CM
#  Rdma_CM_INCLUDE_DIRS - The RDMA CM include directories
#  Rdma_CM_LIBRARIES - The libraries needed to use RDMA CM

find_path(Rdma_CM_INCLUDE_DIR rdma_cma.h HINTS /usr/local/include
                                               /usr/include/rdma
)

find_library(
  Rdma_CM_LIBRARY
  NAMES rdmacm
  PATHS /usr/local/lib /usr/lib
)

set(Rdma_CM_INCLUDE_DIRS ${Rdma_CM_INCLUDE_DIR})
set(Rdma_CM_LIBRARIES ${Rdma_CM_LIBRARY})

include(FindPackageHandleStandardArgs)

# handle the QUIETLY and REQUIRED arguments and set Rdma_CM_FOUND to TRUE if all
# listed variables are TRUE
find_package_handle_standard_args(
  Rdma_CM DEFAULT_MSG Rdma_CM_INCLUDE_DIR Rdma_CM_LIBRARY
)

mark_as_advanced(Rdma_CM_INCLUDE_DIR Rdma_CM_LIBRARY)
