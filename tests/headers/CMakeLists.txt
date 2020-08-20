# Copyright (c) 2015-2019 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# These files can't be included at the same time.
set(exclude_from_all_headers)

# These files are either known to expose incomplete types or are known to
# require special handling to avoid circular #include dependencies.
set(exclude_from_headers)

if(HPX_WITH_NETWORKING AND HPX_WITH_PARCELPORT_MPI)
  set(test_deps Mpi::mpi)
endif()

add_hpx_header_tests(
  "core"
  HEADER_ROOT ${PROJECT_SOURCE_DIR}
  HEADERS ${hpx_HEADERS}
  EXCLUDE ${exclude_from_headers}
  EXCLUDE_FROM_ALL ${exclude_from_all_headers}
  DEPENDENCIES ${test_deps}
)
