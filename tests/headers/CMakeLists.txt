# Copyright (c) 2015-2018 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# collect a list of all header files in the source tree
file(GLOB_RECURSE headers ${DO_CONFIGURE_DEPENDS} "${PROJECT_SOURCE_DIR}/hpx/*hpp")

# These files can't be included at the same time.
set(exclude_from_all_headers)

# These files are either known to expose incomplete types or are known to
# require special handling to avoid circular #include dependencies.
set(exclude_from_headers
  "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/timed_execution_fwd.hpp"
  "${PROJECT_SOURCE_DIR}/hpx/parallel/executors/parallel_executor.hpp"
  "${PROJECT_SOURCE_DIR}/hpx/lcos/local/futures_factory.hpp"
)

# If we compile with the MPI parcelport enabled, we need to additionally
# add the MPI include path here, because for the main library, it's only
# added for the plugin.
if(HPX_WITH_PARCELPORT_MPI)
  find_package(MPI)

  if(MPI_CXX_INCLUDE_PATH)
    include_directories(${MPI_CXX_INCLUDE_PATH})
  endif()
endif()

# If we compile with PAPI performance counters, we need to additionally
# add the PAPI include path here
if(HPX_WITH_PAPI)
  find_package(PAPI)
  include_directories(${PAPI_INCLUDE_DIR})
endif()

add_hpx_header_tests(
  "core"
  HEADER_ROOT ${PROJECT_SOURCE_DIR}
  HEADERS ${headers}
  EXCLUDE ${exclude_from_headers}
  EXCLUDE_FROM_ALL ${exclude_from_all_headers})
