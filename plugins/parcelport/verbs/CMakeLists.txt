# Copyright (c) 2014-2016 John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_SetupVerbs)

if(HPX_WITH_PARCELPORT_VERBS)
  hpx_debug("add_parcelport_verbs_module")

  include(HPX_AddParcelport)
  add_parcelport(
    verbs STATIC
    SOURCES
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/parcelport_verbs.cpp"
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/rdma/rdma_controller.cpp"
    # do we need to declare any of these headers?
    HEADERS
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/header.hpp"
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/locality.hpp"
      "${PROJECT_SOURCE_DIR}/plugins/parcelport/verbs/pinned_memory_vector.hpp"
    DEPENDENCIES hpx_threadmanager hpx_parallelism hpx_core Verbs::verbs
                 hpx_dependencies_boost
    INCLUDE_DIRS ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}
    FOLDER "Core/Plugins/Parcelport/verbs"
  )
endif()
