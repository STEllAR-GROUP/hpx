# Copyright (c) 2020      Weile Wei
# Copyright (c) 2020      John Biddiscombe
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(FETCHCONTENT_UPDATES_DISCONNECTED_libcds ON)

if(HPX_WITH_LIBCDS)
  include(FetchContent)

  set(LIBCDS_GENERATE_SOURCELIST ON)

  set(LIBCDS_WITH_HPX ON)
  set(LIBCDS_AS_HPX_MODULE ON)

  hpx_info(
    "Fetching libCDS from repository: ${HPX_WITH_LIBCDS_GIT_REPOSITORY}, "
    "tag: ${HPX_WITH_LIBCDS_GIT_TAG}"
  )
  fetchcontent_declare(
    libcds
    # GIT_REPOSITORY https://github.com/khizmax/libcds
    GIT_REPOSITORY https://github.com/weilewei/libcds
    GIT_TAG hpx-thread
    GIT_SHALLOW TRUE
  )
  fetchcontent_getproperties(libcds)

  if(NOT libcds_POPULATED)
    fetchcontent_populate(libcds)
    set(LIBCDS_CXX_STANDARD ${HPX_CXX_STANDARD})
    add_subdirectory(${libcds_SOURCE_DIR} ${libcds_BINARY_DIR})
    list(TRANSFORM LIBCDS_SOURCELIST PREPEND "${libcds_SOURCE_DIR}/")
    set(LIBCDS_SOURCE_DIR ${libcds_SOURCE_DIR})
    # leave the FOLDER properties in place set_target_properties(cds PROPERTIES
    # FOLDER "Core") set_target_properties(cds-s PROPERTIES FOLDER "Core")

    # create an imported target that links to the real libcds so that when we
    # link to the imported target, we don't get export X depends on cds that is
    # not in the export set
    add_library(libcds::cds INTERFACE IMPORTED)
    target_link_libraries(libcds::cds INTERFACE cds)
  endif()

endif()
