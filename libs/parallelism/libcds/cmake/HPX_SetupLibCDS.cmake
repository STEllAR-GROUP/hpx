# Copyright (c) 2020 Weile Wei
# Copyright (c) 2020 John Biddiscombe
# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_LIBCDS_WITH_LIBCDS AND NOT TARGET LibCDS::cds)
  if(NOT HPX_FIND_PACKAGE)

    if(LIBCDS_DIR)
      hpx_info("    LIBCDS_DIR is set. Attempting to find LibCDS.")
      find_package(LibCDS REQUIRED)

    else()
      include(FetchContent)

      set(LIBCDS_GENERATE_SOURCELIST ON)

      set(LIBCDS_WITH_HPX ON)
      set(LIBCDS_AS_HPX_MODULE ON)

      set(FETCHCONTENT_UPDATES_DISCONNECTED_libcds ON)

      hpx_info(
        "    Fetching libCDS from repository: ${HPX_LIBCDS_WITH_GIT_REPOSITORY}, "
        "tag: ${HPX_LIBCDS_WITH_GIT_TAG}"
      )
      fetchcontent_declare(
        libcds
        GIT_REPOSITORY ${HPX_LIBCDS_WITH_GIT_REPOSITORY}
        GIT_TAG ${HPX_LIBCDS_WITH_GIT_TAG}
        GIT_SHALLOW TRUE
      )
      fetchcontent_getproperties(libcds)

      if(NOT libcds_POPULATED)
        fetchcontent_populate(libcds)
        hpx_info(
          "    LIBCDS_DIR is not set. Cloning LibCDS into ${libcds_SOURCE_DIR}."
        )
      endif()

      set(LIBCDS_ROOT ${libcds_SOURCE_DIR})

      add_library(LibCDS::cds INTERFACE IMPORTED)

      set(LIBCDS_CXX_STANDARD ${HPX_CXX_STANDARD})
      add_subdirectory(${libcds_SOURCE_DIR} ${libcds_BINARY_DIR})
      list(TRANSFORM LIBCDS_SOURCELIST PREPEND "${libcds_SOURCE_DIR}/")

      target_link_libraries(LibCDS::cds INTERFACE cds)
      if(MSVC)
        target_compile_definitions(
          LibCDS::cds INTERFACE CDS_CXX11_THREAD_LOCAL_SUPPORT=
        )
      endif()

      set_target_properties(cds PROPERTIES FOLDER "Core/Dependencies/libCDS")
      set_target_properties(cds-s PROPERTIES FOLDER "Core/Dependencies/libCDS")

    endif()
  endif()
endif()
