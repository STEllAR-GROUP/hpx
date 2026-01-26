# Copyright (c) 2021-2023 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME : in the future put it directly inside the cmake directory of the
# corresponding plugin

macro(hpx_setup_lcw)
  # LCW always needs LCI
  include(HPX_SetupLCI)
  hpx_setup_lci()

  if(NOT TARGET LCW::LCW)

    # compatibility with older CMake versions
    if(LCW_ROOT AND NOT Lcw_ROOT)
      set(Lcw_ROOT
          ${LCW_ROOT}
          CACHE PATH "LCW base directory"
      )
      unset(LCW_ROOT CACHE)
    endif()

    if(NOT HPX_WITH_FETCH_LCW)
      find_package(
        LCW
        CONFIG
        REQUIRED
        HINTS
        ${Lcw_ROOT}
        $ENV{LCW_ROOT}
        PATH_SUFFIXES
        lib/cmake
        lib64/cmake
      )
    elseif(NOT HPX_FIND_PACKAGE)
      if(FETCHCONTENT_SOURCE_DIR_LCW)
        hpx_info(
          "HPX_WITH_FETCH_LCW=${HPX_WITH_FETCH_LCW}, LCW will be used through CMake's FetchContent and installed alongside HPX (FETCHCONTENT_SOURCE_DIR_LCW=${FETCHCONTENT_SOURCE_DIR_LCW})"
        )
      else()
        hpx_info(
          "HPX_WITH_FETCH_LCW=${HPX_WITH_FETCH_LCW}, LCW will be fetched using CMake's FetchContent and installed alongside HPX (HPX_WITH_LCW_TAG=${HPX_WITH_LCW_TAG})"
        )
      endif()
      include(FetchContent)
      fetchcontent_declare(
        lcw
        GIT_REPOSITORY https://github.com/JiakunYan/lcw
        GIT_TAG ${HPX_WITH_LCW_TAG}
      )

      fetchcontent_getproperties(lcw)
      if(NOT lcw_POPULATED)
        fetchcontent_populate(lcw)
        enable_language(CXX)
        add_subdirectory(${lcw_SOURCE_DIR} ${lcw_BINARY_DIR})
        # Move LCW target into its own FOLDER
        set_target_properties(lcw PROPERTIES FOLDER "Core/Dependencies")
        # add_library(LCW::LCW ALIAS lcw)
        set(HPX_CMAKE_ADDITIONAL_MODULE_PATH_BUILD
            "${lcw_SOURCE_DIR}/cmake_modules"
            CACHE INTERNAL ""
        )
      endif()

      install(
        TARGETS lcw
        EXPORT HPXLCWTarget
        COMPONENT core
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )

      install(
        DIRECTORY ${lcw_SOURCE_DIR}/lcw/api/ ${lcw_BINARY_DIR}/lcw/api/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT core
        FILES_MATCHING
        PATTERN "*.h"
      )

      export(
        TARGETS lcw
        NAMESPACE LCW::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXLCWTarget.cmake"
      )

      install(
        EXPORT HPXLCWTarget
        NAMESPACE LCW::
        FILE HPXLCWTarget.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
        COMPONENT cmake
      )
    endif()
  endif()
endmacro()
