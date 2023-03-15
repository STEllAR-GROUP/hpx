# Copyright (c) 2021 Ste||ar Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# FIXME : in the future put it directly inside the cmake directory of the
# corresponding plugin

macro(hpx_setup_lci)
  if(NOT TARGET LCI::LCI)
    if(NOT HPX_WITH_FETCH_LCI)
      find_package(
        LCI
        CONFIG
        REQUIRED
        HINTS
        ${LCI_ROOT}
        $ENV{LCI_ROOT}
        PATH_SUFFIXES
        lib/cmake
        lib64/cmake
      )
    elseif(NOT HPX_FIND_PACKAGE)
      if(FETCHCONTENT_SOURCE_DIR_LCI)
        hpx_info(
          "HPX_WITH_FETCH_LCI=${HPX_WITH_FETCH_LCI}, LCI will be used through CMake's FetchContent and installed alongside HPX (FETCHCONTENT_SOURCE_DIR_LCI=${FETCHCONTENT_SOURCE_DIR_LCI})"
        )
      else()
        hpx_info(
          "HPX_WITH_FETCH_LCI=${HPX_WITH_FETCH_LCI}, LCI will be fetched using CMake's FetchContent and installed alongside HPX (HPX_WITH_LCI_TAG=${HPX_WITH_LCI_TAG})"
        )
      endif()
      include(FetchContent)
      fetchcontent_declare(
        lci
        GIT_REPOSITORY https://github.com/uiuc-hpc/LC.git
        GIT_TAG ${HPX_WITH_LCI_TAG}
      )

      fetchcontent_getproperties(lci)
      if(NOT lci_POPULATED)
        fetchcontent_populate(lci)
        if(NOT LCI_WITH_EXAMPLES)
          set(LCI_WITH_EXAMPLES
              OFF
              CACHE INTERNAL ""
          )
        endif()
        if(NOT LCI_WITH_TESTS)
          set(LCI_WITH_TESTS
              OFF
              CACHE INTERNAL ""
          )
        endif()
        if(NOT LCI_WITH_BENCHMARKS)
          set(LCI_WITH_BENCHMARKS
              OFF
              CACHE INTERNAL ""
          )
        endif()
        if(NOT LCI_WITH_DOC)
          set(LCI_WITH_DOC
              OFF
              CACHE INTERNAL ""
          )
        endif()
        enable_language(C)
        add_subdirectory(${lci_SOURCE_DIR} ${lci_BINARY_DIR})
        # Move LCI target into its own FOLDER
        set_target_properties(LCI PROPERTIES FOLDER "Core/Dependencies")
        add_library(LCI::LCI ALIAS LCI)
        set(HPX_CMAKE_ADDITIONAL_MODULE_PATH_BUILD
            "${lci_SOURCE_DIR}/cmake_modules"
            CACHE INTERNAL ""
        )
      endif()

      install(
        TARGETS LCI lci-ucx
        EXPORT HPXLCITarget
        COMPONENT core
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      )

      install(
        DIRECTORY ${lci_SOURCE_DIR}/src/api/ ${lci_BINARY_DIR}/src/api/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT core
        FILES_MATCHING
        PATTERN "*.h"
      )

      export(
        TARGETS LCI lci-ucx
        NAMESPACE LCI::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXLCITarget.cmake"
      )

      install(
        EXPORT HPXLCITarget
        NAMESPACE LCI::
        FILE HPXLCITarget.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
      )

      install(FILES "${lci_SOURCE_DIR}/cmake_modules/FindIBV.cmake"
              DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
      )
      install(FILES "${lci_SOURCE_DIR}/cmake_modules/FindOFI.cmake"
              DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
      )
    endif()
  endif()
endmacro()
