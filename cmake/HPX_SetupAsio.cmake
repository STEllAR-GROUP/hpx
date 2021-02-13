# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET ASIO::standalone_asio)
  if(NOT HPX_FIND_PACKAGE)
    if(NOT "${ASIO_ROOT}" AND "$ENV{ASIO_ROOT}")
      set(ASIO_ROOT "$ENV{ASIO_ROOT}")
    endif()

    if(ASIO_ROOT)
      set(HPX_WITH_CLONED_ASIO
          FALSE
          CACHE INTERNAL ""
      )
      set(HPX_ASIO_ROOT ${ASIO_ROOT})
    else()
      set(HPX_WITH_CLONED_ASIO
          TRUE
          CACHE INTERNAL ""
      )

      if(NOT HPX_WITH_ASIO_TAG)
        set(HPX_WITH_ASIO_TAG "asio-1-18-1")
      endif()

      include(FetchContent)
      fetchcontent_declare(
        asio
        GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
        GIT_TAG ${HPX_WITH_ASIO_TAG}
      )

      fetchcontent_getproperties(asio)
      if(NOT asio_POPULATED)
        fetchcontent_populate(asio)
      endif()
      set(ASIO_ROOT ${asio_SOURCE_DIR})

      hpx_info("ASIO_ROOT is not set. Cloning Asio into ${asio_SOURCE_DIR}.")

      add_library(standalone_asio INTERFACE)
      target_include_directories(
        standalone_asio SYSTEM
        INTERFACE $<BUILD_INTERFACE:${ASIO_ROOT}/asio/include>
                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
      )

      install(
        TARGETS standalone_asio
        EXPORT HPXAsioTarget
        COMPONENT core
      )

      install(
        DIRECTORY ${ASIO_ROOT}/asio/include/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT core
        FILES_MATCHING
        PATTERN "*.hpp"
        PATTERN "*.ipp"
      )

      export(
        TARGETS standalone_asio
        NAMESPACE ASIO::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXAsioTarget.cmake"
      )

      install(
        EXPORT HPXAsioTarget
        NAMESPACE ASIO::
        FILE HPXAsioTarget.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
      )

      add_library(ASIO::standalone_asio ALIAS standalone_asio)
    endif()

    # Asio should not use Boost exceptions
    hpx_add_config_cond_define(ASIO_HAS_BOOST_THROW_EXCEPTION 0)
    # Disable concepts support in Asio as a workaround to
    # https://github.com/boostorg/asio/issues/312
    hpx_add_config_cond_define(ASIO_DISABLE_CONCEPTS)
    # Disable experimental std::string_view support as a workaround to
    # https://github.com/chriskohlhoff/asio/issues/597
    hpx_add_config_cond_define(ASIO_DISABLE_STD_EXPERIMENTAL_STRING_VIEW)
  endif()

  if(NOT HPX_WITH_CLONED_ASIO)
    find_path(
      ASIO_INCLUDE_DIR asio.hpp
      HINTS "${HPX_ASIO_ROOT}" "${HPX_ASIO_ROOT}/asio"
      PATH_SUFFIXES include
    )

    if(NOT ASIO_INCLUDE_DIR)
      hpx_error("Could not find ASIO at ASIO_ROOT=${HPX_ASIO_ROOT}")
    endif()

    add_library(ASIO::standalone_asio INTERFACE IMPORTED)
    target_include_directories(
      ASIO::standalone_asio SYSTEM INTERFACE ${ASIO_INCLUDE_DIR}
    )

    mark_as_advanced(ASIO_INCLUDE_DIR)
  endif()
endif()
