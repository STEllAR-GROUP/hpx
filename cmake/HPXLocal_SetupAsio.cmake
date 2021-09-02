# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPXLocal_WITH_FETCH_ASIO)
  find_package(Asio 1.12.0 REQUIRED)
elseif(NOT TARGET Asio::asio AND NOT HPXLocal_FIND_PACKAGE)
  if(FETCHCONTENT_SOURCE_DIR_ASIO)
    hpx_local_info(
      "HPXLocal_WITH_FETCH_ASIO=${HPXLocal_WITH_FETCH_ASIO}, Asio will be used through CMake's FetchContent and installed alongside HPX (FETCHCONTENT_SOURCE_DIR_ASIO=${FETCHCONTENT_SOURCE_DIR_ASIO})"
    )
  else()
    hpx_local_info(
      "HPXLocal_WITH_FETCH_ASIO=${HPXLocal_WITH_FETCH_ASIO}, Asio will be fetched using CMake's FetchContent and installed alongside HPX (HPXLocal_WITH_ASIO_TAG=${HPXLocal_WITH_ASIO_TAG})"
    )
  endif()
  include(FetchContent)
  fetchcontent_declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG ${HPXLocal_WITH_ASIO_TAG}
  )

  fetchcontent_getproperties(asio)
  if(NOT asio_POPULATED)
    fetchcontent_populate(asio)
  endif()
  set(ASIO_ROOT ${asio_SOURCE_DIR})

  add_library(asio INTERFACE)
  target_include_directories(
    asio SYSTEM INTERFACE $<BUILD_INTERFACE:${ASIO_ROOT}/asio/include>
                          $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

  install(
    TARGETS asio
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
    TARGETS asio
    NAMESPACE Asio::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/HPXAsioTarget.cmake"
  )

  install(
    EXPORT HPXAsioTarget
    NAMESPACE Asio::
    FILE HPXAsioTarget.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPXLocal_PACKAGE_NAME}
  )

  add_library(Asio::asio ALIAS asio)
endif()

if(NOT HPXLocal_FIND_PACKAGE)
  # Asio should not use Boost exceptions
  hpx_local_add_config_cond_define(ASIO_HAS_BOOST_THROW_EXCEPTION 0)
  # Disable concepts support in Asio as a workaround to
  # https://github.com/boostorg/asio/issues/312
  hpx_local_add_config_cond_define(ASIO_DISABLE_CONCEPTS)
  # Disable experimental std::string_view support as a workaround to
  # https://github.com/chriskohlhoff/asio/issues/597
  hpx_local_add_config_cond_define(ASIO_DISABLE_STD_EXPERIMENTAL_STRING_VIEW)
endif()
