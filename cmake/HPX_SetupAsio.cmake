# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_FETCH_ASIO)
  find_package(Asio 1.12.0 REQUIRED)
elseif(NOT TARGET Asio::asio AND NOT HPX_FIND_PACKAGE)
  if(FETCHCONTENT_SOURCE_DIR_ASIO)
    hpx_info(
      "HPX_WITH_FETCH_ASIO=${HPX_WITH_FETCH_ASIO}, Asio will be used through CMake's FetchContent and installed alongside HPX (FETCHCONTENT_SOURCE_DIR_ASIO=${FETCHCONTENT_SOURCE_DIR_ASIO})"
    )
  else()
    hpx_info(
      "HPX_WITH_FETCH_ASIO=${HPX_WITH_FETCH_ASIO}, Asio will be fetched using CMake's FetchContent and installed alongside HPX (HPX_WITH_ASIO_TAG=${HPX_WITH_ASIO_TAG})"
    )
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
    FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXAsioTarget.cmake"
  )

  install(
    EXPORT HPXAsioTarget
    NAMESPACE Asio::
    FILE HPXAsioTarget.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
  )

  add_library(Asio::asio ALIAS asio)
endif()

if(NOT HPX_FIND_PACKAGE)
  # Asio should use std::aligned_new only if available
  if(NOT HPX_WITH_CXX17_ALIGNED_NEW)
    hpx_add_config_cond_define(ASIO_DISABLE_STD_ALIGNED_ALLOC)
  endif()
  # Asio does not detect that invoke_result is available, but we assume it
  # always is since we require C++17.
  hpx_add_config_cond_define(ASIO_HAS_STD_INVOKE_RESULT 1)
  # Asio should not use Boost exceptions
  hpx_add_config_cond_define(ASIO_HAS_BOOST_THROW_EXCEPTION 0)
  # Disable concepts support in Asio as a workaround to
  # https://github.com/boostorg/asio/issues/312
  hpx_add_config_cond_define(ASIO_DISABLE_CONCEPTS)
  # Disable experimental std::string_view support as a workaround to
  # https://github.com/chriskohlhoff/asio/issues/597
  hpx_add_config_cond_define(ASIO_DISABLE_STD_EXPERIMENTAL_STRING_VIEW)
  # Disable Asio's definition of NOMINMAX
  hpx_add_config_cond_define(ASIO_NO_NOMINMAX)
endif()
