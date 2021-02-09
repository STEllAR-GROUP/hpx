# Copyright (c) 2021 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET ASIO::standalone_asio)
  if(NOT HPX_FIND_PACKAGE)
    set(_hpx_asio_no_update)
    if(HPX_WITH_ASAIO_NO_UPDATE)
      set(_hpx_asio_no_update NO_UPDATE)
    endif()
    if(NOT HPX_WITH_ASIO_TAG)
      set(HPX_WITH_ASIO_TAG "asio-1-18-1")
    endif()

    # If APEX_ROOT not specified, local clone into hpx source dir
    if(NOT ASIO_ROOT)
      # handle APEX library
      include(GitExternal)
      git_external(
        asio https://github.com/chriskohlhoff/asio.git ${HPX_WITH_ASIO_TAG}
        ${_hpx_asio_no_update} VERBOSE
      )
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/asio)
        set(ASIO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/asio)
      else()
        hpx_error("ASIO could not be found")
      endif()
    endif()

    # copy over a minimal CMakeLists.txt usable to integrate asio
    file(
      WRITE ${CMAKE_CURRENT_SOURCE_DIR}/asio/CMakeLists.txt
      "cmake_minimum_required(VERSION 3.8)\n"
      "project(asio CXX)\n"
      "add_library(asio INTERFACE)\n"
      "install(TARGETS asio EXPORT asio INCLUDES DESTINATION include/)\n"
      "install(DIRECTORY asio/include/asio DESTINATION include/ FILES_MATCHING PATTERN \"*.hpp\" PATTERN \"*.ipp\")\n"
      "install(FILES asio/include/asio.hpp DESTINATION include/)\n"
    )

    add_subdirectory(${ASIO_ROOT})
  endif()

  add_library(ASIO::standalone_asio INTERFACE IMPORTED)
  if(HPX_FIND_PACKAGE)
    target_link_libraries(ASIO::standalone_asio INTERFACE HPX::standalone_asio)
  else()
    target_include_directories(
      ASIO::standalone_asio
      INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/asio/asio/include
    )
    target_link_libraries(ASIO::standalone_asio INTERFACE asio)
  endif()
endif()
