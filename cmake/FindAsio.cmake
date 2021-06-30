# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2010-2011 Matt Anderson
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET Asio::asio)
  find_path(
    ASIO_INCLUDE_DIR asio.hpp
    HINTS "${ASIO_ROOT}" ENV ASIO_ROOT "${HPX_ASIO_ROOT}"
    PATH_SUFFIXES include
  )

  if(NOT ASIO_INCLUDE_DIR)
    hpx_error(
      "Could not find Asio. Set ASIO_ROOT as a CMake or environment variable to point to the Asio root install directory. Alternatively, set HPX_WITH_FETCH_ASIO=ON to fetch Asio using CMake's FetchContent (when using this option Asio will be installed together with HPX, be careful about conflicts with separately installed versions of Asio)."
    )
  endif()

  # Set ASIO_ROOT in case the other hints are used
  if(ASIO_ROOT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${ASIO_ROOT} ASIO_ROOT)
  elseif("$ENV{ASIO_ROOT}")
    file(TO_CMAKE_PATH $ENV{ASIO_ROOT} ASIO_ROOT)
  else()
    file(TO_CMAKE_PATH "${ASIO_INCLUDE_DIR}" ASIO_INCLUDE_DIR)
    string(REPLACE "/include" "" ASIO_ROOT "${ASIO_INCLUDE_DIR}")
  endif()

  hpx_info("Found Asio: ${ASIO_ROOT}")

  add_library(Asio::asio INTERFACE IMPORTED)
  target_include_directories(Asio::asio SYSTEM INTERFACE ${ASIO_INCLUDE_DIR})

  mark_as_advanced(ASIO_ROOT ASIO_INCLUDE_DIR)
endif()
