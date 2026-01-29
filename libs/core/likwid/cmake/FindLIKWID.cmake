# Copyright (c) 2022 Srinivas Yadav
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET LIKWID::likwid)

  find_path(
    LIKWID_INCLUDE_DIR likwid.h
    HINTS "${LIKWID_ROOT}" ENV LIKWID_ROOT "${HPX_LIKWID_ROOT}"
    PATH_SUFFIXES include
  )

  if(NOT LIKWID_INCLUDE_DIR)
    hpx_error("Could not find likwid.h")
  endif()
  message(STATUS "Found likwid header: ${LIKWID_INCLUDE_DIR}")

  find_library(
    LIKWID_LIBRARY likwid
    HINTS "${LIKWID_ROOT}" ENV LIKWID_ROOT "${HPX_LIKWID_ROOT}"
    PATH_SUFFIXES lib
  )

  if(NOT LIKWID_LIBRARY)
    hpx_error("Could not find likwid library")
  endif()
  message(STATUS "Found likwid library: ${LIKWID_LIBRARY}")

  if(LIKWID_ROOT)
    # The call to file is for compatibility with windows paths
    file(TO_CMAKE_PATH ${LIKWID_ROOT} LIKWID_ROOT)
  elseif("$ENV{LIKWID_ROOT}")
    file(TO_CMAKE_PATH $ENV{LIKWID_ROOT} LIKWID_ROOT)
  else()
    file(TO_CMAKE_PATH "${LIKWID_INCLUDE_DIR}" LIKWID_INCLUDE_DIR)
    string(REPLACE "/include" "" LIKWID_ROOT "${LIKWID_INCLUDE_DIR}")
  endif()

  mark_as_advanced(LIKWID_ROOT LIKWID_LIBRARY LIKWID_INCLUDE_DIR)

  add_library(LIKWID::likwid INTERFACE IMPORTED)
  target_include_directories(
    LIKWID::likwid SYSTEM INTERFACE ${LIKWID_INCLUDE_DIR}
  )
  target_link_libraries(LIKWID::likwid INTERFACE ${LIKWID_LIBRARY})
  target_compile_definitions(LIKWID::likwid INTERFACE LIKWID_PERFMON)
endif()
