#  Copyright (c) 2024 Isidoros Tsaousis-Seiras
#
#  SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET STDEXEC::stdexec)
  if (STDEXEC_ROOT AND NOT Stdexec_ROOT)
    set(Stdexec_ROOT ${STDEXEC_ROOT} CACHE PATH "stdexec base directory")
    unset(STDEXEC_ROOT CACHE)
  endif()

  find_path(
    Stdexec_INCLUDE_DIR
    stdexec
    HINTS ${Stdexec_ROOT}
  )
  message(STATUS "stdexec include dir: ${Stdexec_INCLUDE_DIR}")
  if (Stdexec_INCLUDE_DIR)
    file(TO_CMAKE_PATH ${Stdexec_INCLUDE_DIR} Stdexec_INCLUDE_DIR)
  else()
    message(FATAL_ERROR "stdexec not found")
  endif()

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
      Stdexec
      REQUIRED_VARS Stdexec_INCLUDE_DIR
      FOUND_VAR Stdexec_FOUND
      VERSION_VAR Stdexec_VERSION
      FAIL_MESSAGE "stdexec not found"
  )

  add_library(STDEXEC::stdexec INTERFACE IMPORTED)
  target_include_directories(STDEXEC::stdexec SYSTEM INTERFACE ${Stdexec_INCLUDE_DIR})

  mark_as_advanced(Stdexec_INCLUDE_DIR Stdexec_ROOT)
endif()
