#  Copyright (c) 2024 Isidoros Tsaousis-Seiras
#
#  SPDX-License-Identifier: BSL-1.0
#  Distributed under the Boost Software License, Version 1.0. (See accompanying
#  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT TARGET STDEXEC::stdexec)

  find_path(Stdexec_INCLUDE_DIR stdexec HINTS ${STDEXEC_ROOT} ENV STDEXEC_ROOT)
  message(STATUS "stdexec include dir: ${Stdexec_INCLUDE_DIR}")

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Stdexec DEFAULT_MESSAGE Stdexec_INCLUDE_DIR)

  add_library(STDEXEC::stdexec INTERFACE IMPORTED)
  target_include_directories(
    STDEXEC::stdexec SYSTEM INTERFACE ${Stdexec_INCLUDE_DIR}
  )

  mark_as_advanced(Stdexec_INCLUDE_DIR Stdexec_ROOT)
endif()
