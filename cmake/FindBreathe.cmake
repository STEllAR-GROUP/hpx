# Copyright (c) 2018 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_program(
  BREATHE_APIDOC_EXECUTABLE
  NAMES breathe-apidoc
  PATHS ${BREATHE_APIDOC_ROOT} ENV BREATHE_APIDOC_ROOT
  DOC "Path to breathe-apidoc executable"
)

if(BREATHE_APIDOC_EXECUTABLE)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    Breathe DEFAULT_MESSAGE BREATHE_APIDOC_EXECUTABLE
  )
endif()
