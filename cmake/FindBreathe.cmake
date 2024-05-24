# Copyright (c) 2018 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# compatibility with older CMake versions
if(BREATHE_APIDOC_ROOT AND NOT Breathe_APIDOC_ROOT)
  set(Breathe_APIDOC_ROOT
      ${BREATHE_APIDOC_ROOT}
      CACHE PATH "Breathe base directory"
  )
  unset(BREATHE_ROOT CACHE)
endif()

find_program(
  Breathe_APIDOC_EXECUTABLE
  NAMES breathe-apidoc
  PATHS ${Breathe_APIDOC_ROOT} ENV Breathe_APIDOC_ROOT
  DOC "Path to breathe-apidoc executable"
)

if(Breathe_APIDOC_ROOT)
  file(TO_CMAKE_PATH ${Breathe_APIDOC_ROOT} Breathe_APIDOC_ROOT)
endif()

if(Breathe_APIDOC_EXECUTABLE)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    Breathe DEFAULT_MESSAGE Breathe_APIDOC_EXECUTABLE
  )
endif()
