# Copyright (c) 2018 Mikael Simberg
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_program(
  SPHINX_EXECUTABLE
  NAMES sphinx-build sphinx-build2
  PATHS ${SPHINX_ROOT} ENV SPHINX_ROOT
  DOC "Path to sphinx-build executable"
)

if(SPHINX_EXECUTABLE)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Sphinx DEFAULT_MESSAGE SPHINX_EXECUTABLE)
endif()
