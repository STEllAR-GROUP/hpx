# Copyright (c)      2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_program(BOOSTAUTOINDEX_EXECUTABLE
  auto_index
  PATHS
    ${CMAKE_SYSTEM_PREFIX_PATH}
    ${BOOST_ROOT}
    ${BOOSTAUTOINDEX_ROOT}
    ENV BOOST_ROOT ENV BOOSTQUICKBOOK_ROOT
  PATH_SUFFIXES bin dist/bin)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BoostAutoIndex
  REQUIRED_VARS BOOSTAUTOINDEX_EXECUTABLE
  )
