# Copyright (c)      2014 Thomas Heller
# Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_program(BOOSTQUICKBOOK_EXECUTABLE
  quickbook
  PATHS
    ${CMAKE_SYSTEM_PREFIX_PATH}
    ${BOOST_ROOT}
    ${BOOSTQUICKBOOK_ROOT}
    ENV BOOST_ROOT ENV BOOSTQUICKBOOK_ROOT
  PATH_SUFFIXES bin dist/bin)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BoostQuickBook
  REQUIRED_VARS BOOSTQUICKBOOK_EXECUTABLE
)

if(BOOSTQUICKBOOK_EXECUTABLE)
  set(BOOSTQUICKBOOK_FOUND TRUE)
endif()
