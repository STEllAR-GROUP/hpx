# Copyright (c) 2011 Bryce Lelbach
# Copyright (c) 2012-2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_program(BOOSTWAVE_EXECUTABLE
  wave
  PATHS
    ${CMAKE_SYSTEM_PREFIX_PATH}
    ${BOOST_ROOT}
    ${BOOSTWAVE_ROOT}
    ENV BOOST_ROOT ENV BOOSTWAVE_ROOT
  PATH_SUFFIXES bin dist/bin)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BoostWave
  REQUIRED_VARS BOOSTWAVE_EXECUTABLE
)

if(BOOSTWAVE_EXECUTABLE)
  set(BOOSTWAVE_FOUND TRUE)
endif()
