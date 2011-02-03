# Copyright (c) 2007-2009 Hartmut Kaiser
# Copyright (c) 2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We need to use Boost, namely the following Boost libraries
list(APPEND BOOST_LIBRARIES
    date_time
    filesystem
    program_options
    regex
    serialization
    system
    signals
    thread)

if(NOT ${BOOST_MINOR_VERSION} LESS 46)
  list(APPEND BOOST_LIBRARIES chrono)
endif()

if(NOT DEFINED HPX_ROOT)
    set(HPX_ROOT $ENV{HPX_ROOT})
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${HPX_ROOT}/share/cmake)

find_package(HPX)
include(${HPX_ROOT}/share/cmake/HpxUtils.cmake)
include(${HPX_ROOT}/share/cmake/FindBoost.cmake)

if(BOOST_INCLUDE_DIR)
  include_directories(${BOOST_INCLUDE_DIR})
endif()

if(BOOST_LIB_DIR)
  link_directories(${BOOST_LIB_DIR})
endif()

