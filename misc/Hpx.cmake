# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# We need to use Boost, namely the following Boost libraries
set(Boost_FIND_VERSION_EXACT OFF)
set(Boost_COMPONENTS_NEEDED
    date_time
    filesystem
    graph
    program_options
    regex
    serialization
    system
    signals
    thread)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_ADDITIONAL_VERSIONS 
    "1.45", "1.45.0", "1.44", "1.44.0", "1.43", "1.43.0", "1.42" "1.40.0" "1.40" "1.39.0" "1.39")
find_package(Boost 1.33.1 COMPONENTS ${Boost_COMPONENTS_NEEDED})

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

if(NOT DEFINED HPX_ROOT)
    set(HPX_ROOT $ENV{HPX_ROOT})
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${HPX_ROOT}/share/cmake)

find_package(HPX)
include(${HPX_ROOT}/share/cmake/HpxUtils.cmake)
