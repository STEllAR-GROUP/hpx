# Copyright (c) 2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(MSVC)
  hpx_find_package(AMPLIFIER
    LIBRARIES libittnotify.lib
    LIBRARY_PATHS lib64 
    HEADERS ittnotify.h 
    HEADER_PATHS include)
else()
  hpx_find_package(AMPLIFIER
    LIBRARIES libittnotify.a
    LIBRARY_PATHS lib64 
    HEADERS ittnotify.h 
    HEADER_PATHS include)
endif()

