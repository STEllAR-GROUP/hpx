# Copyright (c) 2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(MSVC)
  hpx_find_package(GMP
    LIBRARIES mpir
    LIBRARY_PATHS lib64 lib
    HEADERS mpir.h 
    HEADER_PATHS include)
else()
  hpx_find_package(GMP
    LIBRARIES libgmp gmp
    LIBRARY_PATHS lib64 lib
    HEADERS gmp.h 
    HEADER_PATHS include)
endif()

