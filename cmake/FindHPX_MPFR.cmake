# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

find_package(HPX_GMP)

if(GMP_FOUND)
  hpx_find_package(MPFR
    LIBRARIES libmpfr mpfr
    LIBRARY_PATHS lib64 lib
    HEADERS mpfr.h
    HEADER_PATHS include)
endif()

