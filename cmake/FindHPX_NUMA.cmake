# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_find_package(NUMA
  LIBRARIES numa libnuma
  LIBRARY_PATHS lib64 lib
  HEADERS numa.h
  HEADER_PATHS include)

