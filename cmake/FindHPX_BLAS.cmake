# Copyright (c) 2011 Matt Anderson
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

hpx_find_package(BLAS
  LIBRARIES blas libblas
  LIBRARY_PATHS lib64 lib
  HEADERS cblas.h
  HEADER_PATHS include)

