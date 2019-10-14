# Copyright (c) 2012 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#FIXME: as mp is just a prove of concept for now, we search the build directory.
hpx_find_package(MP
  LIBRARIES libmp mp
  LIBRARY_PATHS lib64 lib build
  HEADERS mp/mp.hpp
  HEADER_PATHS include)

