# Copyright (c) 2012 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

#FIXME: as mp is just a prove of concept for now, we search the build directory.
hpx_find_package(MP
  LIBRARIES libmp mp
  LIBRARY_PATHS lib64 lib build
  HEADERS mp/mp.hpp
  HEADER_PATHS include)

