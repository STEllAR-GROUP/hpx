# Copyright (c) 2012 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  hpx_include(FindPackage)
endif()

hpx_find_package(OCLM
  LIBRARIES oclm.lib
  LIBRARY_PATHS lib/Debug lib/Release
  HEADERS oclm/oclm.hpp
  HEADER_PATHS include inc)