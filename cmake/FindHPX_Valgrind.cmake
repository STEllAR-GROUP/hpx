# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

if(NOT VALGRIND_ROOT AND NOT $ENV{HOME_VALGRIND} STREQUAL "")
  set(VALGRIND_ROOT $ENV{HOME_VALGRIND})
endif()
if(NOT VALGRIND_ROOT AND NOT $ENV{VALGRIND_ROOT} STREQUAL "")
  set(VALGRIND_ROOT $ENV{VALGRIND_ROOT})
endif()


hpx_find_headers(VALGRIND
  HEADERS valgrind/valgrind.h
  HEADER_PATHS include)

if(VALGRIND_INCLUDE_DIR)
  hpx_include_sys_directories(${VALGRIND_INCLUDE_DIR})
  hpx_add_config_define(HPX_HAVE_VALGRIND)
endif()
