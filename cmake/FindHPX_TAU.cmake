# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

# This if statement is specific to TAU, and should not be copied into other
# Find cmake scripts.
if(NOT TAU_ROOT AND NOT $ENV{HOME_TAU} STREQUAL "")
  set(TAU_ROOT $ENV{HOME_TAU})
endif()

# Need to add -L$(TAU_ROOT)/x86_64/lib -lTAU

hpx_find_package(TAU
  LIBRARIES TAU  m
  LIBRARY_PATHS x86_64/lib 
  HEADERS TAU.h
  HEADER_PATHS include)

if(TAU_FOUND)
  set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} ${TAU_LIBRARY})
  hpx_include_sys_directories(${TAU_INCLUDE_DIR})
  hpx_link_sys_directories(${TAU_LIBRARY_DIR})
  add_definitions(-DHPX_HAVE_TAU)
endif()
