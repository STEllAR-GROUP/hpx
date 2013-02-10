# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
  include(HPX_FindPackage)
endif()

# This if statement is specific to APEX, and should not be copied into other
# Find cmake scripts.
if(NOT APEX_ROOT AND NOT $ENV{HOME_APEX} STREQUAL "")
  set(APEX_ROOT $ENV{HOME_APEX})
endif()
if(NOT APEX_ROOT AND NOT $ENV{APEX_ROOT} STREQUAL "")
  set(APEX_ROOT $ENV{APEX_ROOT})
endif()

# Need to add -L$(APEX_ROOT)/x86_64/lib -lApex

hpx_find_package(APEX
  LIBRARIES Apex
  LIBRARY_PATHS lib 
  HEADERS apex.h
  HEADER_PATHS include)

if(APEX_FOUND)
  #set(hpx_LIBRARIES ${hpx_LIBRARIES} ${APEX_LIBRARY})
  set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} ${APEX_LIBRARY})
  hpx_include_sys_directories(${APEX_INCLUDE_DIR})
  hpx_link_sys_directories(${APEX_LIBRARY_DIR})
  add_definitions(-DHPX_HAVE_APEX)
endif()
