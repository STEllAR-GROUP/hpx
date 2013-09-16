# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
# Copyright (c) 2011      Maciej Brodowicz
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_FINDPACKAGE_LOADED)
    include(HPX_FindPackage)
endif()

hpx_find_package(IBVERBS
    LIBRARIES ibverbs
    LIBRARY_PATHS lib
    HEADERS infiniband/verbs.h
    HEADER_PATHS include)

if(IBVERBS_FOUND)
  set(hpx_RUNTIME_LIBRARIES ${hpx_RUNTIME_LIBRARIES} ${IBVERBS_LIBRARY})
  hpx_include_sys_directories(${IBVERBS_INCLUDE_DIR})
  hpx_link_sys_directories(${IBVERBS_LIBRARY_DIR})
  hpx_add_config_define(HPX_HAVE_IBVERBS)
endif()
