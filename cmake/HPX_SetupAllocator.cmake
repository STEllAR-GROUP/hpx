# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011-2013 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(NOT HPX_WITH_MALLOC)
  set(HPX_WITH_MALLOC ${DEFAULT_MALLOC})
  set(allocator_error
    "The default allocator for your system is ${DEFAULT_MALLOC}, but ${DEFAULT_MALLOC} could not be found. "
      "The system allocator has poor performance. As such ${DEFAULT_MALLOC} is a strong optional requirement. "
      "Being aware of the performance hit, you can override this default and get rid of this dependency by setting -DHPX_WITH_MALLOC=system. "
      "Valid options for HPX_WITH_MALLOC are: system, tcmalloc, jemalloc, tbbmalloc, and custom")
else()
  set(allocator_error
    "HPX_WITH_MALLOC was set to ${HPX_WITH_MALLOC}, but ${HPX_WITH_MALLOC} could not be found. "
      "Valid options for HPX_WITH_MALLOC are: system, tcmalloc, jemalloc, tbbmalloc, and custom")
endif()

string(TOUPPER "${HPX_WITH_MALLOC}" HPX_WITH_MALLOC_UPPER)

if(NOT HPX_WITH_MALLOC_DEFAULT)
  if("${HPX_WITH_MALLOC_UPPER}" STREQUAL "TCMALLOC")
    find_package(TCMalloc)
    if(NOT TCMALLOC_LIBRARIES)
      hpx_error(${allocator_error})
    endif()
    if(COMMAND hpx_libraries)
      hpx_libraries(${TCMALLOC_LIBRARIES})
    endif()
    set(hpx_MALLOC_LIBRARY ${TCMALLOC_LIBRARIES})
    if(MSVC)
      hpx_add_link_flag_if_available(/INCLUDE:__tcmalloc)
    endif()
    set(_use_custom_allocator TRUE)
  endif()

  if("${HPX_WITH_MALLOC_UPPER}" STREQUAL "JEMALLOC")
    if(MSVC)
      hpx_error("jemalloc is not usable with MSVC")
    endif()
    find_package(Jemalloc)
    if(NOT JEMALLOC_LIBRARIES)
      hpx_error(${allocator_error})
    endif()
    if(COMMAND hpx_libraries)
      hpx_libraries(${JEMALLOC_LIBRARIES})
    endif()
    set(hpx_MALLOC_LIBRARY ${JEMALLOC_LIBRARIES})
    set(_use_custom_allocator TRUE)
  endif()

  if("${HPX_WITH_MALLOC_UPPER}" STREQUAL "TBBMALLOC")
    find_package(TBBmalloc)
    if(NOT TBBMALLOC_LIBRARY AND NOT TBBMALLOC_PROXY_LIBRARY)
      hpx_error(${allocator_error})
    endif()
    if(MSVC)
      hpx_add_link_flag_if_available(/INCLUDE:__TBB_malloc_proxy)
    endif()
    if(COMMAND hpx_libraries)
      hpx_libraries(${TBBMALLOC_LIBRARY} ${TBBMALLOC_PROXY_LIBRARY})
    endif()
    set(hpx_MALLOC_LIBRARY ${TBBMALLOC_LIBRARY} ${TBBMALLOC_PROXY_LIBRARY})
    set(_use_custom_allocator TRUE)
  endif()

  if("${HPX_WITH_MALLOC_UPPER}" STREQUAL "CUSTOM")
    set(_use_custom_allocator TRUE)
  endif()
else()
  set(_use_custom_allocator TRUE)
endif()

if("${HPX_WITH_MALLOC_UPPER}" MATCHES "SYSTEM")
  if(NOT MSVC)
    hpx_warn("HPX will perform poorly without tcmalloc or jemalloc. See docs for more info.")
  endif()
  set(_use_custom_allocator FALSE)
endif()

hpx_info("Using ${HPX_WITH_MALLOC} allocator.")
