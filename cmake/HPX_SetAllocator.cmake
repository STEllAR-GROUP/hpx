# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011-2013 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


if(NOT HPX_MALLOC)
  set(HPX_MALLOC ${HPX_MALLOC_DEFAULT})
  set(allocator_error
      "The default allocator for your system is ${HPX_MALLOC_DEFAULT}, but ${HPX_MALLOC_DEFAULT} could not be found. "
      "The system allocator has poor performance. As such ${HPX_MALLOC_DEFAULT} is a strong optional requirement. "
      "Being aware of the performance hit, you can override this default and get rid of this dependency by setting -DHPX_MALLOC=system. "
      "Other valid options for HPX_MALLOC are: system, tcmalloc, jemalloc, tbbmalloc")
else()
  set(allocator_error
      "HPX_MALLOC was set to ${HPX_MALLOC}, but ${HPX_MALLOC} could not be found. "
      "Other valid options for HPX_MALLOC are: system, tcmalloc, jemalloc, tbbmalloc")
endif()

hpx_info("malloc" "Using ${HPX_MALLOC} allocator.")

string(TOUPPER "${HPX_MALLOC}" HPX_MALLOC_UPPER)

if("${HPX_MALLOC_UPPER}" STREQUAL "TCMALLOC")
  find_package(HPX_TCMalloc)
  if(NOT TCMALLOC_FOUND)
    hpx_error("malloc" ${allocator_error})
  endif()
  set(hpx_MALLOC_LIBRARY ${TCMALLOC_LIBRARY})
  set(hpx_LIBRARIES ${hpx_LIBRARIES} ${TCMALLOC_LIBRARY})
  if(MSVC)
    set(HPX_LINK_FLAG_TARGET_PROPERTIES "/INCLUDE:__tcmalloc")
  endif()
  hpx_include_sys_directories(${TCMALLOC_INCLUDE_DIR})
  hpx_link_sys_directories(${TCMALLOC_LIBRARY_DIR})
  hpx_add_config_define(HPX_TCMALLOC)
  set(HPX_USE_CUSTOM_ALLOCATOR On)
endif()

if("${HPX_MALLOC_UPPER}" STREQUAL "JEMALLOC")
    if(MSVC)
      hpx_error("malloc" "jemalloc is not usable with MSVC")
    endif()
    find_package(HPX_Jemalloc)
    if(NOT JEMALLOC_FOUND)
      hpx_error("malloc" ${allocator_error})
    endif()
    set(hpx_MALLOC_LIBRARY ${JEMALLOC_LIBRARY})
    set(hpx_LIBRARIES ${hpx_LIBRARIES} ${JEMALLOC_LIBRARY})
    hpx_include_sys_directories(${JEMALLOC_INCLUDE_DIR})
    hpx_link_sys_directories(${JEMALLOC_LIBRARY_DIR})
    hpx_add_config_define(HPX_JEMALLOC)
    set(HPX_USE_CUSTOM_ALLOCATOR On)
endif()

if("${HPX_MALLOC_UPPER}" STREQUAL "TBBMALLOC")
  find_package(HPX_TBBmalloc)
  if(NOT TBBMALLOC_FOUND)
    hpx_error("malloc" ${allocator_error})
  endif()
  if(MSVC)
    set(HPX_LINK_FLAG_TARGET_PROPERTIES "/INCLUDE:__TBB_malloc_proxy")
  endif()
  set(hpx_MALLOC_LIBRARY "${TBBMALLOC_LIBRARY};${TBBMALLOC_PROXY_LIBRARY}")
  set(hpx_LIBRARIES ${hpx_LIBRARIES} ${TBBMALLOC_LIBRARY})
  set(hpx_LIBRARIES ${hpx_LIBRARIES} ${TBBMALLOC_PROXY_LIBRARY})
  hpx_include_sys_directories(${TBBMALLOC_INCLUDE_DIR})
  hpx_link_sys_directories(${TBBMALLOC_LIBRARY_DIR})
  hpx_add_config_define(HPX_TBBMALLOC)
endif()

if("${HPX_MALLOC}" MATCHES "system")
  hpx_info("malloc" "Using system allocator.")

  hpx_warn("malloc"
    "HPX will perform poorly without tcmalloc or jemalloc. See docs for more info.")
endif()
