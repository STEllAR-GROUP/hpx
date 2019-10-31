# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011-2013 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddDefinitions)

if(NOT HPX_WITH_MALLOC)
  set(HPX_WITH_MALLOC CACHE STRING
          "Use the specified allocator. Supported allocators are tcmalloc, jemalloc, tbbmalloc and system."
          ${DEFAULT_MALLOC})
  set(allocator_error
    "The default allocator for your system is ${DEFAULT_MALLOC}, but ${DEFAULT_MALLOC} could not be found. "
      "The system allocator has poor performance. As such ${DEFAULT_MALLOC} is a strong optional requirement. "
      "Being aware of the performance hit, you can override this default and get rid of this dependency by setting -DHPX_WITH_MALLOC=system. "
      "Valid options for HPX_WITH_MALLOC are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom")
else()
  set(allocator_error
    "HPX_WITH_MALLOC was set to ${HPX_WITH_MALLOC}, but ${HPX_WITH_MALLOC} could not be found. "
      "Valid options for HPX_WITH_MALLOC are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom")
endif()

string(TOUPPER "${HPX_WITH_MALLOC}" HPX_WITH_MALLOC_UPPER)

add_library(hpx::allocator INTERFACE IMPORTED)

if(NOT HPX_WITH_MALLOC_DEFAULT)
  if("${HPX_WITH_MALLOC_UPPER}" STREQUAL "TCMALLOC")
    find_package(TCMalloc)
    if(NOT TCMALLOC_LIBRARIES)
      hpx_error(${allocator_error})
    endif()

    if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
      set_property(TARGET hpx::allocator PROPERTY
        INTERFACE_LINK_LIBRARIES ${TCMALLOC_LIBRARIES})
    else()
        target_link_libraries(hpx::allocator INTERFACE ${TCMALLOC_LIBRARIES})
    endif()

    if(MSVC)
      hpx_add_link_flag_if_available(/INCLUDE:__tcmalloc)
    endif()
    set(_use_custom_allocator TRUE)
  endif()

  if("${HPX_WITH_MALLOC_UPPER}" STREQUAL "JEMALLOC")
    find_package(Jemalloc)
    if(NOT JEMALLOC_LIBRARIES)
      hpx_error(${allocator_error})
    endif()

    set_property(TARGET hpx::allocator PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES ${JEMALLOC_INCLUDE_DIR}
      ${JEMALLOC_ADDITIONAL_INCLUDE_DIR})
    if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
      set_property(TARGET hpx::allocator PROPERTY
        INTERFACE_LINK_LIBRARIES ${JEMALLOC_LIBRARIES})
    else()
      target_link_libraries(hpx::allocator INTERFACE ${JEMALLOC_LIBRARIES})
    endif()
  endif()

  if("${HPX_WITH_MALLOC_UPPER}" STREQUAL "MIMALLOC")
    find_package(mimalloc 1.0)
    if(NOT mimalloc_FOUND)
      hpx_error(${allocator_error})
    endif()
    set_property(TARGET hpx::allocator PROPERTY
      INTERFACE_LINK_LIBRARIES ${MIMALLOC_LIBRARIES})
    if(MSVC)
      hpx_add_link_flag_if_available(/INCLUDE:mi_version)
    endif()
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
    if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
      set_property(TARGET hpx::allocator PROPERTY
        INTERFACE_LINK_LIBRARIES ${TBBMALLOC_LIBRARY} ${TBBMALLOC_PROXY_LIBRARY})
    else()
      target_link_libraries(hpx::allocator INTERFACE
        ${TBBMALLOC_LIBRARY} ${TBBMALLOC_PROXY_LIBRARY})
    endif()
  endif()

  if("${HPX_WITH_MALLOC_UPPER}" STREQUAL "CUSTOM")
    set(_use_custom_allocator TRUE)
  endif()
else()
  set(HPX_WITH_MALLOC ${HPX_WITH_MALLOC_DEFAULT})
endif()

if("${HPX_WITH_MALLOC_UPPER}" MATCHES "SYSTEM")
  if(NOT MSVC)
    hpx_warn("HPX will perform poorly without tcmalloc, jemalloc, or mimalloc. See docs for more info.")
  endif()
  set(_use_custom_allocator FALSE)
endif()

hpx_info("Using ${HPX_WITH_MALLOC} allocator.")

# Setup Intel amplifier
if((NOT HPX_WITH_APEX) AND HPX_WITH_ITTNOTIFY)
  find_package(Amplifier)
  if(NOT AMPLIFIER_FOUND)
    hpx_error("Intel Amplifier could not be found and HPX_WITH_ITTNOTIFY=On, please specify AMPLIFIER_ROOT to point to the root of your Amplifier installation")
  endif()

  add_library(hpx::amplifier INTERFACE IMPORTED)
  set_property(TARGET hpx::amplifier PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES ${AMPLIFIER_INCLUDE_DIR})
  if(${CMAKE_VERSION} VERSION_LESS "3.12.0")
    set_property(TARGET hpx::amplifier PROPERTY
      INTERFACE_LINK_LIBRARIES ${AMPLIFIER_LIBRARIES})
  else()
    target_link_libraries(hpx::allocator INTERFACE ${AMPLIFIER_LIBRARIES})
  endif()

  hpx_add_config_define(HPX_HAVE_ITTNOTIFY 1)
  hpx_add_config_define(HPX_HAVE_THREAD_DESCRIPTION)
endif()

# convey selected allocator type to the build configuration
hpx_add_config_define(HPX_HAVE_MALLOC "\"${HPX_WITH_MALLOC}\"")
if(${HPX_WITH_MALLOC} STREQUAL "jemalloc")
  if(NOT ("${HPX_WITH_JEMALLOC_PREFIX}" STREQUAL "<none>") AND
     NOT ("${HPX_WITH_JEMALLOC_PREFIX}x" STREQUAL "x"))
    hpx_add_config_define(HPX_HAVE_JEMALLOC_PREFIX ${HPX_WITH_JEMALLOC_PREFIX})
    hpx_add_config_define(HPX_HAVE_INTERNAL_ALLOCATOR)
  endif()
endif()
