# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011-2013 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPXLocal_AddDefinitions)

# In case find_package(HPX) is called multiple times
if(NOT TARGET hpx_local_dependencies_allocator)

  if(NOT HPXLocal_WITH_MALLOC)
    set(HPXLocal_WITH_MALLOC
        CACHE
          STRING
          "Use the specified allocator. Supported allocators are tcmalloc, jemalloc, tbbmalloc and system."
          ${DEFAULT_MALLOC}
    )
    set(allocator_error
        "The default allocator for your system is ${DEFAULT_MALLOC}, but ${DEFAULT_MALLOC} could not be found. "
        "The system allocator has poor performance. As such ${DEFAULT_MALLOC} is a strong optional requirement. "
        "Being aware of the performance hit, you can override this default and get rid of this dependency by setting -DHPXLocal_WITH_MALLOC=system. "
        "Valid options for HPXLocal_WITH_MALLOC are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom"
    )
  else()
    set(allocator_error
        "HPXLocal_WITH_MALLOC was set to ${HPXLocal_WITH_MALLOC}, but ${HPXLocal_WITH_MALLOC} could not be found. "
        "Valid options for HPXLocal_WITH_MALLOC are: system, tcmalloc, jemalloc, mimalloc, tbbmalloc, and custom"
    )
  endif()

  string(TOUPPER "${HPXLocal_WITH_MALLOC}" HPXLocal_WITH_MALLOC_UPPER)

  add_library(hpx_local_dependencies_allocator INTERFACE IMPORTED)

  if(NOT HPXLocal_WITH_MALLOC_DEFAULT)

    # ##########################################################################
    # TCMALLOC
    if("${HPXLocal_WITH_MALLOC_UPPER}" STREQUAL "TCMALLOC")
      find_package(TCMalloc)
      if(NOT TCMALLOC_LIBRARIES)
        hpx_local_error(${allocator_error})
      endif()

      target_link_libraries(
        hpx_local_dependencies_allocator INTERFACE ${TCMALLOC_LIBRARIES}
      )

      if(MSVC)
        target_compile_options(
          hpx_local_dependencies_allocator INTERFACE /INCLUDE:__tcmalloc
        )
      endif()
      set(_use_custom_allocator TRUE)
    endif()

    # ##########################################################################
    # JEMALLOC
    if("${HPXLocal_WITH_MALLOC_UPPER}" STREQUAL "JEMALLOC")
      find_package(Jemalloc)
      if(NOT JEMALLOC_LIBRARIES)
        hpx_local_error(${allocator_error})
      endif()
      target_include_directories(
        hpx_local_dependencies_allocator
        INTERFACE ${JEMALLOC_INCLUDE_DIR} ${JEMALLOC_ADDITIONAL_INCLUDE_DIR}
      )
      target_link_libraries(
        hpx_local_dependencies_allocator INTERFACE ${JEMALLOC_LIBRARIES}
      )
    endif()

    # ##########################################################################
    # MIMALLOC
    if("${HPXLocal_WITH_MALLOC_UPPER}" STREQUAL "MIMALLOC")
      find_package(mimalloc 1.0)
      if(NOT mimalloc_FOUND)
        hpx_local_error(${allocator_error})
      endif()
      target_link_libraries(hpx_local_dependencies_allocator INTERFACE mimalloc)
      set(hpx_local_MALLOC_LIBRARY mimalloc)
      if(MSVC)
        target_compile_options(
          hpx_local_dependencies_allocator INTERFACE /INCLUDE:mi_version
        )
      endif()
      set(_use_custom_allocator TRUE)
    endif()

    # ##########################################################################
    # TBBMALLOC
    if("${HPXLocal_WITH_MALLOC_UPPER}" STREQUAL "TBBMALLOC")
      find_package(TBBmalloc)
      if(NOT TBBMALLOC_LIBRARY AND NOT TBBMALLOC_PROXY_LIBRARY)
        hpx_local_error(${allocator_error})
      endif()
      if(MSVC)
        target_compile_options(
          hpx_local_dependencies_allocator
          INTERFACE /INCLUDE:__TBB_malloc_proxy
        )
      endif()
      target_link_libraries(
        hpx_local_dependencies_allocator INTERFACE ${TBBMALLOC_LIBRARY}
                                                   ${TBBMALLOC_PROXY_LIBRARY}
      )
    endif()

    if("${HPXLocal_WITH_MALLOC_UPPER}" STREQUAL "CUSTOM")
      set(_use_custom_allocator TRUE)
    endif()

  else()

    set(HPXLocal_WITH_MALLOC ${HPXLocal_WITH_MALLOC_DEFAULT})

  endif(NOT HPXLocal_WITH_MALLOC_DEFAULT)

  if("${HPXLocal_WITH_MALLOC_UPPER}" MATCHES "SYSTEM")
    if(NOT MSVC)
      hpx_local_warn(
        "HPX will perform poorly without tcmalloc, jemalloc, or mimalloc. See docs for more info."
      )
    endif()
    set(_use_custom_allocator FALSE)
  endif()

  hpx_local_info("Using ${HPXLocal_WITH_MALLOC} allocator.")

  # Setup Intel amplifier
  if((NOT HPXLocal_WITH_APEX) AND HPXLocal_WITH_ITTNOTIFY)

    find_package(Amplifier)
    if(NOT AMPLIFIER_FOUND)
      hpx_local_error(
        "Intel Amplifier could not be found and HPXLocal_WITH_ITTNOTIFY=On, please specify AMPLIFIER_ROOT to point to the root of your Amplifier installation"
      )
    endif()

    hpx_local_add_config_define(HPX_HAVE_ITTNOTIFY 1)
    hpx_local_add_config_define(HPX_HAVE_THREAD_DESCRIPTION)
  endif()

  # convey selected allocator type to the build configuration
  if(NOT HPXLocal_FIND_PACKAGE)
    hpx_local_add_config_define(HPX_HAVE_MALLOC "\"${HPXLocal_WITH_MALLOC}\"")
    if(${HPXLocal_WITH_MALLOC} STREQUAL "jemalloc")
      if(NOT ("${HPXLocal_WITH_JEMALLOC_PREFIX}" STREQUAL "<none>")
         AND NOT ("${HPXLocal_WITH_JEMALLOC_PREFIX}x" STREQUAL "x")
      )
        hpx_local_add_config_define(
          HPX_HAVE_JEMALLOC_PREFIX ${HPXLocal_WITH_JEMALLOC_PREFIX}
        )
      endif()
    endif()
  endif()

endif()
