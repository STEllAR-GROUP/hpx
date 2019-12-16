# Copyright (c) 2007-2019 Hartmut Kaiser
# Copyright (c) 2011-2014 Thomas Heller
# Copyright (c) 2007-2008 Chirag Dekate
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2011 Vinay C Amatya
# Copyright (c)      2013 Jeroen Habraken
# Copyright (c) 2014-2016 Andreas Schaefer
# Copyright (c) 2017      Abhimanyu Rawat
# Copyright (c) 2017      Google
# Copyright (c) 2017      Taeguk Kwon
# Copyright (c) 2018 Christopher Hinz
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if(HPX_WITH_APEX)
  if(NOT HPX_FIND_PACKAGE)
    set(_hpx_apex_no_update)
    if(HPX_WITH_APEX_NO_UPDATE)
      set(_hpx_apex_no_update NO_UPDATE)
    endif()

    # We want to track parent dependencies
    hpx_add_config_define(HPX_HAVE_THREAD_PARENT_REFERENCE)

    # If APEX_ROOT not specified, local clone into hpx source dir
    if (NOT APEX_ROOT)
      # handle APEX library
      include(GitExternal)
      git_external(apex
        https://github.com/khuck/xpress-apex.git
        ${HPX_WITH_APEX_TAG}
        ${_hpx_apex_no_update}
        VERBOSE)
      if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apex)
        set(APEX_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/apex)
      else()
        hpx_error("Apex could not be found and HPX_WITH_APEX=On")
      endif()
    endif(NOT APEX_ROOT)

    LIST(APPEND CMAKE_MODULE_PATH "${APEX_ROOT}/cmake/Modules")
    add_subdirectory(${APEX_ROOT}/src/apex ${CMAKE_BINARY_DIR}/apex/src/apex)
    if(AMPLIFIER_FOUND)
      hpx_error("AMPLIFIER_FOUND has been set. Please disable the use of the \
      Intel Amplifier (WITH_AMPLIFIER=Off) in order to use Apex")
    endif()
  endif()

  add_library(hpx::apex INTERFACE IMPORTED)
  # System has been removed when passing at set_property for cmake < 3.11
  get_target_property(_apex_include_directories apex INTERFACE_INCLUDE_DIRECTORIES)
  set_property(TARGET hpx::apex PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${_apex_include_directories})
  set_property(TARGET hpx::apex PROPERTY INTERFACE_LINK_LIBRARIES apex)
  if(UNIX AND NOT APPLE)
    set_property(TARGET hpx::apex PROPERTY INTERFACE_LINK_OPTIONS "-Wl,-no-as-needed")
  endif()

  # handle optional ITTNotify library (private dependency, skip when called in find_package(HPX))
  if(HPX_WITH_ITTNOTIFY AND NOT HPX_FIND_PACKAGE)
    add_subdirectory(${APEX_ROOT}/src/ITTNotify)
    if(NOT ITTNOTIFY_FOUND)
      hpx_error("ITTNotify could not be found and HPX_WITH_ITTNOTIFY=On")
    endif()

    add_library(hpx::ittnotify INTERFACE IMPORTED)
    # System has been removed when passing at set_property for cmake < 3.11
    set_property(TARGET hpx::ittnotify PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${ITTNOTIFY_SOURCE_DIR})
    set_property(TARGET hpx::apex APPEND PROPERTY INTERFACE_LINK_LIBRARIES hpx::ittnotify)
    hpx_add_config_define(HPX_HAVE_ITTNOTIFY 1)
  endif()
endif()
