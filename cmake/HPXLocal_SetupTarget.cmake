# Copyright (c) 2014      Thomas Heller
# Copyright (c) 2007-2018 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_policy(PUSH)

include(HPXLocal_GeneratePackageUtils)

hpx_local_set_cmake_policy(CMP0054 NEW)
hpx_local_set_cmake_policy(CMP0060 NEW)

function(hpx_local_setup_target target)
  set(options
      EXPORT
      INSTALL
      INSTALL_HEADERS
      INTERNAL_FLAGS
      NOLIBS
      NONAMEPREFIX
      NOTLLKEYWORD
      UNITY_BUILD
  )
  set(one_value_args
      TYPE
      FOLDER
      NAME
      SOVERSION
      VERSION
      HPX_PREFIX
      HEADER_ROOT
  )
  set(multi_value_args DEPENDENCIES COMPILE_FLAGS LINK_FLAGS INSTALL_FLAGS
                       INSTALL_PDB
  )
  cmake_parse_arguments(
    target "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(NOT TARGET ${target})
    hpx_local_error("${target} does not represent a target")
  endif()

  # Figure out which type we want...
  if(target_TYPE)
    string(TOUPPER "${target_TYPE}" _type)
  else()
    hpx_local_get_target_property(type_prop ${target} TYPE)
    if(type_prop STREQUAL "STATIC_LIBRARY")
      set(_type "LIBRARY")
    endif()
    if(type_prop STREQUAL "MODULE_LIBRARY")
      set(_type "LIBRARY")
    endif()
    if(type_prop STREQUAL "SHARED_LIBRARY")
      set(_type "LIBRARY")
    endif()
    if(type_prop STREQUAL "EXECUTABLE")
      set(_type "EXECUTABLE")
    endif()
  endif()

  if(target_FOLDER)
    set_target_properties(${target} PROPERTIES FOLDER "${target_FOLDER}")
  endif()

  hpx_local_get_target_property(target_SOURCES ${target} SOURCES)

  if(target_COMPILE_FLAGS)
    hpx_local_append_property(${target} COMPILE_FLAGS ${target_COMPILE_FLAGS})
  endif()

  if(target_LINK_FLAGS)
    hpx_local_append_property(${target} LINK_FLAGS ${target_LINK_FLAGS})
  endif()

  if(target_NAME)
    set(name "${target_NAME}")
  else()
    set(name "${target}")
  endif()

  if(target_NOTLLKEYWORD)
    set(__tll_private)
    set(__tll_public)
  else()
    set(__tll_private PRIVATE)
    set(__tll_public PUBLIC)
  endif()

  set(target_STATIC_LINKING OFF)
  if(HPXLocal_WITH_STATIC_LINKING)
    set(target_STATIC_LINKING ON)
  else()
    set(_hpx_library_type)
    if(TARGET hpx)
      hpx_local_get_target_property(_hpx_library_type hpx TYPE)
    endif()

    if("${_hpx_library_type}" STREQUAL "STATIC_LIBRARY")
      set(target_STATIC_LINKING ON)
    endif()
  endif()

  if("${_type}" STREQUAL "EXECUTABLE")
    target_compile_definitions(
      ${target} PRIVATE "HPX_APPLICATION_NAME=${name}"
                        "HPX_APPLICATION_STRING=\"${name}\""
    )

    if(target_HPX_PREFIX)
      set(_prefix ${target_HPX_PREFIX})

      if(MSVC)
        string(REPLACE ";" ":" _prefix "${_prefix}")
      endif()

      target_compile_definitions(${target} PRIVATE "HPX_PREFIX=\"${_prefix}\"")
    endif()
  endif()

  if("${_type}" STREQUAL "LIBRARY")
    if(DEFINED HPXLocal_LIBRARY_VERSION AND DEFINED HPXLocal_SOVERSION)
      # set properties of generated shared library
      set_target_properties(
        ${target} PROPERTIES VERSION ${HPXLocal_LIBRARY_VERSION}
                             SOVERSION ${HPXLocal_SOVERSION}
      )
    endif()
    if(NOT target_NONAMEPREFIX)
      hpx_local_set_lib_name(${target} ${name})
    endif()
    set_target_properties(
      ${target}
      PROPERTIES # create *nix style library versions + symbolic links
                 # allow creating static and shared libs without conflicts
                 CLEAN_DIRECT_OUTPUT 1 OUTPUT_NAME ${name}
    )
  endif()

  if(NOT target_NOLIBS)
    target_link_libraries(${target} ${__tll_public} HPX::hpx_local)
  endif()

  target_link_libraries(${target} ${__tll_public} ${target_DEPENDENCIES})

  if(target_INTERNAL_FLAGS AND TARGET hpx_local_private_flags)
    target_link_libraries(${target} ${__tll_private} hpx_local_private_flags)
  endif()

  if(target_UNITY_BUILD)
    set_target_properties(${target} PROPERTIES UNITY_BUILD ON)
  endif()

  if(HPXLocal_WITH_PRECOMPILED_HEADERS_INTERNAL)
    target_precompile_headers(
      ${target} REUSE_FROM hpx_local_precompiled_headers
    )
  endif()

  hpx_local_get_target_property(
    target_EXCLUDE_FROM_ALL ${target} EXCLUDE_FROM_ALL
  )

  if(target_EXPORT AND NOT target_EXCLUDE_FROM_ALL)
    hpx_local_export_targets(${target})
    set(install_export EXPORT HPXLocalTargets)
  endif()

  if(target_INSTALL AND NOT target_EXCLUDE_FROM_ALL)
    install(TARGETS ${target} ${install_export} ${target_INSTALL_FLAGS})
    if(target_INSTALL_PDB)
      install(FILES ${target_INSTALL_PDB})
    endif()
    if(target_INSTALL_HEADERS AND (NOT target_HEADER_ROOT STREQUAL ""))
      install(
        DIRECTORY "${target_HEADER_ROOT}/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        COMPONENT ${name}
      )
    endif()
  endif()
endfunction()

cmake_policy(POP)
