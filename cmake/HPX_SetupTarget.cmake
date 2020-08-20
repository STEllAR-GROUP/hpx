# Copyright (c) 2014      Thomas Heller
# Copyright (c) 2007-2018 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_policy(PUSH)

hpx_set_cmake_policy(CMP0054 NEW)
hpx_set_cmake_policy(CMP0060 NEW)

function(hpx_setup_target target)
  # retrieve arguments
  set(options
      EXPORT
      INSTALL
      INSTALL_HEADERS
      INTERNAL_FLAGS
      NOLIBS
      PLUGIN
      NONAMEPREFIX
      NOTLLKEYWORD
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
  set(multi_value_args DEPENDENCIES COMPONENT_DEPENDENCIES COMPILE_FLAGS
                       LINK_FLAGS INSTALL_FLAGS INSTALL_PDB
  )
  cmake_parse_arguments(
    target "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(NOT TARGET ${target})
    hpx_error("${target} does not represent a target")
  endif()

  # Figure out which type we want...
  if(target_TYPE)
    string(TOUPPER "${target_TYPE}" _type)
  else()
    get_target_property(type_prop ${target} TYPE)
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

  get_target_property(target_SOURCES ${target} SOURCES)

  # Manage files with .cu extension in case When Cuda Clang is used
  if(target_SOURCES AND HPX_WITH_CUDA_CLANG)
    foreach(source ${target_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        set_source_files_properties(
          ${source} PROPERTIES COMPILE_FLAGS "${HPX_CUDA_CLANG_FLAGS}"
        )
      endif()
    endforeach()
  endif()

  if(target_COMPILE_FLAGS)
    hpx_append_property(${target} COMPILE_FLAGS ${target_COMPILE_FLAGS})
  endif()

  if(target_LINK_FLAGS)
    hpx_append_property(${target} LINK_FLAGS ${target_LINK_FLAGS})
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
  if(HPX_WITH_STATIC_LINKING)
    set(target_STATIC_LINKING ON)
  else()
    set(_hpx_library_type)
    if(TARGET hpx)
      get_target_property(_hpx_library_type hpx TYPE)
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

  if("${_type}" STREQUAL "LIBRARY" OR "${_type}" STREQUAL "COMPONENT")
    if(DEFINED HPX_LIBRARY_VERSION AND DEFINED HPX_SOVERSION)
      # set properties of generated shared library
      set_target_properties(
        ${target} PROPERTIES VERSION ${HPX_LIBRARY_VERSION} SOVERSION
                                                            ${HPX_SOVERSION}
      )
    endif()
    if(NOT target_NONAMEPREFIX)
      hpx_set_lib_name(${target} ${name})
    endif()
    set_target_properties(
      ${target}
      PROPERTIES # create *nix style library versions + symbolic links
                 # allow creating static and shared libs without conflicts
                 CLEAN_DIRECT_OUTPUT 1 OUTPUT_NAME ${name}
    )
  endif()

  if("${_type}" STREQUAL "LIBRARY" AND target_PLUGIN)
    set(plugin_name "HPX_PLUGIN_NAME=hpx_${name}")
    target_link_libraries(${target} ${__tll_private} HPX::plugin)
  endif()

  if("${_type}" STREQUAL "COMPONENT")
    target_compile_definitions(
      ${target} PRIVATE "HPX_COMPONENT_NAME=hpx_${name}"
                        "HPX_COMPONENT_STRING=\"hpx_${name}\""
    )
    target_link_libraries(${target} ${__tll_private} HPX::component)
  endif()

  if(NOT target_NOLIBS)
    set(_wrap_main_deps)
    if("${_type}" STREQUAL "EXECUTABLE")
      set(_wrap_main_deps HPX::wrap_main)
    endif()
    target_link_libraries(${target} ${__tll_public} HPX::hpx ${_wrap_main_deps})
    hpx_handle_component_dependencies(target_COMPONENT_DEPENDENCIES)
    target_link_libraries(
      ${target} ${__tll_public} ${target_COMPONENT_DEPENDENCIES}
    )
  endif()

  target_link_libraries(${target} ${__tll_public} ${target_DEPENDENCIES})

  if(target_INTERNAL_FLAGS AND TARGET hpx_private_flags)
    target_link_libraries(${target} ${__tll_private} hpx_private_flags)
  endif()

  get_target_property(target_EXCLUDE_FROM_ALL ${target} EXCLUDE_FROM_ALL)

  if(target_EXPORT AND NOT target_EXCLUDE_FROM_ALL)
    hpx_export_targets(${target})
    set(install_export EXPORT HPXTargets)
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
