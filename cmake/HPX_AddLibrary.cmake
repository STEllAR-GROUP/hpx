# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_library name)
  # retrieve arguments
  set(options
      EXCLUDE_FROM_ALL
      INTERNAL_FLAGS
      NOLIBS
      NOEXPORT
      AUTOGLOB
      STATIC
      PLUGIN
      NONAMEPREFIX
      UNITY_BUILD
  )
  set(one_value_args
      FOLDER
      SOURCE_ROOT
      HEADER_ROOT
      SOURCE_GLOB
      HEADER_GLOB
      OUTPUT_SUFFIX
      INSTALL_SUFFIX
  )
  set(multi_value_args
      SOURCES
      HEADERS
      AUXILIARY
      DEPENDENCIES
      COMPONENT_DEPENDENCIES
      COMPILER_FLAGS
      LINK_FLAGS
  )
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  hpx_debug("add_library.${name}" "${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

  if(NOT ${name}_HEADER_ROOT)
    set(${name}_HEADER_ROOT ".")
  endif()
  hpx_debug("add_library.${name}" "${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

  if(${${name}_AUTOGLOB})
    if(NOT ${name}_SOURCE_GLOB)
      set(${name}_SOURCE_GLOB
          "${${name}_SOURCE_ROOT}/*.cpp"
          "${${name}_SOURCE_ROOT}/*.c"
          "${${name}_SOURCE_ROOT}/*.f"
          "${${name}_SOURCE_ROOT}/*.F"
          "${${name}_SOURCE_ROOT}/*.f77"
          "${${name}_SOURCE_ROOT}/*.F77"
          "${${name}_SOURCE_ROOT}/*.f90"
          "${${name}_SOURCE_ROOT}/*.F90"
          "${${name}_SOURCE_ROOT}/*.f95"
          "${${name}_SOURCE_ROOT}/*.F95"
      )
    endif()
    hpx_debug("add_library.${name}"
              "${name}_SOURCE_GLOB: ${${name}_SOURCE_GLOB}"
    )

    add_hpx_library_sources(${name} GLOB_RECURSE GLOBS "${${name}_SOURCE_GLOB}")

    set(${name}_SOURCES ${${name}_SOURCES})
    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_SOURCES}
    )

    if(NOT ${name}_HEADER_GLOB)
      set(${name}_HEADER_GLOB "${${name}_HEADER_ROOT}/*.hpp"
                              "${${name}_HEADER_ROOT}/*.h"
      )
    endif()
    hpx_debug("add_library.${name}"
              "${name}_HEADER_GLOB: ${${name}_HEADER_GLOB}"
    )

    add_hpx_library_headers(${name} GLOB_RECURSE GLOBS "${${name}_HEADER_GLOB}")

    set(${name}_HEADERS ${${name}_HEADERS})
    add_hpx_source_group(
      NAME ${name}
      CLASS "Header Files"
      ROOT ${${name}_HEADER_ROOT}
      TARGETS ${${name}_HEADERS}
    )
  else()
    add_hpx_library_sources_noglob(${name} SOURCES "${${name}_SOURCES}")

    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_SOURCES}
    )

    add_hpx_library_headers_noglob(${name} HEADERS "${${name}_HEADERS}")

    add_hpx_source_group(
      NAME ${name}
      CLASS "Header Files"
      ROOT ${${name}_HEADER_ROOT}
      TARGETS ${${name}_HEADERS}
    )
  endif()

  hpx_print_list(
    "DEBUG" "add_library.${name}" "Sources for ${name}" ${name}_SOURCES
  )
  hpx_print_list(
    "DEBUG" "add_library.${name}" "Headers for ${name}" ${name}_HEADERS
  )
  hpx_print_list(
    "DEBUG" "add_library.${name}" "Dependencies for ${name}"
    ${name}_DEPENDENCIES
  )
  hpx_print_list(
    "DEBUG" "add_library.${name}" "Component dependencies for ${name}"
    ${name}_COMPONENT_DEPENDENCIES
  )

  set(exclude_from_all)

  if(${name}_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  else()
    if(${name}_PLUGIN AND NOT HPX_WITH_STATIC_LINKING)
      if(MSVC)
        set(library_install_destination ${CMAKE_INSTALL_BINDIR}/hpx)
      else()
        set(library_install_destination ${CMAKE_INSTALL_LIBDIR}/hpx)
      endif()
      set(archive_install_destination ${CMAKE_INSTALL_LIBDIR}/hpx)
      set(runtime_install_destination ${CMAKE_INSTALL_BINDIR}/hpx)
      set(${name}_OUTPUT_SUFFIX hpx)
    else()
      if(MSVC)
        set(library_install_destination ${CMAKE_INSTALL_BINDIR})
      else()
        set(library_install_destination ${CMAKE_INSTALL_LIBDIR})
      endif()
      set(archive_install_destination ${CMAKE_INSTALL_LIBDIR})
      set(runtime_install_destination ${CMAKE_INSTALL_BINDIR})
    endif()
    if(${name}_INSTALL_SUFFIX)
      set(library_install_destination ${${name}_INSTALL_SUFFIX})
      set(archive_install_destination ${${name}_INSTALL_SUFFIX})
      set(runtime_install_destination ${${name}_INSTALL_SUFFIX})
    endif()
    # cmake-format: off
    set(_target_flags
        INSTALL INSTALL_FLAGS
          LIBRARY DESTINATION ${library_install_destination}
          ARCHIVE DESTINATION ${archive_install_destination}
          RUNTIME DESTINATION ${runtime_install_destination}
    )
    # cmake-format: on

    # install PDB if needed
    if(MSVC
       AND NOT ${name}_STATIC
       AND NOT HPX_WITH_STATIC_LINKING
    )
      # cmake-format: off
      set(_target_flags
          ${_target_flags}
          INSTALL_PDB $<TARGET_PDB_FILE:${name}>
            DESTINATION ${runtime_install_destination}
          CONFIGURATIONS Debug RelWithDebInfo
          OPTIONAL
      )
      # cmake-format: on
    endif()
  endif()

  if(${name}_PLUGIN)
    set(_target_flags ${_target_flags} PLUGIN)
  endif()
  if(${name}_NONAMEPREFIX)
    set(_target_flags ${_target_flags} NONAMEPREFIX)
  endif()

  if(${name}_STATIC)
    set(${name}_linktype STATIC)
  else()
    if(HPX_WITH_STATIC_LINKING)
      set(${name}_linktype STATIC)
    else()
      set(${name}_linktype SHARED)
    endif()
  endif()

  # Manage files with .cu extension in case When Cuda Clang is used
  if(HPX_WITH_CUDA_CLANG OR HPX_WITH_HIP)
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        set_source_files_properties(${source} PROPERTIES LANGUAGE CXX)
      endif()
    endforeach()
  endif()

  if(HPX_WITH_CUDA AND NOT HPX_WITH_CUDA_CLANG)
    cuda_add_library(
      ${name} ${${name}_linktype} ${exclude_from_all} ${${name}_SOURCES}
      ${${name}_HEADERS} ${${name}_AUXILIARY}
    )
  else()
    add_library(
      ${name} ${${name}_linktype} ${exclude_from_all} ${${name}_SOURCES}
              ${${name}_HEADERS} ${${name}_AUXILIARY}
    )
  endif()

  if(${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(
        ${name}
        PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY_RELEASE
          "${PROJECT_BINARY_DIR}/Release/bin/${${name}_OUTPUT_SUFFIX}"
          LIBRARY_OUTPUT_DIRECTORY_RELEASE
          "${PROJECT_BINARY_DIR}/Release/bin/${${name}_OUTPUT_SUFFIX}"
          ARCHIVE_OUTPUT_DIRECTORY_RELEASE
          "${PROJECT_BINARY_DIR}/Release/lib/${${name}_OUTPUT_SUFFIX}"
          RUNTIME_OUTPUT_DIRECTORY_DEBUG
          "${PROJECT_BINARY_DIR}/Debug/bin/${${name}_OUTPUT_SUFFIX}"
          LIBRARY_OUTPUT_DIRECTORY_DEBUG
          "${PROJECT_BINARY_DIR}/Debug/bin/${${name}_OUTPUT_SUFFIX}"
          ARCHIVE_OUTPUT_DIRECTORY_DEBUG
          "${PROJECT_BINARY_DIR}/Debug/lib/${${name}_OUTPUT_SUFFIX}"
          RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL
          "${PROJECT_BINARY_DIR}/MinSizeRel/bin/${${name}_OUTPUT_SUFFIX}"
          LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL
          "${PROJECT_BINARY_DIR}/MinSizeRel/bin/${${name}_OUTPUT_SUFFIX}"
          ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL
          "${PROJECT_BINARY_DIR}/MinSizeRel/lib/${${name}_OUTPUT_SUFFIX}"
          RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO
          "${PROJECT_BINARY_DIR}/RelWithDebInfo/bin/${${name}_OUTPUT_SUFFIX}"
          LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO
          "${PROJECT_BINARY_DIR}/RelWithDebInfo/bin/${${name}_OUTPUT_SUFFIX}"
          ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO
          "${PROJECT_BINARY_DIR}/RelWithDebInfo/lib/${${name}_OUTPUT_SUFFIX}"
      )
    else()
      set_target_properties(
        ${name}
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                   "${PROJECT_BINARY_DIR}/bin/${${name}_OUTPUT_SUFFIX}"
                   LIBRARY_OUTPUT_DIRECTORY
                   "${PROJECT_BINARY_DIR}/lib/${${name}_OUTPUT_SUFFIX}"
                   ARCHIVE_OUTPUT_DIRECTORY
                   "${PROJECT_BINARY_DIR}/lib/${${name}_OUTPUT_SUFFIX}"
      )
    endif()
  endif()

  # get public and private compile options that hpx needs
  if(${${name}_NOLIBS})
    set(_target_flags ${_target_flags} NOLIBS)
  endif()

  if(NOT ${${name}_NOEXPORT})
    set(_target_flags ${_target_flags} EXPORT)
  endif()

  if(${name}_INTERNAL_FLAGS)
    set(_target_flags ${_target_flags} INTERNAL_FLAGS)
  endif()

  if(${name}_UNITY_BUILD)
    set(_target_flags ${_target_flags} UNITY_BUILD)
  endif()

  hpx_setup_target(
    ${name}
    TYPE LIBRARY
    NAME ${name}
    FOLDER ${${name}_FOLDER}
    COMPILE_FLAGS ${${name}_COMPILE_FLAGS}
    LINK_FLAGS ${${name}_LINK_FLAGS}
    DEPENDENCIES ${${name}_DEPENDENCIES}
    COMPONENT_DEPENDENCIES ${${name}_COMPONENT_DEPENDENCIES} ${_target_flags}
  )

endfunction()
