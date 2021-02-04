# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(add_hpx_executable name)
  # retrieve arguments
  set(options
      EXCLUDE_FROM_ALL
      EXCLUDE_FROM_DEFAULT_BUILD
      AUTOGLOB
      INTERNAL_FLAGS
      NOLIBS
      NOHPX_INIT
      UNITY_BUILD
  )
  set(one_value_args
      INI
      FOLDER
      SOURCE_ROOT
      HEADER_ROOT
      SOURCE_GLOB
      HEADER_GLOB
      OUTPUT_SUFFIX
      INSTALL_SUFFIX
      LANGUAGE
      HPX_PREFIX
  )
  set(multi_value_args
      SOURCES
      HEADERS
      AUXILIARY
      DEPENDENCIES
      COMPONENT_DEPENDENCIES
      COMPILE_FLAGS
      LINK_FLAGS
  )
  cmake_parse_arguments(
    ${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(NOT ${name}_LANGUAGE)
    set(${name}_LANGUAGE CXX)
  endif()

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  hpx_debug(
    "Add executable ${name}: ${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}"
  )

  if(NOT ${name}_HEADER_ROOT)
    set(${name}_HEADER_ROOT ".")
  endif()
  hpx_debug(
    "Add executable ${name}: ${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}"
  )

  # Collect sources and headers from the given (current) directory
  # (recursively), but only if AUTOGLOB flag is specified.
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
    hpx_debug(
      "Add executable ${name}: ${name}_SOURCE_GLOB: ${${name}_SOURCE_GLOB}"
    )

    add_hpx_library_sources(
      ${name}_executable GLOB_RECURSE GLOBS "${${name}_SOURCE_GLOB}"
    )

    set(${name}_SOURCES ${${name}_executable_SOURCES})
    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_executable_SOURCES}
    )

    if(NOT ${name}_HEADER_GLOB)
      set(${name}_HEADER_GLOB "${${name}_HEADER_ROOT}/*.hpp"
                              "${${name}_HEADER_ROOT}/*.h"
      )
    endif()
    hpx_debug(
      "Add executable ${name}: ${name}_HEADER_GLOB: ${${name}_HEADER_GLOB}"
    )

    add_hpx_library_headers(
      ${name}_executable GLOB_RECURSE GLOBS "${${name}_HEADER_GLOB}"
    )

    set(${name}_HEADERS ${${name}_executable_HEADERS})
    add_hpx_source_group(
      NAME ${name}
      CLASS "Header Files"
      ROOT ${${name}_HEADER_ROOT}
      TARGETS ${${name}_executable_HEADERS}
    )
  else()
    add_hpx_library_sources_noglob(
      ${name}_executable SOURCES "${${name}_SOURCES}"
    )

    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_executable_SOURCES}
    )

    add_hpx_library_headers_noglob(
      ${name}_executable HEADERS "${${name}_HEADERS}"
    )

    add_hpx_source_group(
      NAME ${name}
      CLASS "Header Files"
      ROOT ${${name}_HEADER_ROOT}
      TARGETS ${${name}_executable_HEADERS}
    )
  endif()

  set(${name}_SOURCES ${${name}_executable_SOURCES})
  set(${name}_HEADERS ${${name}_executable_HEADERS})

  hpx_print_list(
    "DEBUG" "Add executable ${name}: Sources for ${name}" ${name}_SOURCES
  )
  hpx_print_list(
    "DEBUG" "Add executable ${name}: Headers for ${name}" ${name}_HEADERS
  )
  hpx_print_list(
    "DEBUG" "Add executable ${name}: Dependencies for ${name}"
    ${name}_DEPENDENCIES
  )
  hpx_print_list(
    "DEBUG" "Add executable ${name}: Component dependencies for ${name}"
    ${name}_COMPONENT_DEPENDENCIES
  )

  set(_target_flags)

  # add the executable build target
  if(${name}_EXCLUDE_FROM_ALL)
    set(exclude_from_all EXCLUDE_FROM_ALL TRUE)
  else()
    set(install_destination ${CMAKE_INSTALL_BINDIR})
    if(${name}_INSTALL_SUFFIX)
      set(install_destination ${${name}_INSTALL_SUFFIX})
    endif()
    set(_target_flags INSTALL INSTALL_FLAGS DESTINATION ${install_destination})
    # install PDB if needed
    if(MSVC)
      # cmake-format: off
      set(_target_flags
          ${_target_flags}
          INSTALL_PDB $<TARGET_PDB_FILE:${name}>
            DESTINATION ${install_destination}
          CONFIGURATIONS Debug RelWithDebInfo
          OPTIONAL
      )
      # cmake-format: on
    endif()
  endif()

  if(${name}_EXCLUDE_FROM_DEFAULT_BUILD)
    set(exclude_from_all ${exclude_from_all} EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  endif()

  # Manage files with .cu extension in case When Cuda Clang is used
  if(HPX_WITH_CUDA_CLANG OR HPX_WITH_HIP)
    foreach(source ${${name}_SOURCES})
      get_filename_component(extension ${source} EXT)
      if(${extension} STREQUAL ".cu")
        message(${extension})
        set_source_files_properties(${source} PROPERTIES LANGUAGE CXX)
      endif()
    endforeach()
  endif()

  if(HPX_WITH_CUDA AND NOT HPX_WITH_CUDA_CLANG)
    cuda_add_executable(
      ${name} ${${name}_SOURCES} ${${name}_HEADERS} ${${name}_AUXILIARY}
    )
  else()
    add_executable(
      ${name} ${${name}_SOURCES} ${${name}_HEADERS} ${${name}_AUXILIARY}
    )
  endif()

  if(${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(
        ${name}
        PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY_RELEASE
          "${PROJECT_BINARY_DIR}/Release/bin/${${name}_OUTPUT_SUFFIX}"
          RUNTIME_OUTPUT_DIRECTORY_DEBUG
          "${PROJECT_BINARY_DIR}/Debug/bin/${${name}_OUTPUT_SUFFIX}"
          RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL
          "${PROJECT_BINARY_DIR}/MinSizeRel/bin/${${name}_OUTPUT_SUFFIX}"
          RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO
          "${PROJECT_BINARY_DIR}/RelWithDebInfo/bin/${${name}_OUTPUT_SUFFIX}"
      )
    else()
      set_target_properties(
        ${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                           "${PROJECT_BINARY_DIR}/bin/${${name}_OUTPUT_SUFFIX}"
      )
    endif()
  endif()

  set_target_properties(
    ${name} PROPERTIES OUTPUT_NAME "${HPX_WITH_EXECUTABLE_PREFIX}${name}"
  )

  if(exclude_from_all)
    set_target_properties(${name} PROPERTIES ${exclude_from_all})
  endif()

  if(${${name}_NOLIBS})
    set(_target_flags ${_target_flags} NOLIBS)
  endif()

  if(${${name}_NOHPX_INIT})
    set(_target_flags ${_target_flags} NOHPX_INIT)
  endif()

  if(${name}_INTERNAL_FLAGS)
    set(_target_flags ${_target_flags} INTERNAL_FLAGS)
  endif()

  if(${name}_UNITY_BUILD)
    set(_target_flags ${_target_flags} UNITY_BUILD)
  endif()

  hpx_setup_target(
    ${name}
    TYPE EXECUTABLE
    FOLDER ${${name}_FOLDER}
    COMPILE_FLAGS ${${name}_COMPILE_FLAGS}
    LINK_FLAGS ${${name}_LINK_FLAGS}
    DEPENDENCIES ${${name}_DEPENDENCIES}
    COMPONENT_DEPENDENCIES ${${name}_COMPONENT_DEPENDENCIES}
    HPX_PREFIX ${${name}_HPX_PREFIX} ${_target_flags}
  )
endfunction()
