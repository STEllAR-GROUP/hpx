# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

macro(add_hpx_component name)
  # retrieve arguments
  set(options ESSENTIAL AUTOGLOB NOLIBS STATIC)
  set(one_value_args INI FOLDER SOURCE_ROOT HEADER_ROOT SOURCE_GLOB HEADER_GLOB OUTPUT_SUFFIX INSTALL_SUFFIX LANGUAGE)
  set(multi_value_args SOURCES HEADERS DEPENDENCIES COMPONENT_DEPENDENCIES COMPILE_FLAGS LINK_FLAGS)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT ${name}_LANGUAGE)
    set(${name}_LANGUAGE CXX)
  endif()

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  hpx_debug("Add component ${name}: ${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

  if(NOT ${name}_HEADER_ROOT)
    set(${name}_HEADER_ROOT ".")
  endif()
  hpx_debug("Add component ${name}: ${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

  # Collect sources and headers from the given (current) directory
  # (recursively), but only if AUTOGLOB flag is specified.
  if(${${name}_AUTOGLOB})
    if(NOT ${name}_SOURCE_GLOB)
      set(${name}_SOURCE_GLOB "${${name}_SOURCE_ROOT}/*.cpp"
                              "${${name}_SOURCE_ROOT}/*.c"
                              "${${name}_SOURCE_ROOT}/*.f"
                              "${${name}_SOURCE_ROOT}/*.F"
                              "${${name}_SOURCE_ROOT}/*.f77"
                              "${${name}_SOURCE_ROOT}/*.F77"
                              "${${name}_SOURCE_ROOT}/*.f90"
                              "${${name}_SOURCE_ROOT}/*.F90"
                              "${${name}_SOURCE_ROOT}/*.f95"
                              "${${name}_SOURCE_ROOT}/*.F95")
    endif()
    hpx_debug("Add component ${name}: ${name}_SOURCE_GLOB: ${${name}_SOURCE_GLOB}")

    add_hpx_library_sources(${name}_component
      GLOB_RECURSE GLOBS "${${name}_SOURCE_GLOB}")

    set(${name}_SOURCES ${${name}_component_SOURCES})
    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_component_SOURCES})

    if(NOT ${name}_HEADER_GLOB)
      set(${name}_HEADER_GLOB "${${name}_HEADER_ROOT}/*.hpp"
                              "${${name}_HEADER_ROOT}/*.h")
    endif()
    hpx_debug("Add component ${name}: ${name}_HEADER_GLOB: ${${name}_HEADER_GLOB}")

    add_hpx_library_headers(${name}_component
      GLOB_RECURSE GLOBS "${${name}_HEADER_GLOB}")

    set(${name}_HEADERS ${${name}_component_HEADERS})
    add_hpx_source_group(
      NAME ${name}
      CLASS "Header Files"
      ROOT ${${name}_HEADER_ROOT}
      TARGETS ${${name}_component_HEADERS})
  else()
    add_hpx_library_sources_noglob(${name}_component
        SOURCES "${${name}_SOURCES}")

    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_component_SOURCES})

    add_hpx_library_headers_noglob(${name}_component
        HEADERS "${${name}_HEADERS}")

    add_hpx_source_group(
      NAME ${name}
      CLASS "Header Files"
      ROOT ${${name}_HEADER_ROOT}
      TARGETS ${${name}_component_HEADERS})
  endif()

  set(${name}_SOURCES ${${name}_component_SOURCES})
  set(${name}_HEADERS ${${name}_component_HEADERS})

  hpx_print_list("DEBUG" "Add component ${name}: Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "Add component ${name}: Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "Add component ${name}: Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "Add component ${name}: Component dependencies for ${name}" ${name}_COMPONENT_DEPENDENCIES)
  hpx_print_list("DEBUG" "Add component ${name}: Configuration files for ${name}" ${name}_INI)

  set(exclude_from_all)
  if(NOT ${name}_ESSENTIAL)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  add_library(${name}_component SHARED ${exclude_from_all}
    ${${name}_SOURCES} ${${name}_HEADERS})

  hpx_handle_component_dependencies(${name}_COMPONENT_DEPENDENCIES)

  set(hpx_libs "")

  if(NOT ${name}_NOLIBS)
    set(hpx_libs ${hpx_LIBRARIES})

    set(hpx_libs hpx hpx_serialization ${hpx_libs})
  endif()

  list(REMOVE_DUPLICATES hpx_libs)

  target_link_libraries(${name}_component
    ${${name}_DEPENDENCIES} ${${name}_COMPONENT_DEPENDENCIES} ${hpx_libs})

  # set properties of generated shared library
  set_target_properties(${name}_component PROPERTIES
    # create *nix style library versions + symbolic links
    VERSION ${HPX_LIBRARY_VERSION}
    SOVERSION ${HPX_SOVERSION}
    # allow creating static and shared libs without conflicts
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME ${name})

  if(${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties("${name}_component" PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/${${name}_OUTPUT_SUFFIX}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/${${name}_OUTPUT_SUFFIX}"
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/MinSizeRel/${${name}_OUTPUT_SUFFIX}"
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/${${name}_OUTPUT_SUFFIX}"
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/${${name}_OUTPUT_SUFFIX}"
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/${${name}_OUTPUT_SUFFIX}"
        ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/MinSizeRel/${${name}_OUTPUT_SUFFIX}"
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/${${name}_OUTPUT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/${${name}_OUTPUT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Debug/${${name}_OUTPUT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/MinSizeRel/${${name}_OUTPUT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/RelWithDebInfo/${${name}_OUTPUT_SUFFIX}")
    else()
      set_target_properties("${name}_component" PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${${name}_OUTPUT_SUFFIX}"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${${name}_OUTPUT_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${${name}_OUTPUT_SUFFIX}")
    endif()
  endif()

  if(${name}_COMPILE_FLAGS)
    hpx_append_property(${name}_component COMPILE_FLAGS ${${name}_COMPILE_FLAGS})
  endif()

  if(${name}_LINK_FLAGS)
    hpx_append_property(${name}_component LINK_FLAGS ${${name}_LINK_FLAGS})
  endif()

  if(HPX_${${name}_LANGUAGE}_COMPILE_FLAGS)
    set_property(TARGET ${name}_component APPEND
      PROPERTY COMPILE_FLAGS ${HPX_${${name}_LANGUAGE}_COMPILE_FLAGS})
  endif()

  if(${name}_FOLDER)
    set_target_properties(${name}_component PROPERTIES FOLDER "${${name}_FOLDER}")
  endif()

  set_property(TARGET ${name}_component APPEND
               PROPERTY COMPILE_DEFINITIONS
               "HPX_COMPONENT_NAME=${name}"
               "HPX_COMPONENT_STRING=\"${name}\""
               "HPX_COMPONENT_EXPORTS")

#   if(NOT HPX_NO_INSTALL)
#     if(${name}_INSTALL_SUFFIX)
#       hpx_library_install("${name}_component" "${${name}_INSTALL_SUFFIX}")
#     else()
#       hpx_library_install(${name}_component ${LIB}/hpx)
#     endif()
#
#     foreach(target ${${name}_INI})
#       hpx_debug("add_component.${name}" "installing ini: ${name}")
#       hpx_ini_install(${target})
#     endforeach()
#   endif()
endmacro()

