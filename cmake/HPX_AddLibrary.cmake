# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDLIBRARY_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments
            Install)

macro(add_hpx_library name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "SOURCES;HEADERS;DEPENDENCIES;COMPONENT_DEPENDENCIES;FOLDER;SOURCE_ROOT;HEADER_ROOT;SOURCE_GLOB;HEADER_GLOB"
    "ESSENTIAL;NOLIBS;AUTOGLOB;STATIC" ${ARGN})

  if(${${name}_AUTOGLOB})
    if(NOT ${name}_SOURCE_ROOT)
      set(${name}_SOURCE_ROOT ".")
    endif()
    hpx_debug("add_library.${name}" "${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

    if(NOT ${name}_HEADER_ROOT)
      set(${name}_HEADER_ROOT ".")
    endif()
    hpx_debug("add_library.${name}" "${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

    if(NOT ${name}_SOURCE_GLOB)
      set(${name}_SOURCE_GLOB "${${name}_SOURCE_ROOT}/*.cpp"
                              "${${name}_SOURCE_ROOT}/*.f"
                              "${${name}_SOURCE_ROOT}/*.F"
                              "${${name}_SOURCE_ROOT}/*.f77"
                              "${${name}_SOURCE_ROOT}/*.F77"
                              "${${name}_SOURCE_ROOT}/*.f90"
                              "${${name}_SOURCE_ROOT}/*.F90"
                              "${${name}_SOURCE_ROOT}/*.f95"
                              "${${name}_SOURCE_ROOT}/*.F95")
    endif()
    hpx_debug("add_library.${name}" "${name}_SOURCE_GLOB: ${${name}_SOURCE_GLOB}")

    add_hpx_library_sources(${name}_lib
      GLOB_RECURSE GLOBS "${${name}_SOURCE_GLOB}")

    set(${name}_SOURCES ${${name}_lib_SOURCES})
    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_SOURCES})

    if(NOT ${name}_HEADER_GLOB AND NOT ${${name}_HEADER_GLOB} STREQUAL "")
      set(${name}_HEADER_GLOB "${${name}_HEADER_ROOT}/*.hpp")
    endif()
    hpx_debug("add_library.${name}" "${name}_HEADER_GLOB: ${${name}_HEADER_GLOB}")

    if(NOT ${${name}_HEADER_GLOB} STREQUAL "")
      add_hpx_library_headers(${name}_lib
        GLOB_RECURSE GLOBS "${${name}_HEADER_GLOB}")

      set(${name}_HEADERS ${${name}_lib_HEADERS})
      add_hpx_library_headers(${name}_library
        GLOB_RECURSE GLOBS "${${name}_HEADER_GLOB}")

      add_hpx_source_group(
        NAME ${name}
        CLASS "Header Files"
        ROOT ${${name}_HEADER_ROOT}
        TARGETS ${${name}_HEADERS})
    endif()
  endif()

  hpx_print_list("DEBUG" "add_library.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_library.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_library.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_library.${name}" "Component dependencies for ${name}" ${name}_COMPONENT_DEPENDENCIES)

  if(${${name}_STATIC})
    set(${name}_lib_linktype STATIC)
  else()
    set(${name}_lib_linktype SHARED)
  endif()

  if(NOT MSVC)
    if(${${name}_ESSENTIAL})
      add_library(${name}_lib ${${name}_lib_linktype}
        ${${name}_SOURCES} ${${name}_HEADERS})
    else()
      add_library(${name}_lib ${${name}_lib_linktype} EXCLUDE_FROM_ALL
        ${${name}_SOURCES} ${${name}_HEADERS})
    endif()
  else()
    if(${${name}_ESSENTIAL})
      add_library(${name}_lib ${${name}_lib_linktype} ${${name}_SOURCES})
    else()
      add_library(${name}_lib ${${name}_lib_linktype} EXCLUDE_FROM_ALL ${${name}_SOURCES})
    endif()
  endif()

  hpx_handle_component_dependencies(${name}_COMPONENT_DEPENDENCIES)

  if(HPX_FOUND AND "${HPX_BUILD_TYPE}" STREQUAL "Debug")
    set(hpx_lib hpx${HPX_DEBUG_POSTFIX} hpx_serialization${HPX_DEBUG_POSTFIX})
  else()
    set(hpx_lib hpx hpx_serialization)
  endif()

  set(prefix "")

  if(NOT MSVC)
    if(NOT ${${name}_NOLIBS})
      target_link_libraries(${name}_lib
        ${${name}_DEPENDENCIES} ${${name}_COMPONENT_DEPENDENCIES} ${hpx_lib})
      set_property(TARGET ${name}_lib APPEND
                   PROPERTY COMPILE_DEFINITIONS
                   "BOOST_ENABLE_ASSERT_HANDLER")
    else()
      target_link_libraries(${name}_lib
        ${${name}_DEPENDENCIES} ${${name}_COMPONENT_DEPENDENCIES})
    endif()
    set(prefix "hpx_")
  else()
    if(NOT ${${name}_NOLIBS})
      target_link_libraries(${name}_lib
        ${${name}_DEPENDENCIES} ${${name}_COMPONENT_DEPENDENCIES} ${hpx_lib})
    else()
      target_link_libraries(${name}_lib
        ${${name}_DEPENDENCIES} ${${name}_COMPONENT_DEPENDENCIES})
    endif()
  endif()

  # set properties of generated shared library
  set_target_properties(${name}_lib PROPERTIES
    # create *nix style library versions + symbolic links
    VERSION ${HPX_VERSION}
    SOVERSION ${HPX_SOVERSION}
    # allow creating static and shared libs without conflicts
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME ${prefix}${name}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO})

  if(HPX_COMPILE_FLAGS)
    set_property(TARGET ${name}_lib APPEND PROPERTY COMPILE_FLAGS ${HPX_COMPILE_FLAGS})
    if(NOT MSVC)
      set_property(TARGET ${name}_lib APPEND PROPERTY LINK_FLAGS ${HPX_COMPILE_FLAGS})
    endif()
  endif()

  if(NOT MSVC)
    set_target_properties(${name}_lib
                          PROPERTIES SKIP_BUILD_RPATH TRUE
                                     BUILD_WITH_INSTALL_RPATH TRUE
                                     INSTALL_RPATH_USE_LINK_PATH TRUE
                                     INSTALL_RPATH ${HPX_RPATH})
  endif()

  if(${name}_FOLDER)
    set_target_properties(${name}_lib PROPERTIES FOLDER ${${name}_FOLDER})
  endif()

  set_property(TARGET ${name}_lib APPEND
               PROPERTY COMPILE_DEFINITIONS "HPX_LIBRARY_EXPORTS")

  if(NOT HPX_NO_INSTALL)
    hpx_library_install(${name}_lib)
  endif()
endmacro()

