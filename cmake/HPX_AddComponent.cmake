# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDCOMPONENT_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments
            HandleComponentDependencies
            Install
            AddSourceGroup)

macro(add_hpx_component name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "SOURCES;HEADERS;DEPENDENCIES;COMPONENT_DEPENDENCIES;INI;FOLDER;HEADER_ROOT;SOURCE_ROOT;HEADER_GLOB;SOURCE_GLOB"
    "ESSENTIAL;NOLIBS;AUTOGLOB" ${ARGN})

  # Collect sources and headers from the given (current) directory
  # (recursively), but only if AUTOGLOB flag is specified.
  if(${${name}_AUTOGLOB})
    if(NOT ${name}_SOURCE_ROOT)
      set(${name}_SOURCE_ROOT ".")
    endif()
    hpx_debug("add_component.${name}_component" "${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

    if(NOT ${name}_SOURCE_GLOB)
      set(${name}_SOURCE_GLOB "${${name}_SOURCE_ROOT}/*.cpp")
    endif()
    hpx_debug("add_component.${name}_component" "${name}_SOURCE_GLOB: ${${name}_SOURCE_GLOB}")

    add_hpx_library_sources(${name}_component
      GLOB_RECURSE GLOBS "${${name}_SOURCE_GLOB}")

    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_component_SOURCES})

    set(${name}_SOURCES ${${name}_component_SOURCES})

    if(NOT ${name}_HEADER_ROOT)
      set(${name}_HEADER_ROOT ".")
    endif()
    hpx_debug("add_component.${name}_component" "${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

    if(NOT ${name}_HEADER_GLOB)
      set(${name}_HEADER_GLOB "${${name}_HEADER_ROOT}/*.hpp")
    endif()
    hpx_debug("add_component.${name}_component" "${name}_HEADER_GLOB: ${${name}_HEADER_GLOB}")

    add_hpx_library_headers(${name}_component
      GLOB_RECURSE GLOBS "${${name}_HEADER_GLOB}")

    add_hpx_source_group(
      NAME ${name}
      CLASS "Header Files"
      ROOT ${${name}_HEADER_ROOT}
      TARGETS ${${name}_component_HEADERS})

    set(${name}_HEADERS ${${name}_component_HEADERS})
  endif()

  hpx_print_list("DEBUG" "add_component.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_component.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_component.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_component.${name}" "Component dependencies for ${name}" ${name}_COMPONENT_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_component.${name}" "Configuration files for ${name}" ${name}_INI)

  if(${${name}_ESSENTIAL})
    add_library(${name}_component SHARED
      ${${name}_SOURCES} ${${name}_HEADERS})
  else()
    add_library(${name}_component SHARED EXCLUDE_FROM_ALL
      ${${name}_SOURCES} ${${name}_HEADERS})
  endif()

  set(prefix "")
  set(libs "")

  if(NOT ${${name}_NOLIBS})
    set(libs ${hpx_LIBRARIES})
    set_property(TARGET ${name}_component APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "BOOST_ENABLE_ASSERT_HANDLER")
  endif()

  hpx_handle_component_dependencies(${name}_COMPONENT_DEPENDENCIES)

  if(NOT MSVC)
    target_link_libraries(${name}_component
      ${${name}_DEPENDENCIES} ${${name}_COMPONENT_DEPENDENCIES} hpx)
    set(prefix "hpx_component_")
  else()
    target_link_libraries(${name}_component
      ${${name}_DEPENDENCIES} ${${name}_COMPONENT_DEPENDENCIES} hpx)
  endif()

  # set properties of generated shared library
  set_target_properties(${name}_component PROPERTIES
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

  if(HPX_FLAGS)
    set_property(TARGET ${name}_component APPEND PROPERTY COMPILE_FLAGS ${HPX_FLAGS})
    set_property(TARGET ${name}_component APPEND PROPERTY LINK_FLAGS ${HPX_FLAGS})
  endif()

  set_target_properties(${name}_component 
                        PROPERTIES SKIP_BUILD_RPATH TRUE
                                   BUILD_WITH_INSTALL_RPATH TRUE
                                   INSTALL_RPATH_USE_LINK_PATH TRUE 
                                   INSTALL_RPATH ${HPX_RPATH})

  if(${name}_FOLDER)
    set_target_properties(${name}_component PROPERTIES FOLDER ${${name}_FOLDER})
  endif()

  set_property(TARGET ${name}_component APPEND
               PROPERTY COMPILE_DEFINITIONS
               "HPX_COMPONENT_NAME=${name}"
               "HPX_COMPONENT_STRING=\"${name}\""
               "HPX_COMPONENT_EXPORTS")

  if(NOT HPX_NO_INSTALL)
    hpx_library_install(${name}_component)

    foreach(target ${${name}_INI})
      hpx_ini_install(${install_target} ${target})
    endforeach()
  endif()
endmacro()

