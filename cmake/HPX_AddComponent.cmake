# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDCOMPONENT_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments
            Install)

macro(add_hpx_component name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "SOURCES;HEADERS;DEPENDENCIES;INI" "ESSENTIAL;NOLIBS" ${ARGN})

  hpx_print_list("DEBUG" "add_component.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_component.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_component.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_component.${name}" "Configuration files for ${name}" ${name}_INI)

  if(NOT MSVC)
    if(${name}_ESSENTIAL)
      add_library(${name}_component SHARED
        ${${name}_SOURCES} ${${name}_HEADERS})
    else()
      add_library(${name}_component SHARED EXCLUDE_FROM_ALL
        ${${name}_SOURCES} ${${name}_HEADERS})
    endif()
  else()
    if(${name}_ESSENTIAL)
      add_library(${name}_component SHARED ${${name}_SOURCES})
    else()
      add_library(${name}_component SHARED EXCLUDE_FROM_ALL ${${name}_SOURCES})
    endif()
  endif()

  set(prefix "")
  set(libs "")

  if(NOT ${name}_NOLIBS)
    set(libs ${hpx_LIBRARIES})
    set_property(TARGET ${name}_component APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "BOOST_ENABLE_ASSERT_HANDLER")
  endif()

  if(NOT MSVC)
    target_link_libraries(${name}_component
      ${${name}_DEPENDENCIES} ${libs} ${BOOST_FOUND_LIBRARIES})
    set(prefix "hpx_component_")
  else()
    target_link_libraries(${name}_component
      ${${name}_DEPENDENCIES} ${libs})
  endif()

  # set properties of generated shared library
  set_target_properties(${name}_component PROPERTIES
    # create *nix style library versions + symbolic links
    VERSION ${HPX_VERSION}
    SOVERSION ${HPX_SOVERSION}
    # allow creating static and shared libs without conflicts
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME ${prefix}${name})

  set_property(TARGET ${name}_component APPEND
               PROPERTY COMPILE_DEFINITIONS
               "HPX_COMPONENT_NAME=${name}"
               "HPX_COMPONENT_STRING=\"${name}\""
               "HPX_COMPONENT_EXPORTS")

  hpx_mangle_name(install_target ${name}_component)

  hpx_library_install(${install_target})

  foreach(target ${${name}_INI})
    hpx_ini_install(${install_target} ${target})
  endforeach()
endmacro()

