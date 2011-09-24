# Copyright (c) 2007-2011 Hartmut Kaiser
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
    "MODULE;SOURCES;HEADERS;DEPENDENCIES;INI" "ESSENTIAL;NOLIBS" ${ARGN})

  hpx_print_list("DEBUG" "add_library.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_library.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_library.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_library.${name}" "Configuration files for ${name}" ${name}_INI)

  if(NOT MSVC)
    if(${name}_ESSENTIAL)
      add_library(${name}_lib SHARED 
        ${${name}_SOURCES} ${${name}_HEADERS})
    else()
      add_library(${name}_lib SHARED EXCLUDE_FROM_ALL
        ${${name}_SOURCES} ${${name}_HEADERS})
    endif() 
  else()
    if(${name}_ESSENTIAL)
      add_library(${name}_lib SHARED ${${name}_SOURCES}) 
    else()
      add_library(${name}_lib SHARED EXCLUDE_FROM_ALL ${${name}_SOURCES}) 
    endif() 
  endif()

  set(prefix "")

  if(NOT MSVC)
    if(NOT ${name}_NOLIBS)
      target_link_libraries(${name}_lib
        ${${name}_DEPENDENCIES} ${hpx_LIBRARIES} ${BOOST_FOUND_LIBRARIES})
      set_property(TARGET ${name}_lib APPEND
                   PROPERTY COMPILE_DEFINITIONS
                   "BOOST_ENABLE_ASSERT_HANDLER")
    else()
      target_link_libraries(${name}_lib ${${name}_DEPENDENCIES})
    endif()
    set(prefix "hpx_")
    # main_target is checked by the ini install code, to see if ini files for
    # this component need to be installed
    set(main_target lib${prefix}${name}.so)
    set(install_targets lib${prefix}${name}.so)
  else()
    if(NOT ${name}_NOLIBS)
      target_link_libraries(${name}_lib
        ${${name}_DEPENDENCIES} ${hpx_LIBRARIES} ${BOOST_FOUND_LIBRARIES})
    else()
      target_link_libraries(${name}_lib ${${name}_DEPENDENCIES})
    endif()
    set(main_target ${name}.dll)
    set(install_targets ${name}.dll)
  endif()

  # set properties of generated shared library
  set_target_properties(${name}_lib PROPERTIES
    # create *nix style library versions + symbolic links
    VERSION ${HPX_VERSION}      
    SOVERSION ${HPX_SOVERSION}
    # allow creating static and shared libs without conflicts
    CLEAN_DIRECT_OUTPUT 1 
    OUTPUT_NAME ${prefix}${name})
  
  set_property(TARGET ${name}_lib APPEND
               PROPERTY COMPILE_DEFINITIONS "HPX_EXPORTS")
  
  if(NOT ${name}_MODULE)
    set(${name}_MODULE "Unspecified")
    hpx_debug("add_library.${name}" "Module was not specified for component.")
  endif()

  foreach(target ${install_targets})
    hpx_library_install(${${name}_MODULE} ${target})
  endforeach()

  foreach(target ${${name}_INI})
    hpx_ini_install(${${name}_MODULE} ${main_target} ${target})
  endforeach()
endmacro()

