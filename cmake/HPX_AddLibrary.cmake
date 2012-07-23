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
    "SOURCES;HEADERS;DEPENDENCIES;COMPONENT_DEPENDENCIES;FOLDER" "ESSENTIAL;NOLIBS" ${ARGN})

  hpx_print_list("DEBUG" "add_library.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_library.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_library.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_library.${name}" "Component dependencies for ${name}" ${name}_COMPONENT_DEPENDENCIES)

  if(NOT MSVC)
    if(${${name}_ESSENTIAL})
      add_library(${name}_lib SHARED
        ${${name}_SOURCES} ${${name}_HEADERS})
    else()
      add_library(${name}_lib SHARED EXCLUDE_FROM_ALL
        ${${name}_SOURCES} ${${name}_HEADERS})
    endif()
  else()
    if(${${name}_ESSENTIAL})
      add_library(${name}_lib SHARED ${${name}_SOURCES})
    else()
      add_library(${name}_lib SHARED EXCLUDE_FROM_ALL ${${name}_SOURCES})
    endif()
  endif()

  hpx_handle_component_dependencies(${name}_COMPONENT_DEPENDENCIES)

  if(HPX_FOUND AND "${HPX_BUILD_TYPE}" STREQUAL "Debug")
    set(hpx_lib hpx${HPX_DEBUG_POSTFIX})
  else()
    set(hpx_lib hpx)
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

