# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDEXECUTABLE_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments
            Install)

macro(add_hpx_executable name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "SOURCES;HEADERS;DEPENDENCIES;FOLDER;HEADER_ROOT;SOURCE_ROOT" "ESSENTIAL;NOLIBS" ${ARGN})

  hpx_print_list("DEBUG" "add_executable.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_executable.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_executable.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)

  # add the executable build target
  if(NOT MSVC)
    if(${${name}_ESSENTIAL})
      add_executable(${name}_exe
        ${${name}_SOURCES} ${${name}_HEADERS})
    else()
      add_executable(${name}_exe EXCLUDE_FROM_ALL
        ${${name}_SOURCES} ${${name}_HEADERS})
    endif()
  else()
    if(${${name}_ESSENTIAL})
      add_executable(${name}_exe
        ${${name}_SOURCES} ${${name}_HEADERS})
    else()
      add_executable(${name}_exe EXCLUDE_FROM_ALL
        ${${name}_SOURCES} ${${name}_HEADERS})
    endif()
  endif()

  set_target_properties(${name}_exe PROPERTIES OUTPUT_NAME ${name})

  if(${name}_FOLDER)
    set_target_properties(${name}_exe PROPERTIES FOLDER ${${name}_FOLDER})
  endif()

  set_property(TARGET ${name}_exe APPEND
               PROPERTY COMPILE_DEFINITIONS
               "HPX_APPLICATION_NAME=${name}"
               "HPX_APPLICATION_STRING=\"${name}\""
               "HPX_APPLICATION_EXPORTS")

  set(libs "")

  if(NOT MSVC)
    set(libs ${BOOST_FOUND_LIBRARIES})
  endif()

  # linker instructions
  if(NOT ${${name}_NOLIBS})
    target_link_libraries(${name}_exe
      ${${name}_DEPENDENCIES}
      ${hpx_LIBRARIES}
      hpx_init
      ${libs})
    set_property(TARGET ${name}_exe APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "BOOST_ENABLE_ASSERT_HANDLER")
  else()
    target_link_libraries(${name}_exe
      ${${name}_DEPENDENCIES})
  endif()

  if(NOT HPX_NO_INSTALL)
    if(${name}_ESSENTIAL)
      hpx_executable_install(${name}_exe ESSENTIAL)
    else()
      hpx_executable_install(${name}_exe)
    endif()
  endif()
endmacro()

