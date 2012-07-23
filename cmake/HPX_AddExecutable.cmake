# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDEXECUTABLE_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments
            HandleComponentDependencies
            Install)

macro(add_hpx_executable name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "SOURCES;HEADERS;DEPENDENCIES;COMPONENT_DEPENDENCIES;FOLDER;HEADER_ROOT;SOURCE_ROOT" "ESSENTIAL;NOLIBS" ${ARGN})

  hpx_print_list("DEBUG" "add_executable.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_executable.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_executable.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_executable.${name}" "Component dependencies for ${name}" ${name}_COMPONENT_DEPENDENCIES)

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

  if(HPX_COMPILE_FLAGS)
    set_property(TARGET ${name}_exe APPEND PROPERTY COMPILE_FLAGS ${HPX_COMPILE_FLAGS})
    if(NOT MSVC)
      set_property(TARGET ${name}_exe APPEND PROPERTY LINK_FLAGS ${HPX_COMPILE_FLAGS})
    endif()
  endif()

  if(NOT MSVC)
    set_target_properties(${name}_exe
                          PROPERTIES SKIP_BUILD_RPATH TRUE
                                     BUILD_WITH_INSTALL_RPATH TRUE
                                     INSTALL_RPATH_USE_LINK_PATH TRUE
                                     INSTALL_RPATH ${HPX_RPATH})
  endif()

#  set(libs "")

#  if(NOT MSVC)
#    set(libs ${BOOST_FOUND_LIBRARIES})
#  endif()

  # linker instructions
  if(NOT ${${name}_NOLIBS})
    if(HPX_FOUND AND "${HPX_BUILD_TYPE}" STREQUAL "Debug")
      set(hpx_libs
        hpx${HPX_DEBUG_POSTFIX}
        hpx_init${HPX_DEBUG_POSTFIX}
        hpx_serialization${HPX_DEBUG_POSTFIX})
    else()
      set(hpx_libs
        hpx
        hpx_init
        hpx_serialization)
    endif()

    hpx_handle_component_dependencies(${name}_COMPONENT_DEPENDENCIES)
    target_link_libraries(${name}_exe
      ${${name}_DEPENDENCIES}
      ${${name}_COMPONENT_DEPENDENCIES}
      ${hpx_libs})
    set_property(TARGET ${name}_exe APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "BOOST_ENABLE_ASSERT_HANDLER")
  else()
    target_link_libraries(${name}_exe ${${name}_DEPENDENCIES})
  endif()

  if(NOT HPX_NO_INSTALL)
    hpx_executable_install(${name}_exe)
  endif()
endmacro()

