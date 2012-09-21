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
    "SOURCES;HEADERS;DEPENDENCIES;COMPONENT_DEPENDENCIES;COMPILE_FLAGS;LINK_FLAGS;INI;FOLDER;SOURCE_ROOT;HEADER_ROOT;SOURCE_GLOB;HEADER_GLOB;OUTPUT_SUFFIX;LANGUAGE"
    "ESSENTIAL;NOLIBS;AUTOGLOB;STATIC" ${ARGN})

  if(NOT ${name}_LANGUAGE)
    set(${name}_LANGUAGE CXX)
  endif() 

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  hpx_debug("add_component.${name}" "${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

  if(NOT ${name}_HEADER_ROOT)
    set(${name}_HEADER_ROOT ".")
  endif()
  hpx_debug("add_component.${name}" "${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

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
    hpx_debug("add_component.${name}" "${name}_SOURCE_GLOB: ${${name}_SOURCE_GLOB}")

    add_hpx_library_sources(${name}_component
      GLOB_RECURSE GLOBS "${${name}_SOURCE_GLOB}")

    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_component_SOURCES})

    if(NOT ${name}_HEADER_GLOB)
      set(${name}_HEADER_GLOB "${${name}_HEADER_ROOT}/*.hpp"
                              "${${name}_HEADER_ROOT}/*.h")
    endif()
    hpx_debug("add_component.${name}" "${name}_HEADER_GLOB: ${${name}_HEADER_GLOB}")

    if(NOT ${name}_HEADER_GLOB)
      add_hpx_library_headers(${name}_component
        GLOB_RECURSE GLOBS "${${name}_HEADER_GLOB}")

      add_hpx_source_group(
        NAME ${name}
        CLASS "Header Files"
        ROOT ${${name}_HEADER_ROOT}
        TARGETS ${${name}_component_HEADERS})
    endif()
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

  set(libs "")

  if(NOT ${${name}_NOLIBS})
    set(libs ${hpx_LIBRARIES})
    set_property(TARGET ${name}_component APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "BOOST_ENABLE_ASSERT_HANDLER")
  endif()

  hpx_handle_component_dependencies(${name}_COMPONENT_DEPENDENCIES)

  if(HPX_EXTERNAL_CMAKE AND "${HPX_BUILD_TYPE}" STREQUAL "Debug")
    set(hpx_libs hpx${HPX_DEBUG_POSTFIX} hpx_serialization${HPX_DEBUG_POSTFIX})
  else()
    set(hpx_libs hpx hpx_serialization)
  endif()

  target_link_libraries(${name}_component
    ${${name}_DEPENDENCIES} ${${name}_COMPONENT_DEPENDENCIES} ${hpx_libs})

  if(HPX_EXTERNAL_CMAKE AND "${HPX_BUILD_TYPE}" STREQUAL "Debug")
    set(lib_name ${name}${HPX_DEBUG_POSTFIX})
  else()
    set(lib_name ${name})
  endif()
  # set properties of generated shared library
  set_target_properties(${name}_component PROPERTIES
    # create *nix style library versions + symbolic links
    VERSION ${HPX_VERSION}
    SOVERSION ${HPX_SOVERSION}
    # allow creating static and shared libs without conflicts
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME ${lib_name})

  if(HPX_INTERNAL_CMAKE AND NOT ${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(${name}_component PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELEASE}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${HPX_LIBRARY_OUTPUT_DIRECTORY_DEBUG}
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO}
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${HPX_ARCHIVE_OUTPUT_DIRECTORY_RELEASE}
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${HPX_ARCHIVE_OUTPUT_DIRECTORY_DEBUG}
        ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL}
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELEASE}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${HPX_LIBRARY_OUTPUT_DIRECTORY_DEBUG}
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL}
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO})
    else()
      set_target_properties(${name}_component PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${HPX_LIBRARY_OUTPUT_DIRECTORY}
        ARCHIVE_OUTPUT_DIRECTORY ${HPX_LIBRARY_OUTPUT_DIRECTORY}
        LIBRARY_OUTPUT_DIRECTORY ${HPX_LIBRARY_OUTPUT_DIRECTORY})
    endif()
  elseif(${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(${name}_component PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/${${name}_OUTPUT_SUFFIX}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/${${name}_OUTPUT_SUFFIX}
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/MinSizeRel/${${name}_OUTPUT_SUFFIX}
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo/${${name}_OUTPUT_SUFFIX}
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/${${name}_OUTPUT_SUFFIX}
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/${${name}_OUTPUT_SUFFIX}
        ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/MinSizeRel/${${name}_OUTPUT_SUFFIX}
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo/${${name}_OUTPUT_SUFFIX}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/${${name}_OUTPUT_SUFFIX}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/${${name}_OUTPUT_SUFFIX}
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/MinSizeRel/${${name}_OUTPUT_SUFFIX}
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo/${${name}_OUTPUT_SUFFIX})
    else()
      set_target_properties(${name}_component PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${${name}_OUTPUT_SUFFIX}
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${${name}_OUTPUT_SUFFIX}
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${${name}_OUTPUT_SUFFIX})
    endif()
  endif()

  if(${name}_COMPILE_FLAGS)
    set_property(TARGET ${name}_component APPEND
      PROPERTY COMPILE_FLAGS ${${name}_COMPILE_FLAGS})
  endif()

  if(${name}_LINK_FLAGS)
    set_property(TARGET ${name}_component APPEND
      PROPERTY LINK_FLAGS ${${name}_LINK_FLAGS})
  endif()

  if(HPX_${${name}_LANGUAGE}_COMPILE_FLAGS)
    set_property(TARGET ${name}_component APPEND
      PROPERTY COMPILE_FLAGS ${HPX_${${name}_LANGUAGE}_COMPILE_FLAGS})
    if(NOT MSVC)
      set_property(TARGET ${name}_component APPEND
        PROPERTY LINK_FLAGS ${HPX_${${name}_LANGUAGE}_COMPILE_FLAGS})
    endif()
  endif()

  if(${name}_COMPILE_FLAGS)
    set_property(TARGET ${name}_component APPEND
      PROPERTY COMPILE_FLAGS ${${name}_COMPILE_FLAGS})
  endif()

  if(${name}_LINK_FLAGS)
    set_property(TARGET ${name}_component APPEND
      PROPERTY LINK_FLAGS ${${name}_LINK_FLAGS})
  endif()

  if(HPX_COMPILE_FLAGS)
    set_property(TARGET ${name}_component APPEND
      PROPERTY COMPILE_FLAGS ${HPX_COMPILE_FLAGS})
    if(NOT MSVC)
      set_property(TARGET ${name}_component APPEND
        PROPERTY LINK_FLAGS ${HPX_COMPILE_FLAGS})
    endif()
  endif()

  if(NOT MSVC)
    set_target_properties(${name}_component
                          PROPERTIES SKIP_BUILD_RPATH TRUE
                                     BUILD_WITH_INSTALL_RPATH TRUE
                                     INSTALL_RPATH_USE_LINK_PATH TRUE
                                     INSTALL_RPATH ${HPX_RPATH})
  endif()

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
      hpx_debug("add_component.${name}" "installing ini: ${name}")
      hpx_ini_install(${target})
    endforeach()
  endif()
endmacro()

