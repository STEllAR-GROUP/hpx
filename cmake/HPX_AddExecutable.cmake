# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_ADDEXECUTABLE_LOADED TRUE)

include(HPX_Include)

hpx_include(Message
            ParseArguments
            AppendProperty 
            HandleComponentDependencies
            Install)

macro(add_hpx_executable name)
  # retrieve arguments
  hpx_parse_arguments(${name}
    "SOURCES;HEADERS;DEPENDENCIES;COMPONENT_DEPENDENCIES;COMPILE_FLAGS;LINK_FLAGS;FOLDER;SOURCE_ROOT;HEADER_ROOT;SOURCE_GLOB;HEADER_GLOB;OUTPUT_SUFFIX;INSTALL_SUFFIX;LANGUAGE"
    "ESSENTIAL;NOLIBS;NOHPXINIT;AUTOGLOB" ${ARGN})

  if(NOT ${name}_LANGUAGE)
    set(${name}_LANGUAGE CXX)
  endif()

  if(NOT ${name}_SOURCE_ROOT)
    set(${name}_SOURCE_ROOT ".")
  endif()
  hpx_debug("add_executable.${name}" "${name}_SOURCE_ROOT: ${${name}_SOURCE_ROOT}")

  if(NOT ${name}_HEADER_ROOT)
    set(${name}_HEADER_ROOT ".")
  endif()
  hpx_debug("add_executable.${name}" "${name}_HEADER_ROOT: ${${name}_HEADER_ROOT}")

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
    hpx_debug("add_executable.${name}" "${name}_SOURCE_GLOB: ${${name}_SOURCE_GLOB}")

    add_hpx_library_sources(${name}_executable
      GLOB_RECURSE GLOBS "${${name}_SOURCE_GLOB}")

    set(${name}_SOURCES ${${name}_executable_SOURCES})
    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_executable_SOURCES})

    if(NOT ${name}_HEADER_GLOB)
      set(${name}_HEADER_GLOB "${${name}_HEADER_ROOT}/*.hpp"
                              "${${name}_HEADER_ROOT}/*.h")
    endif()
    hpx_debug("add_executable.${name}" "${name}_HEADER_GLOB: ${${name}_HEADER_GLOB}")

    add_hpx_library_headers(${name}_executable
      GLOB_RECURSE GLOBS "${${name}_HEADER_GLOB}")

    set(${name}_HEADERS ${${name}_executable_HEADERS})
    add_hpx_source_group(
      NAME ${name}
      CLASS "Header Files"
      ROOT ${${name}_HEADER_ROOT}
      TARGETS ${${name}_executable_HEADERS})
  else()
    add_hpx_library_sources_noglob(${name}_executable
        SOURCES "${${name}_SOURCES}")

    add_hpx_source_group(
      NAME ${name}
      CLASS "Source Files"
      ROOT ${${name}_SOURCE_ROOT}
      TARGETS ${${name}_executable_SOURCES})

    add_hpx_library_headers_noglob(${name}_executable
        HEADERS "${${name}_HEADERS}")

    add_hpx_source_group(
      NAME ${name}
      CLASS "Header Files"
      ROOT ${${name}_HEADER_ROOT}
      TARGETS ${${name}_executable_HEADERS})
  endif()

  set(${name}_SOURCES ${${name}_executable_SOURCES})
  set(${name}_HEADERS ${${name}_executable_HEADERS})

  hpx_print_list("DEBUG" "add_executable.${name}" "Sources for ${name}" ${name}_SOURCES)
  hpx_print_list("DEBUG" "add_executable.${name}" "Headers for ${name}" ${name}_HEADERS)
  hpx_print_list("DEBUG" "add_executable.${name}" "Dependencies for ${name}" ${name}_DEPENDENCIES)
  hpx_print_list("DEBUG" "add_executable.${name}" "Component dependencies for ${name}" ${name}_COMPONENT_DEPENDENCIES)

  # add the executable build target
  if(NOT HPX_EXTERNAL_CMAKE)
    set(exclude_from_all EXCLUDE_FROM_ALL)
  endif()

  if(${${name}_ESSENTIAL})
    add_executable(${name}_exe
      ${${name}_SOURCES} ${${name}_HEADERS})
  else()
    add_executable(${name}_exe ${exclude_from_all}
      ${${name}_SOURCES} ${${name}_HEADERS})
  endif()

  if(HPX_SET_OUTPUT_PATH AND NOT ${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(${name}_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${HPX_RUNTIME_OUTPUT_DIRECTORY_RELEASE}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${HPX_RUNTIME_OUTPUT_DIRECTORY_DEBUG}
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${HPX_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL}
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${HPX_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO})
    else()
      set_target_properties(${name}_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${HPX_RUNTIME_OUTPUT_DIRECTORY})
    endif()
  elseif(${name}_OUTPUT_SUFFIX)
    if(MSVC)
      set_target_properties(${name}_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release/${${name}_OUTPUT_SUFFIX}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug/${${name}_OUTPUT_SUFFIX}
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/MinSizeRel/${${name}_OUTPUT_SUFFIX}
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo/${${name}_OUTPUT_SUFFIX})
    else()
      set_target_properties(${name}_exe PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${${name}_OUTPUT_SUFFIX})
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

  if(${name}_COMPILE_FLAGS)
    hpx_append_property(${name}_exe COMPILE_FLAGS ${${name}_COMPILE_FLAGS})
  endif()

  if(${name}_LINK_FLAGS)
    hpx_append_property(${name}_exe LINK_FLAGS ${${name}_LINK_FLAGS})
  endif()

  if(HPX_HAVE_PARCELPORT_MPI AND MPI_FOUND)
    hpx_append_property(${name}_exe LINK_FLAGS ${MPI_${${name}_LANGUAGE}_LINK_FLAGS})
  endif()

  if(HPX_${${name}_LANGUAGE}_COMPILE_FLAGS)
    hpx_append_property(${name}_exe COMPILE_FLAGS ${HPX_${${name}_LANGUAGE}_COMPILE_FLAGS})
    if(NOT MSVC)
      hpx_append_property(${name}_exe LINK_FLAGS ${HPX_${${name}_LANGUAGE}_COMPILE_FLAGS})
    endif()
  endif()

  if(NOT MSVC)
    set_target_properties(${name}_exe
                          PROPERTIES SKIP_BUILD_RPATH TRUE
                                     BUILD_WITH_INSTALL_RPATH TRUE
                                     INSTALL_RPATH_USE_LINK_PATH TRUE
                                     INSTALL_RPATH ${HPX_RPATH})
    if(HPX_PIE)
      hpx_append_property(${name}_exe LINK_FLAGS -pie)
    endif()
  endif()

  # linker instructions
  if(NOT ${${name}_NOLIBS})
    if(HPX_EXTERNAL_CMAKE AND "${HPX_BUILD_TYPE}" STREQUAL "Debug")
      set(hpx_libs
        hpx${HPX_DEBUG_POSTFIX}
        hpx_serialization${HPX_DEBUG_POSTFIX})
      if(NOT ${${name}_NOHPXINIT})
        set(hpx_libs ${hpx_libs} hpx_init${HPX_DEBUG_POSTFIX})
      endif()
    else()
      set(hpx_libs
        hpx
        hpx_serialization)
      if(NOT ${${name}_NOHPXINIT})
        set(hpx_libs ${hpx_libs} hpx_init)
      endif()
    endif()

    if(HPX_EXTERNAL_CMAKE)
      set(hpx_libs ${hpx_libs} ${HPX_LIBRARIES})
    else()
      set(hpx_libs ${hpx_libs} ${hpx_LIBRARIES})
    endif()

    hpx_handle_component_dependencies(${name}_COMPONENT_DEPENDENCIES)

    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      target_link_libraries(${name}_exe
        ${${name}_DEPENDENCIES}
        ${${name}_COMPONENT_DEPENDENCIES}
        ${hpx_libs}
        imf svml irng intlc)
    else()
      target_link_libraries(${name}_exe
        ${${name}_DEPENDENCIES}
        ${${name}_COMPONENT_DEPENDENCIES}
        ${hpx_libs})
    endif()
    set_property(TARGET ${name}_exe APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "BOOST_ENABLE_ASSERT_HANDLER")
  else()
    target_link_libraries(${name}_exe ${${name}_DEPENDENCIES})
  endif()

  if(MSVC AND HPX_LINK_FLAG_TARGET_PROPERTIES)
    set_target_properties(${name}_exe PROPERTIES LINK_FLAGS "${HPX_LINK_FLAG_TARGET_PROPERTIES}")
  endif()

  if(NOT HPX_NO_INSTALL)
    if(${name}_INSTALL_SUFFIX)
      hpx_executable_install(${name}_exe ${${name}_INSTALL_SUFFIX})
    else()
      hpx_executable_install(${name}_exe bin)
    endif()
  endif()
endmacro()

