# Copyright (c) 2014      Thomas Heller
# Copyright (c) 2007-2012 Hartmut Kaiser
# Copyright (c) 2011      Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

function(hpx_setup_target target)
  # retrieve arguments
  set(options EXPORT NOHPX_INIT INSTALL NOLIBS PLUGIN)
  set(one_value_args TYPE FOLDER NAME SOVERSION VERSION HPX_PREFIX)
  set(multi_value_args DEPENDENCIES COMPONENT_DEPENDENCIES COMPILE_FLAGS LINK_FLAGS INSTALL_FLAGS)
  cmake_parse_arguments(target "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  hpx_is_target(is_target ${target})
  if(NOT is_target)
    hpx_error("${target} does not represent a target")
  endif()

  # Figure out which type we want...
  if(target_TYPE)
    string(TOUPPER "${target_TYPE}" _type)
  else()
    get_target_property(type_prop ${target} TYPE)
    if(type_prop STREQUAL "STATIC_LIBRARY")
      set(_type "LIBRARY")
    endif()
    if(type_prop STREQUAL "MODULE_LIBRARY")
      set(_type "LIBRARY")
    endif()
    if(type_prop STREQUAL "SHARED_LIBRARY")
      set(_type "LIBRARY")
    endif()
    if(type_prop STREQUAL "EXECUTABLE")
      set(_type "EXECUTABLE")
    endif()
  endif()

  if(target_FOLDER)
    set_target_properties(${target} PROPERTIES FOLDER "${target_FOLDER}")
  endif()

  if(target_COMPILE_FLAGS)
    hpx_append_property(${target} COMPILE_FLAGS ${target_COMPILE_FLAGS})
  endif()

  if(target_LINK_FLAGS)
    hpx_append_property(${target} LINK_FLAGS ${target_LINK_FLAGS})
  endif()

  if(target_NAME)
    set(name "${target_NAME}")
  else()
    set(name "${target}")
  endif()

  set(nohpxinit FALSE)
  if(target_NOHPXINIT)
    set(nohpxinit TRUE)
  endif()

  set(target_STATIC_LINKING OFF)
  if(HPX_STATIC_LINKING)
    set(target_STATIC_LINKING ON)
  else()
    if(POLICY CMP0045)
      cmake_policy(PUSH)
      cmake_policy(SET CMP0045 OLD)
    endif()

    get_target_property(_hpx_library_type hpx TYPE)

    if(POLICY CMP0045)
      cmake_policy(POP)
    endif()

    if("${_hpx_library_type}" STREQUAL "STATIC_LIBRARY")
      set(target_STATIC_LINKING ON)
    endif()
  endif()

  if(HPX_INCLUDE_DIRS)
    set_property(TARGET ${target} APPEND
      PROPERTY INCLUDE_DIRECTORIES
      "${HPX_INCLUDE_DIRS}"
    )
  endif()

  if("${_type}" STREQUAL "EXECUTABLE")
    if(target_HPX_PREFIX)
      set(_prefix ${target_HPX_PREFIX})
    else()
      set(_prefix ${HPX_PREFIX})
    endif()

    if(MSVC)
      string(REPLACE ";" ":" _prefix "${_prefix}")
    endif()
    set_property(TARGET ${target} APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "HPX_APPLICATION_NAME=${name}"
                 "HPX_APPLICATION_STRING=\"${name}\""
                 "HPX_PREFIX=\"${_prefix}\""
                 "HPX_APPLICATION_EXPORTS")
  endif()
  if("${_type}" STREQUAL "LIBRARY")
    set(nohpxinit FALSE)
    if(DEFINED HPX_LIBRARY_VERSION AND DEFINED HPX_SOVERSION)
      # set properties of generated shared library
      set_target_properties(${target}
        PROPERTIES
        VERSION ${HPX_LIBRARY_VERSION}
        SOVERSION ${HPX_SOVERSION})
    endif()
    set_target_properties(${target}
      PROPERTIES
      # create *nix style library versions + symbolic links
      # allow creating static and shared libs without conflicts
      CLEAN_DIRECT_OUTPUT 1
      OUTPUT_NAME ${name})
    if(target_PLUGIN)
      set(plugin_name "HPX_PLUGIN_NAME=${name}")
    endif()

    set_property(TARGET ${target} APPEND
                 PROPERTY COMPILE_DEFINITIONS
                   "HPX_LIBRARY_EXPORTS"
                   ${plugin_name})

  endif()

  if("${_type}" STREQUAL "COMPONENT")
    set(nohpxinit FALSE)
    if(DEFINED HPX_LIBRARY_VERSION AND DEFINED HPX_SOVERSION)
    # set properties of generated shared library
      set_target_properties(${target}
        PROPERTIES
        VERSION ${HPX_LIBRARY_VERSION}
        SOVERSION ${HPX_SOVERSION})
    endif()
    set_target_properties(${target}
      PROPERTIES
      # create *nix style library versions + symbolic links
      # allow creating static and shared libs without conflicts
      CLEAN_DIRECT_OUTPUT 1
      OUTPUT_NAME ${name})

    set_property(TARGET ${target} APPEND
                 PROPERTY COMPILE_DEFINITIONS
                 "HPX_COMPONENT_NAME=${name}"
                 "HPX_COMPONENT_STRING=\"${name}\""
                 "HPX_COMPONENT_EXPORTS")
  endif()

  # We force the -DDEBUG and -D_DEBUG defines in debug mode to avoid
  # ABI differences
  if(CMAKE_MAJOR_VERSION GREATER 2)
    set_property(TARGET ${target} APPEND PROPERTY
	    COMPILE_DEFINITIONS $<$<CONFIG:Debug>:HPX_DEBUG>)
  else()
    set_property(TARGET ${target} APPEND PROPERTY
      COMPILE_DEFINITIONS_DEBUG HPX_DEBUG)
  endif()

  # linker instructions
  if(NOT target_NOLIBS)
    set(hpx_libs hpx)
    if(NOT target_STATIC_LINKING)
      set(hpx_libs ${hpx_libs} hpx_serialization)
      if(NOT nohpxinit)
        set(hpx_libs ${hpx_libs} hpx_init)
      endif()
    endif()
    hpx_handle_component_dependencies(target_COMPONENT_DEPENDENCIES)
    set(hpx_libs ${hpx_libs} ${target_COMPONENT_DEPENDENCIES})
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
      set(hpx_libs ${hpx_libs} imf svml irng intlc)
    endif()
    if(DEFINED HPX_LIBRARIES)
      set(hpx_libs ${hpx_libs} ${HPX_LIBRARIES})
    endif()
  endif()

  target_link_libraries(${target} ${hpx_libs} ${target_DEPENDENCIES})

  get_target_property(target_EXCLUDE_FROM_ALL ${target} EXCLUDE_FROM_ALL)

  if(target_EXPORT AND NOT target_EXCLUDE_FROM_ALL)
    hpx_export_targets(${target})
    set(install_export EXPORT HPXTargets)
  endif()

  if(target_INSTALL AND NOT target_EXCLUDE_FROM_ALL)
    install(TARGETS ${target}
      ${target_INSTALL_FLAGS}
      ${install_export}
    )
  endif()
endfunction()
