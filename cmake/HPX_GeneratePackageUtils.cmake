# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# Replace the "-NOTFOUND" by empty var in case the property is not found
macro(get_target_property var target property)
  _get_target_property(${var} ${target} ${property})
  list(FILTER ${var} EXCLUDE REGEX "-NOTFOUND$")
endmacro(get_target_property)

# https://github.com/boost-cmake/bcm/blob/master/share/bcm/cmake/BCMPkgConfig.cmake
# https://gitlab.kitware.com/cmake/cmake/issues/17984

# Recursively add the interface_include_dirs of the dependencies and link them
function(
  hpx_collect_usage_requirements
  target
  compile_definitions
  compile_options
  pic_option
  include_directories
  system_include_directories
  link_libraries
  link_options
  already_processed_targets
  is_component
)
  cmake_parse_arguments(collect "" "" "EXCLUDE" ${ARGN})

  if(${target} IN_LIST collect_EXCLUDE)
    return()
  endif()

  # Check if the target has already been processed
  list(FIND ${already_processed_targets} ${target} _found)
  if(NOT (${_found} EQUAL -1))
    return()
  endif()

  set(_already_processed_targets ${${already_processed_targets}} ${target})

  get_target_property(
    _target_compile_definitions ${target} INTERFACE_COMPILE_DEFINITIONS
  )
  get_target_property(
    _target_compile_options ${target} INTERFACE_COMPILE_OPTIONS
  )
  get_target_property(
    _target_pic_option ${target} INTERFACE_POSITION_INDEPENDENT_CODE
  )
  get_target_property(
    _target_include_directories ${target} INTERFACE_INCLUDE_DIRECTORIES
  )
  get_target_property(
    _target_system_include_directories ${target}
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
  )
  get_target_property(_target_link_libraries ${target} INTERFACE_LINK_LIBRARIES)
  get_target_property(_target_link_options ${target} INTERFACE_LINK_OPTIONS)
  get_target_property(_target_type ${target} TYPE)

  if(NOT "${_target_type}" STREQUAL "INTERFACE_LIBRARY")
    get_target_property(_target_imported_location ${target} IMPORTED_LOCATION)
    list(APPEND _target_link_libraries ${_target_imported_location})
  endif()

  # If the target is a library link against it.
  if("${_target_type}" STREQUAL "STATIC_LIBRARY" OR "${_target_type}" STREQUAL
                                                    "SHARED_LIBRARY"
  )
    # Is there a better way to handle this? When HPX::hpx_local is an imported
    # target the generator expression $<TARGET_FILE_BASE_NAME:HPX::hpx_local>
    # evaluates to HPX::hpx_local, which is wrong.
    if("${target}" STREQUAL "HPX::hpx_local")
      if(HPX_WITH_FETCH_HPXLOCAL)
        set(_libraries
            $<INSTALL_INTERFACE:-L${CMAKE_INSTALL_LIBDIR};-l$<TARGET_FILE_BASE_NAME:${target}>>$<BUILD_INTERFACE:$<TARGET_FILE:${target}>>
        )
      else()
        string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
        if("${build_type}" STREQUAL "debug")
          set(debug_postfix ${HPXLocal_DEBUG_POSTFIX})
        endif()
        set(_libraries
            -L$<TARGET_FILE_DIR:HPX::hpx_local>;-lhpx_local${debug_postfix}
        )
      endif()
    else()
      if(${is_component})
        # We put the link directory and let the user specify -l<component>
        set(_libraries
            -L$<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}>$<BUILD_INTERFACE:$<TARGET_FILE_DIR:${target}>>
        )
      else()
        set(_libraries
            $<INSTALL_INTERFACE:-L${CMAKE_INSTALL_LIBDIR};-l$<TARGET_FILE_BASE_NAME:${target}>>$<BUILD_INTERFACE:$<TARGET_FILE:${target}>>
        )
      endif()
    endif()
  else()
    set(_libraries "")
  endif()

  # In case of components no need to do the recursive search
  if(NOT ${is_component})
    foreach(dep IN LISTS _target_link_libraries)

      if(${dep} MATCHES "^\\$<LINK_ONLY:([^>]+)>$")
        # This a private link dependency. Do not inherit the target's usage
        # requirements.
        set(dep_target ${CMAKE_MATCH_1})
        if(TARGET ${dep_target})
          get_target_property(dep_type ${dep_target} TYPE)
          if("${dep_type}" STREQUAL "STATIC_LIBRARY" OR "${dep_type}" STREQUAL
                                                        "SHARED_LIBRARY"
          )
            set(_libraries
                ${_libraries}
                $<INSTALL_INTERFACE:${CMAKE_INSTALL_LIBDIR}/$<TARGET_FILE_NAME:${target}>>$<BUILD_INTERFACE:$<TARGET_FILE:${target}>>
            )
          endif()
        elseif("${dep_target}" MATCHES "^-l")
          set(_libraries ${_libraries} ${dep_target})
        else()
          set(_libraries ${_libraries} -l${dep_target})
        endif()

      elseif(TARGET ${dep})
        # This is a public dependency. Follow the dependency graph and add the
        # target's usage requirements.
        if(NOT dep IN_LIST _already_processed_targets)
          # This is not put inside a function in order not to hide the
          # recursivity
          hpx_collect_usage_requirements(
            ${dep}
            dep_compile_definitions
            dep_compile_options
            dep_pic_option
            dep_include_directories
            dep_system_include_directories
            dep_link_libraries
            dep_link_options
            _already_processed_targets
            ${is_component}
            EXCLUDE ${collect_EXCLUDE}
          )
          list(APPEND _target_compile_definitions "${dep_compile_definitions}")
          list(APPEND _target_compile_options "${dep_compile_options}")
          list(APPEND _target_include_directories "${dep_include_directories}")
          list(APPEND _target_system_include_directories
               "${dep_system_include_directories}"
          )
          list(APPEND _libraries "${dep_link_libraries}")
          list(APPEND _target_link_options "${dep_link_options}")
          if(dep_pic_option)
            set(_target_pic_option ON)
          endif()
        endif()

      elseif(${dep} MATCHES "::@")
        # Skip targets beginning with ::@ as they come from object libraries
        # which do not need to be linked.
      elseif(${dep} MATCHES "\\$<TARGET_NAME_IF_EXISTS:")
        # Skip conditional targets like $<TARGET_NAME_IF_EXISTS:hpx> as they are
        # not useful for pkgconfig file generation.
      elseif(${dep} MATCHES "\\$<")
        # This is a plain generator expression. As we can not determine its type
        # at this point, just assume that it is an absolute path.
        set(_libraries ${_libraries} ${dep})

      elseif(${dep} MATCHES "^-l")
        # This is a library with a linker flag.
        set(_libraries ${_libraries} ${dep})

      else()
        # This is a plain path.
        if(IS_ABSOLUTE ${dep})
          set(_libraries ${_libraries} ${dep})
          # This is a link flag put as a link_libraries (to solve some cmake
          # problems)
        elseif(${dep} MATCHES "^-")
          set(_libraries ${_libraries} ${dep})
        else()
          set(_libraries ${_libraries} -l${dep})
        endif()

      endif()

    endforeach()
  endif()

  # Remove duplicates for include dir but not for link cause order is important
  list(REMOVE_DUPLICATES _target_include_directories)
  list(REMOVE_DUPLICATES _target_system_include_directories)

  set(${compile_definitions}
      ${_target_compile_definitions}
      PARENT_SCOPE
  )
  set(${compile_options}
      ${_target_compile_options}
      PARENT_SCOPE
  )
  set(${pic_option}
      "${_target_pic_option}"
      PARENT_SCOPE
  )
  set(${include_directories}
      ${_target_include_directories}
      PARENT_SCOPE
  )
  set(${system_include_directories}
      ${_target_system_include_directories}
      PARENT_SCOPE
  )
  set(${link_libraries}
      ${_libraries}
      PARENT_SCOPE
  )
  set(${link_options}
      ${_target_link_options}
      PARENT_SCOPE
  )
  set(${already_processed_targets}
      ${_already_processed_targets}
      PARENT_SCOPE
  )

endfunction(hpx_collect_usage_requirements)

# Isolate the build properties from the install ones
function(hpx_sanitize_usage_requirements property is_build)

  foreach(prop IN LISTS ${property})
    if(is_build)
      string(REPLACE "$<BUILD_INTERFACE:" "$<1:" prop "${prop}")
      string(REPLACE "$<INSTALL_INTERFACE:" "$<0:" prop "${prop}")
    else()
      string(REPLACE "$<BUILD_INTERFACE:" "$<0:" prop "${prop}")
      string(REPLACE "$<INSTALL_INTERFACE:-L$<TARGET_FILE_DIR:"
                     "$<1:-L$<TARGET_FILE_DIR:" prop "${prop}"
      )
      string(REPLACE "$<INSTALL_INTERFACE:-L/" "$<1:-L/" prop "${prop}")
      string(REPLACE "$<INSTALL_INTERFACE:-L" "$<1:-L${CMAKE_INSTALL_PREFIX}/"
                     prop "${prop}"
      )
      string(REPLACE "$<INSTALL_INTERFACE:" "$<1:${CMAKE_INSTALL_PREFIX}/" prop
                     "${prop}"
      )
    endif()
    set(_sanitized_property ${_sanitized_property} ${prop})
  endforeach()
  set(${property}
      ${_sanitized_property}
      PARENT_SCOPE
  )

endfunction(hpx_sanitize_usage_requirements)

function(hpx_filter_cuda_flags cflag_list)
  set(_cflag_list "${${cflag_list}}")
  string(REGEX REPLACE "\\$<\\$<COMPILE_LANGUAGE:CUDA>:[^>]*>;?" "" _cflag_list
                       "${_cflag_list}"
  )
  set(${cflag_list}
      ${_cflag_list}
      PARENT_SCOPE
  )
endfunction(hpx_filter_cuda_flags)

# Append the corresponding (-D, -I) flags for the compilation
function(
  hpx_construct_cflag_list
  compile_definitions
  compile_options
  pic_option
  include_directories
  sys_include_dirs
  cflag_list
)

  if(pic_option)
    set(_cflag_list "${_cflag_list} -fPIC")
  endif()
  set(_cflag_list
      "${_cflag_list} $<$<BOOL:${${compile_definitions}}>:-D$<JOIN:${${compile_definitions}}, -D>>"
  )
  set(_cflag_list "${_cflag_list} $<JOIN:${${compile_options}}, >")
  set(_cflag_list
      "${_cflag_list} $<$<BOOL:${${include_directories}}>:-I$<JOIN:${${include_directories}}, -I>>"
  )
  # NOTE: This uses -I and not -isystem in lack of a good way to filter out
  # compiler search paths.
  set(_cflag_list
      "${_cflag_list} $<$<BOOL:${${sys_include_dirs}}>:-I$<JOIN:${${sys_include_dirs}}, -I>>"
  )
  set(${cflag_list}
      ${_cflag_list}
      PARENT_SCOPE
  )

endfunction(hpx_construct_cflag_list)

function(hpx_construct_library_list link_libraries link_options library_list)
  foreach(library IN LISTS ${link_libraries})
    if(IS_ABSOLUTE ${library})
      get_filename_component(_library_path ${library} DIRECTORY)
      get_filename_component(_library_name ${library} NAME_WE)
      string(REPLACE ${CMAKE_SHARED_LIBRARY_PREFIX} "" _library_name
                     ${_library_name}
      )
      set(_library_list
          "${_library_list} -L${_library_path} -l${_library_name}"
      )
    else()
      set(_library_list "${_library_list} ${library}")
    endif()
  endforeach()
  foreach(option IN LISTS ${link_options})
    set(_library_list "${_library_list} ${option}")
  endforeach()
  set(${library_list}
      ${_library_list}
      PARENT_SCOPE
  )
endfunction(hpx_construct_library_list)

# Configure the corresponding package config template for the specified
# ${template}
function(hpx_generate_pkgconfig_from_target target template is_build)
  cmake_parse_arguments(pkgconfig "" "" "EXCLUDE" ${ARGN})

  if(${is_build})
    set(OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/)
    set(hpx_local_library_list ${HPXLocal_BUILD_PKGCONFIG_LIBRARY_LIST})
    set(hpx_local_cflags_list ${HPXLocal_BUILD_PKGCONFIG_CFLAGS_LIST})
  else()
    # Parent_scope to use the same variable to install
    set(OUTPUT_DIR ${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/)
    set(OUTPUT_DIR_PC
        ${OUTPUT_DIR}
        PARENT_SCOPE
    )
    set(hpx_local_library_list ${HPXLocal_INSTALL_PKGCONFIG_LIBRARY_LIST})
    set(hpx_local_cflags_list ${HPXLocal_INSTALL_PKGCONFIG_CFLAGS_LIST})
  endif()

  set(is_component FALSE)
  hpx_collect_usage_requirements(
    ${target}
    hpx_compile_definitions
    hpx_compile_options
    hpx_pic_option
    hpx_include_directories
    hpx_system_include_directories
    hpx_link_libraries
    hpx_link_options
    processed_targets
    ${is_component}
    EXCLUDE ${pkgconfig_EXCLUDE}
  )

  # Add all the components which aren't linked to hpx
  set(_component_list ${HPX_COMPONENTS})
  hpx_handle_component_dependencies(_component_list)
  set(is_component TRUE)
  foreach(component IN LISTS _component_list)
    hpx_collect_usage_requirements(
      ${component}
      dep_compile_definitions
      dep_compile_options
      dep_pic_option
      dep_include_directories
      dep_system_include_directories
      dep_link_libraries
      dep_link_options
      processed_targets
      ${is_component}
    )
    list(APPEND hpx_compile_definitions ${dep_compile_definitions})
    list(APPEND hpx_compile_options ${dep_compile_options})
    list(APPEND hpx_include_directories ${dep_include_directories})
    list(APPEND hpx_system_include_directories
         ${dep_system_include_directories}
    )
    list(APPEND hpx_link_libraries ${dep_link_libraries})
    list(APPEND hpx_link_options ${dep_link_options})
    if(${dep_pic_option})
      set(hpx_pic_option ON)
    endif()
  endforeach()

  # Filter between install and build interface
  hpx_sanitize_usage_requirements(hpx_compile_definitions ${is_build})
  hpx_sanitize_usage_requirements(hpx_compile_options ${is_build})
  hpx_sanitize_usage_requirements(hpx_include_directories ${is_build})
  hpx_sanitize_usage_requirements(hpx_system_include_directories ${is_build})
  hpx_sanitize_usage_requirements(hpx_link_libraries ${is_build})
  hpx_sanitize_usage_requirements(hpx_link_options ${is_build})

  hpx_construct_cflag_list(
    hpx_compile_definitions hpx_compile_options hpx_pic_option
    hpx_include_directories hpx_system_include_directories hpx_cflags_list
  )
  # Cannot generate one file per language yet so filter out cuda
  hpx_filter_cuda_flags(hpx_cflags_list)
  hpx_construct_library_list(
    hpx_link_libraries hpx_link_options hpx_library_list
  )

  string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)

  configure_file(
    cmake/templates/${template}.pc.in
    ${OUTPUT_DIR}${template}_${build_type}.pc.in @ONLY ESCAPE_QUOTES
  )
  # Can't use generator expression directly as name of output file (solved in
  # CMake 3.20)
  file(
    GENERATE
    OUTPUT ${OUTPUT_DIR}/${template}_${build_type}.pc
    INPUT ${OUTPUT_DIR}${template}_${build_type}.pc.in
  )
  # Temporary (to deprecate gradually)
  if("${build_type}" MATCHES "rel")
    file(
      GENERATE
      OUTPUT ${OUTPUT_DIR}/${template}.pc
      INPUT ${OUTPUT_DIR}${template}_${build_type}.pc.in
    )
  endif()

endfunction(hpx_generate_pkgconfig_from_target)
