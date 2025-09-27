# Copyright (c) 2025 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_AddCompileFlag)
include(HPX_Message)

# hpx_configure_module_producer(<producer> [MODULE_OUT_DIR <dir>])
#
# * Ensures a stable module output dir for producer target
# * Adds compiler flags to write module cache there (Clang/GCC)
# * Creates a target '<producer>_module' for consumers to link to
function(hpx_configure_module_producer producer)
  if(NOT TARGET ${producer})
    hpx_error("configure_module_producer: target '${producer}' not found")
  endif()

  # parse optional args
  set(options)
  set(one_value_args MODULE_OUT_DIR)
  set(multi_value_args)
  cmake_parse_arguments(
    _args "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN}
  )

  if(_args_MODULE_OUT_DIR)
    set(_moddir "${_args_MODULE_OUT_DIR}")
  else()
    set(_moddir "$<TARGET_FILE_DIR:${producer}>")
  endif()

  set(_iface "${producer}_if")
  if(NOT TARGET ${_iface})
    add_library(${_iface} INTERFACE)
    target_link_libraries(${_iface} INTERFACE ${producer})
    # target_include_directories(${_iface} INTERFACE "${_moddir}")
  endif()

  # Set a property so consumers can query the BMI directory via
  # get_target_property.
  set_target_properties(${producer} PROPERTIES EXPORT_MODULE_DIR "${_moddir}")
  set_target_properties(${producer} PROPERTIES CXX_SCAN_FOR_MODULES On)

  if(MSVC)
    # MSVC: CMake/MSVC handle IFCs automatically; create a target for
    # convenience, consumers can link to this to get ordering and include info
    return()
  endif()

  # Compiler-specific flags to instruct where to write module cache
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES
                                              "AppleClang"
  )
    # Clang common flags
    target_compile_options(${producer} PRIVATE "-fmodule-output=${_moddir}")
    target_compile_options(
      ${_iface} INTERFACE "-fprebuilt-module-path=${_moddir}"
    )
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # GCC: try a few likely flags depending on version Prefer flags used in GCC
    # 11+: -fmodules-ts, -fmodules-cache-path (or -fmodule-cache-path)
    hpx_add_target_compile_option_if_available(
      ${producer} PRIVATE "-fmodule-output=${_moddir}" RESULT ok
    )
    if(NOT ok)
      hpx_error(
        "configure_module_producer: the used version of gcc does not support '-fmodule-output'"
      )
    endif()
    hpx_add_target_compile_option_if_available(
      ${_iface} INTERFACE "-fprebuilt-module-path=${_moddir}" RESULT ok
    )
    if(NOT ok)
      hpx_error(
        "configure_module_producer: the used version of gcc does not support '-fprebuilt-module-path='"
      )
    endif()
  else()
    hpx_warn(
      "configure_module_producer: unknown compiler '${CMAKE_CXX_COMPILER_ID}'; "
      "exposing CXX_MODULE_OUTPUT_DIRECTORY='${_moddir}' for manual handling"
    )
  endif()
endfunction()
