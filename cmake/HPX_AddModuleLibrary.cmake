# Copyright (c) 2022 Hartmut Kaiser
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(HPX_CXX_MODULES_FLAGS /experimental:module)
  set(HPX_CXX_MODULES_INTERFACE_FLAGS /interface)
  set(HPX_CXX_MODULES_INTERNAL_PARTITION /internalPartition)
  set(HPX_CXX_MODULES_MODULE_ONLY /ifcOnly)
  set(HPX_CXX_MODULES_TRANSLATE_HEADER /translateInclude)
  set(HPX_CXX_MODULES_EXT ifc)
  set(HPX_CXX_MODULES_CREATE_FLAGS -c)
  set(HPX_CXX_MODULES_USE_FLAG /reference)
  set(HPX_CXX_MODULES_OUTPUT_FLAG /ifcOutput)
  set(HPX_CXX_MODULES_BMI_SEARCHDIR /ifcSearchDir)
else()
  hpx_error("C++20 modules are currently supported for MSVC only")
endif()

function(add_hpx_module_library libname modulename)

  set(options)
  set(one_value_args LINKTYPE FOLDER)
  set(multi_value_args MODULE_SOURCES MODULE_HEADERS SOURCES HEADERS OBJECTS)
  cmake_parse_arguments(
    ${modulename} "${options}" "${one_value_args}" "${multi_value_args}"
    ${ARGN}
  )

  hpx_debug("Add C++20 module ${modulename}: SOURCES: ${${modulename}_SOURCES}")
  hpx_debug("Add C++20 module ${modulename}: HEADERS: ${${modulename}_HEADERS}")
  hpx_debug(
    "Add C++20 module ${modulename}: MODULE_SOURCES: ${${modulename}_MODULE_SOURCES}"
  )
  hpx_debug(
    "Add C++20 module ${modulename}: MODULE_HEADERS: ${${modulename}_MODULE_HEADERS}"
  )

  # Allow to use CXX compiler on C++ module files
  set_source_files_properties(
    ${${modulename}_MODULE_SOURCES} PROPERTIES LANGUAGE CXX
  )
  set_source_files_properties(
    ${${modulename}_MODULE_HEADERS} PROPERTIES LANGUAGE CXX
  )

  if(NOT ${modulename}_LINKTYPE)
    set(${modulename}_LINKTYPE STATIC)
  endif()
  
  set(hpx_module_name "hpx.${libname}.${modulename}.${HPX_CXX_MODULES_EXT}")
  set(hpx_module_path "${module_output_dir}/${hpx_module_name}")

  # Create normal library
  add_library(
    hpx_module_name
    ${${modulename}_LINKTYPE}
    ${${modulename}_SOURCES}
    ${${modulename}_MODULE_SOURCES}
    ${${modulename}_HEADERS}
    ${${modulename}_MODULE_HEADERS}
    ${${modulename}_OBJECTS}
    ${${modulename}_UNPARSED_ARGUMENTS}
  )

  # Enable modules for target
  if(MSVC)
    file(TO_NATIVE_PATH "${PROJECT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/BMI/"
         module_output_dir
    )
  else()
    file(TO_NATIVE_PATH "${PROJECT_BINARY_DIR}/BMI/" module_output_dir)
  endif()



  # Create targets for interface files
  foreach(source ${${module}_MODULE_SOURCES})
    set_source_files_properties(
      ${source} PROPERTIES COMPILE_OPTIONS /translateInclude
    )
  endforeach()

  target_compile_definitions(${hpx_module_name} PRIVATE HPX_HAVE_CXX20_MODULES)

  target_compile_options(${hpx_module_name} PRIVATE ${HPX_CXX_MODULES_FLAGS})
  target_compile_options(
    ${hpx_module_name}
    PRIVATE "${HPX_CXX_MODULES_BMI_SEARCHDIR}${module_output_dir}"
  )
  target_compile_options(
    ${hpx_module_name}
    PRIVATE "${HPX_CXX_MODULES_OUTPUT_FLAG}${hpx_module_path}"
  )
  target_compile_options(
    ${hpx_module_name}
    PRIVATE "${HPX_CXX_MODULES_INTERFACE_FLAGS}${source}"
  )

  add_executable(hpx_${modulename})
endfunction()
