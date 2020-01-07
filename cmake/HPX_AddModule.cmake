# Copyright (c) 2019 Auriane Reverdell
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_ExportTargets)

function(add_hpx_module name)
  # Retrieve arguments
  set(options DEPRECATION_WARNINGS FORCE_LINKING_GEN
    CUDA CONFIG_FILES)
  # Compatibility needs to be on/off to allow 3 states : ON/OFF and disabled
  set(one_value_args COMPATIBILITY_HEADERS GLOBAL_HEADER_GEN)
  set(multi_value_args SOURCES HEADERS COMPAT_HEADERS DEPENDENCIES CMAKE_SUBDIRS
    EXCLUDE_FROM_GLOBAL_HEADER)
  cmake_parse_arguments(${name} "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  include(HPX_Message)
  include(HPX_Option)

  # Global headers should be always generated except if explicitly disabled
  if ("${${name}_GLOBAL_HEADER_GEN}" STREQUAL "")
      set(${name}_GLOBAL_HEADER_GEN ON)
  endif()

  string(TOUPPER ${name} name_upper)

  # enable the module (see hpx/libs/CMakeLists.txt)
  set_property(GLOBAL PROPERTY HPX_${name}_LIBRARY_ENABLED ON)

  # HPX options
  hpx_option(HPX_${name_upper}_WITH_TESTS
    BOOL
    "Build HPX ${name} module tests. (default: ${HPX_WITH_TESTS})"
    ${HPX_WITH_TESTS} ADVANCED
    CATEGORY "Modules")

  set(_deprecation_warnings_default OFF)
  if(${HPX_WITH_DEPRECATION_WARNINGS} AND ${name}_DEPRECATION_WARNINGS)
    set(_deprecation_warnings_default ON)
  endif()
  hpx_option(HPX_${name_upper}_WITH_DEPRECATION_WARNINGS
    BOOL
    "Enable warnings for deprecated facilities. (default: ${HPX_WITH_DEPRECATION_WARNINGS})"
    ${_deprecation_warnings_default} ADVANCED
    CATEGORY "Modules")
  if(${HPX_${name_upper}_WITH_DEPRECATION_WARNINGS})
    hpx_add_config_define_namespace(
      DEFINE HPX_${name_upper}_HAVE_DEPRECATION_WARNINGS
      NAMESPACE ${name_upper})
  endif()

  if(NOT "${${name}_COMPATIBILITY_HEADERS}" STREQUAL "")
    set(_compatibility_headers_default OFF)
    if(${name}_COMPATIBILITY_HEADERS)
      set(_compatibility_headers_default ON)
    endif()
    hpx_option(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS
      BOOL
      "Enable compatibility headers for old headers. (default: ${_compatibility_headers_default})"
      ${_compatibility_headers_default} ADVANCED
      CATEGORY "Modules")
    if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
      hpx_add_config_define_namespace(
        DEFINE HPX_${name_upper}_HAVE_COMPATIBILITY_HEADERS
        NAMESPACE ${name_upper})
    endif()
  endif()

  # Main directories of the module
  set(SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/src")
  set(HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include")

  hpx_debug("Add module ${name}: SOURCE_ROOT: ${SOURCE_ROOT}")
  hpx_debug("Add module ${name}: HEADER_ROOT: ${HEADER_ROOT}")

  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    set(COMPAT_HEADER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility")
    hpx_debug("Add module ${name}: COMPAT_HEADER_ROOT: ${COMPAT_HEADER_ROOT}")
  endif()

  # Write full path for the sources files
  include(HPX_CMakeUtils)
  prepend(sources ${SOURCE_ROOT} ${${name}_SOURCES})
  prepend(headers ${HEADER_ROOT} ${${name}_HEADERS})
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    prepend(compat_headers ${COMPAT_HEADER_ROOT} ${${name}_COMPAT_HEADERS})
  endif()

  # This header generation is disabled for config module specific generated
  # headers are included
  if (${name}_GLOBAL_HEADER_GEN)
    # Add a global include file that include all module headers
    set(global_header "${CMAKE_CURRENT_BINARY_DIR}/include/hpx/${name}.hpp")
    set(module_headers)
    foreach(header_file ${${name}_HEADERS})
      # Exclude the files specified
      if (NOT (${header_file} IN_LIST ${name}_EXCLUDE_FROM_GLOBAL_HEADER))
        set(module_headers "${module_headers}#include <${header_file}>\n")
      endif()
    endforeach(header_file)
    configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/global_module_header.hpp.in"
      "${global_header}")
    set(generated_headers ${global_header})
  endif()

  if(${name}_FORCE_LINKING_GEN)
      # Add a header to force linking of modules on Windows
      set(force_linking_header "${CMAKE_CURRENT_BINARY_DIR}/include/hpx/${name}/force_linking.hpp")
      configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/force_linking.hpp.in"
        "${force_linking_header}")

      # Add a source file implementing the above function
      set(force_linking_source "${CMAKE_CURRENT_BINARY_DIR}/src/force_linking.cpp")
      configure_file("${PROJECT_SOURCE_DIR}/cmake/templates/force_linking.cpp.in"
        "${force_linking_source}")
  endif()

  # generate configuration header for this module
  set(config_header
    "${CMAKE_CURRENT_BINARY_DIR}/include/hpx/${name}/config/defines.hpp")
  write_config_defines_file(NAMESPACE ${name_upper} FILENAME ${config_header})
  set(generated_headers ${generated_headers} ${config_header})

  if (${name}_CONFIG_FILES)
    # Version file
    set(global_config_file ${CMAKE_CURRENT_BINARY_DIR}/include/hpx/config/version.hpp)
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/config_version.hpp.in"
      "${global_config_file}"
      @ONLY)
    set(generated_headers ${generated_headers} ${global_config_file})
    # Global config defines file (different from the one for each module)
    set(global_config_file ${CMAKE_CURRENT_BINARY_DIR}/include/hpx/config/defines.hpp)
    write_config_defines_file(
      TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/cmake/templates/config_defines.hpp.in"
      NAMESPACE default
      FILENAME "${global_config_file}")
    set(generated_headers ${generated_headers} ${global_config_file})
  endif()

  # list all specified headers
  foreach(header_file ${headers})
    hpx_debug(${header_file})
  endforeach(header_file)

  # create library modules
  if(${name}_CUDA AND HPX_WITH_CUDA)
    cuda_add_library(hpx_${name} STATIC
      ${sources} ${force_linking_source}
      ${headers} ${force_linking_header}
      ${generated_headers} ${compat_headers})
  else()
    add_library(hpx_${name} STATIC
      ${sources} ${force_linking_source}
      ${headers} ${force_linking_header}
      ${generated_headers} ${compat_headers})
  endif()

  target_link_libraries(hpx_${name} PUBLIC ${${name}_DEPENDENCIES})
  target_include_directories(hpx_${name} PUBLIC
    $<BUILD_INTERFACE:${HEADER_ROOT}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

  target_link_libraries(hpx_${name} PRIVATE hpx_internal_flags)

  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    target_include_directories(hpx_${name} PUBLIC
      $<BUILD_INTERFACE:${COMPAT_HEADER_ROOT}>)
  endif()

  target_compile_definitions(hpx_${name} PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Debug>:_DEBUG>
    HPX_MODULE_EXPORTS
  )

  # This is a temporary solution until all of HPX has been modularized as it
  # enables using header files from HPX for compiling this module.
  target_include_directories(hpx_${name} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
  )

  add_hpx_source_group(
    NAME hpx_{name}
    ROOT ${HEADER_ROOT}/hpx
    CLASS "Header Files"
    TARGETS ${headers})
  add_hpx_source_group(
    NAME hpx_{name}
    ROOT ${SOURCE_ROOT}
    CLASS "Source Files"
    TARGETS ${sources})
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    add_hpx_source_group(
      NAME hpx_${name}
      ROOT ${COMPAT_HEADER_ROOT}/hpx
      CLASS "Header Files"
      TARGETS ${compat_headers})
  endif()

  if (${name}_GLOBAL_HEADER_GEN OR ${name}_CONFIG_FILES)
    add_hpx_source_group(
      NAME hpx_{name}
      ROOT ${CMAKE_CURRENT_BINARY_DIR}/include/hpx
      CLASS "Generated Files"
      TARGETS ${generated_headers})
  endif()
  if (${name}_FORCE_LINKING_GEN)
    add_hpx_source_group(
      NAME hpx_{name}
      ROOT ${CMAKE_CURRENT_BINARY_DIR}/include/hpx
      CLASS "Generated Files"
      TARGETS ${force_linking_header})
    add_hpx_source_group(
      NAME hpx_{name}
      ROOT ${CMAKE_CURRENT_BINARY_DIR}/src
      CLASS "Generated Files"
      TARGETS ${force_linking_source})
  endif()
  add_hpx_source_group(
    NAME hpx_{name}
    ROOT ${CMAKE_CURRENT_BINARY_DIR}/include/hpx
    CLASS "Generated Files"
    TARGETS ${config_header})

  set_target_properties(hpx_${name} PROPERTIES
    FOLDER "Core/Modules"
    POSITION_INDEPENDENT_CODE ON)

  if(MSVC)
    set_target_properties(hpx_${name} PROPERTIES
      COMPILE_PDB_NAME_DEBUG hpx_${name}d
      COMPILE_PDB_NAME_RELWITHDEBINFO hpx_${name}
      COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG
        ${CMAKE_CURRENT_BINARY_DIR}/Debug
      COMPILE_PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO
        ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo)
  endif()

  # Install the static library for the module
  install(TARGETS hpx_${name} EXPORT HPXModulesTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT ${name}
  )
  hpx_export_modules_targets(hpx_${name})

  # Install the headers from the source
  install(
    DIRECTORY include/hpx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${name})

  # Install the compatibility headers from the source
  if(HPX_${name_upper}_WITH_COMPATIBILITY_HEADERS)
    install(
      DIRECTORY include_compatibility/hpx
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
      COMPONENT ${name})
  endif()

  # Installing the generated header files from the build dir
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/hpx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT ${name}
  )

  # install PDB if needed
  if(MSVC)
    foreach(cfg DEBUG;RELWITHDEBINFO)
      get_target_property(_pdb_file hpx_${name} COMPILE_PDB_NAME_${cfg})
      get_target_property(_pdb_dir hpx_${name} COMPILE_PDB_OUTPUT_DIRECTORY_${cfg})
      install(
        FILES ${_pdb_dir}/${_pdb_file}.pdb
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        CONFIGURATIONS ${cfg}
        OPTIONAL
      )
    endforeach()
  endif()

  foreach(dir ${${name}_CMAKE_SUBDIRS})
    add_subdirectory(${dir})
  endforeach(dir)

  include(HPX_PrintSummary)
  create_configuration_summary(
    "  Module configuration summary (${name}):" "${name}")

endfunction(add_hpx_module)
