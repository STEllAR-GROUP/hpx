# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakePackageConfigHelpers)
include(HPXLocal_GeneratePackageUtils)

set(CMAKE_DIR
    "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}"
    CACHE STRING "directory (in share), where to put FindHPX cmake module"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/${HPXLocal_PACKAGE_NAME}ConfigVersion.cmake"
  VERSION ${HPXLocal_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Export HPXLocalInternalTargets in the build directory
export(
  TARGETS ${HPXLocal_EXPORT_INTERNAL_TARGETS}
  NAMESPACE HPXLocalInternal::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/HPXLocalInternalTargets.cmake"
)

# Export HPXLocalInternalTargets in the install directory
install(
  EXPORT HPXLocalInternalTargets
  NAMESPACE HPXLocalInternal::
  FILE HPXLocalInternalTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPXLocal_PACKAGE_NAME}
)

# Export HPXLocalTargets in the build directory
export(
  TARGETS ${HPXLocal_EXPORT_TARGETS}
  NAMESPACE HPX::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/HPXLocalTargets.cmake"
)

# Add aliases with the namespace for use within HPX
foreach(export_target ${HPXLocal_EXPORT_TARGETS})
  add_library(HPX::${export_target} ALIAS ${export_target})
endforeach()

foreach(export_target ${HPXLocal_EXPORT_INTERNAL_TARGETS})
  add_library(HPXLocalInternal::${export_target} ALIAS ${export_target})
endforeach()

# Export HPXLocalTargets in the install directory
install(
  EXPORT HPXLocalTargets
  NAMESPACE HPX::
  FILE HPXLocalTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPXLocal_PACKAGE_NAME}
)

# Install dir
configure_file(
  cmake/templates/${HPXLocal_PACKAGE_NAME}Config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPXLocal_PACKAGE_NAME}Config.cmake"
  ESCAPE_QUOTES
  @ONLY
)
# Build dir
configure_file(
  cmake/templates/${HPXLocal_PACKAGE_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/${HPXLocal_PACKAGE_NAME}Config.cmake"
  ESCAPE_QUOTES
  @ONLY
)

# Configure macros for the install dir ...
set(HPXLocal_CMAKE_MODULE_PATH "\${CMAKE_CURRENT_LIST_DIR}")
configure_file(
  cmake/templates/${HPXLocal_PACKAGE_NAME}Macros.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPXLocal_PACKAGE_NAME}Macros.cmake"
  ESCAPE_QUOTES
  @ONLY
)
# ... and the build dir
set(HPXLocal_CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
configure_file(
  cmake/templates/${HPXLocal_PACKAGE_NAME}Macros.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/${HPXLocal_PACKAGE_NAME}Macros.cmake"
  ESCAPE_QUOTES
  @ONLY
)

install(
  FILES
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPXLocal_PACKAGE_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPXLocal_PACKAGE_NAME}Macros.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPXLocal_PACKAGE_NAME}/${HPXLocal_PACKAGE_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPXLocal_PACKAGE_NAME}
  COMPONENT cmake
)

# This exists purely for HPX. It will generate cache variables which HPX can add
# to the pkgconfig files.
if(NOT
   (MSVC
    OR APPLE
    OR HPXLocal_WITH_GPU_SUPPORT)
   AND HPX_WITH_PKGCONFIG
)
  set(pkgconfig_EXCLUDE Threads::Threads)

  # First generate pkgconfig variables for the build directory
  hpx_local_collect_usage_requirements(
    HPX::hpx_local
    hpx_local_compile_definitions
    hpx_local_compile_options
    hpx_local_pic_option
    hpx_local_include_directories
    hpx_local_system_include_directories
    hpx_local_link_libraries
    hpx_local_link_options
    processed_targets
    false
    EXCLUDE ${pkgconfig_EXCLUDE}
  )

  hpx_local_sanitize_usage_requirements(hpx_local_compile_definitions true)
  hpx_local_sanitize_usage_requirements(hpx_local_compile_options true)
  hpx_local_sanitize_usage_requirements(hpx_local_include_directories true)
  hpx_local_sanitize_usage_requirements(
    hpx_local_system_include_directories true
  )
  hpx_local_sanitize_usage_requirements(hpx_local_link_libraries true)
  hpx_local_sanitize_usage_requirements(hpx_local_link_options true)

  hpx_local_construct_cflag_list(
    hpx_local_compile_definitions hpx_local_compile_options
    hpx_local_pic_option hpx_local_include_directories
    hpx_local_system_include_directories hpx_local_cflags_list
  )
  hpx_local_filter_cuda_flags(hpx_local_cflags_list)
  hpx_local_construct_library_list(
    hpx_local_link_libraries hpx_local_link_options hpx_library_list
  )
  set(HPXLocal_BUILD_PKGCONFIG_CFLAGS_LIST
      ${hpx_local_cflags_list}
      CACHE INTERNAL "" FORCE
  )
  set(HPXLocal_BUILD_PKGCONFIG_LIBRARY_LIST
      ${hpx_library_list}
      CACHE INTERNAL "" FORCE
  )

  # Then generate pkgconfig variables for the install directory
  set(hpx_library_list)
  set(hpx_local_cflags_list)
  set(hpx_local_compile_definitions)
  set(hpx_local_compile_options)
  set(hpx_local_pic_option)
  set(hpx_local_include_directories)
  set(hpx_local_system_include_directories)
  set(hpx_local_link_libraries)
  set(hpx_local_link_options)
  set(processed_targets)

  hpx_local_collect_usage_requirements(
    HPX::hpx_local
    hpx_local_compile_definitions
    hpx_local_compile_options
    hpx_local_pic_option
    hpx_local_include_directories
    hpx_local_system_include_directories
    hpx_local_link_libraries
    hpx_local_link_options
    processed_targets
    false
    EXCLUDE ${pkgconfig_EXCLUDE}
  )

  hpx_local_sanitize_usage_requirements(hpx_local_compile_definitions false)
  hpx_local_sanitize_usage_requirements(hpx_local_compile_options false)
  hpx_local_sanitize_usage_requirements(hpx_local_include_directories false)
  hpx_local_sanitize_usage_requirements(
    hpx_local_system_include_directories false
  )
  hpx_local_sanitize_usage_requirements(hpx_local_link_libraries false)
  hpx_local_sanitize_usage_requirements(hpx_local_link_options false)

  hpx_local_construct_cflag_list(
    hpx_local_compile_definitions hpx_local_compile_options
    hpx_local_pic_option hpx_local_include_directories
    hpx_local_system_include_directories hpx_local_cflags_list
  )
  hpx_local_filter_cuda_flags(hpx_local_cflags_list)
  hpx_local_construct_library_list(
    hpx_local_link_libraries hpx_local_link_options hpx_library_list
  )
  set(HPXLocal_INSTALL_PKGCONFIG_CFLAGS_LIST
      ${hpx_local_cflags_list}
      CACHE INTERNAL "" FORCE
  )
  set(HPXLocal_INSTALL_PKGCONFIG_LIBRARY_LIST
      ${hpx_library_list}
      CACHE INTERNAL "" FORCE
  )
endif()
