# Copyright (c) 2018 Christopher Hinz
# Copyright (c) 2014 Thomas Heller
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakePackageConfigHelpers)
include(HPX_GeneratePackageUtils)

set(CMAKE_DIR
    "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}"
    CACHE STRING "directory (in share), where to put FindHPX cmake module"
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}ConfigVersion.cmake"
  VERSION ${HPX_VERSION}
  COMPATIBILITY AnyNewerVersion
)

# Export HPXInternalTargets in the build directory
export(
  TARGETS ${HPX_EXPORT_INTERNAL_TARGETS}
  NAMESPACE HPXInternal::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXInternalTargets.cmake"
)

# Export HPXInternalTargets in the install directory
install(
  EXPORT HPXInternalTargets
  NAMESPACE HPXInternal::
  FILE HPXInternalTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
)

# Export HPXTargets in the build directory
export(
  TARGETS ${HPX_EXPORT_TARGETS}
  NAMESPACE HPX::
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXTargets.cmake"
)

# Export HPXTargets in the install directory
install(
  EXPORT HPXTargets
  NAMESPACE HPX::
  FILE HPXTargets.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
)

if(NOT MSVC)
  add_library(hpx_pkgconfig_application INTERFACE)
  target_link_libraries(
    hpx_pkgconfig_application INTERFACE hpx hpx_wrap hpx_init
  )
  target_compile_definitions(
    hpx_pkgconfig_application INTERFACE HPX_APPLICATION_EXPORTS
  )
  target_compile_options(
    hpx_pkgconfig_application INTERFACE "-std=c++${HPX_CXX_STANDARD}"
  )

  add_library(hpx_pkgconfig_component INTERFACE)
  target_compile_definitions(
    hpx_pkgconfig_component INTERFACE HPX_COMPONENT_EXPORTS
  )
  target_link_libraries(hpx_pkgconfig_component INTERFACE hpx)
  target_compile_options(
    hpx_pkgconfig_component INTERFACE "-std=c++${HPX_CXX_STANDARD}"
  )

  # Generate the pkconfig files for HPX_APPLICATION (both for build and install)
  hpx_generate_pkgconfig_from_target(
    hpx_pkgconfig_application hpx_application TRUE EXCLUDE hpx_interface
  )
  hpx_generate_pkgconfig_from_target(
    hpx_pkgconfig_application hpx_application FALSE EXCLUDE hpx_interface
  )
  # Generate the pkconfig files for HPX_COMPONENT (both for build and install)
  hpx_generate_pkgconfig_from_target(
    hpx_pkgconfig_component hpx_component TRUE EXCLUDE hpx_interface
  )
  hpx_generate_pkgconfig_from_target(
    hpx_pkgconfig_component hpx_component FALSE EXCLUDE hpx_interface
  )
endif()

# Install dir
configure_file(
  cmake/templates/${HPX_PACKAGE_NAME}Config.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPX_PACKAGE_NAME}Config.cmake"
  ESCAPE_QUOTES
  @ONLY
)
set(HPX_CONF_PREFIX ${CMAKE_INSTALL_PREFIX})
configure_file(
  cmake/templates/hpxcxx.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpxcxx" @ONLY
)
# Build dir
configure_file(
  cmake/templates/${HPX_PACKAGE_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}Config.cmake"
  ESCAPE_QUOTES
  @ONLY
)
set(HPX_CONF_PREFIX ${PROJECT_BINARY_DIR})
configure_file(
  cmake/templates/hpxcxx.in "${CMAKE_CURRENT_BINARY_DIR}/bin/hpxcxx" @ONLY
)

# Configure macros for the install dir ...
set(HPX_CMAKE_MODULE_PATH "\${CMAKE_CURRENT_LIST_DIR}")
configure_file(
  cmake/templates/HPXMacros.cmake.in
  "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/HPXMacros.cmake" ESCAPE_QUOTES
  @ONLY
)
# ... and the build dir
set(HPX_CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
configure_file(
  cmake/templates/HPXMacros.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXMacros.cmake"
  ESCAPE_QUOTES @ONLY
)

install(
  FILES
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/${HPX_PACKAGE_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/HPXMacros.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${HPX_PACKAGE_NAME}
  COMPONENT cmake
)

string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
if(NOT MSVC)
  install(
    FILES ${OUTPUT_DIR_PC}/hpx_application_${build_type}.pc
          ${OUTPUT_DIR_PC}/hpx_component_${build_type}.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    COMPONENT pkgconfig
  )
  # Temporary (to deprecate gradually)
  install(
    FILES ${OUTPUT_DIR_PC}/hpx_application.pc ${OUTPUT_DIR_PC}/hpx_component.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
    CONFIGURATIONS Release
    COMPONENT pkgconfig
  )
endif()

install(
  FILES "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpxcxx"
  DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT compiler_wrapper
  PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
)
