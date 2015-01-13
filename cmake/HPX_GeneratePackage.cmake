# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakePackageConfigHelpers)

set(CMAKE_DIR "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" CACHE STRING "directory (in share), where to put FindHPX cmake module")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/hpx/HPXConfigVersion.cmake"
  VERSION ${HPX_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(TARGETS ${HPX_EXPORT_TARGETS}
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/hpx/HPXTargets.cmake"
#  NAMESPACE hpx::
)

export(PACKAGE hpx)

# Generate library list for pkg config ...
set(_is_debug FALSE)
set(_is_release FALSE)
foreach(lib ${HPX_LIBRARIES})
  if(lib STREQUAL "debug")
    set(_is_debug TRUE)
    set(_is_release FALSE)
  elseif(lib STREQUAL "optimized")
    set(_is_debug FALSE)
    set(_is_release TRUE)
  elseif(lib STREQUAL "general")
    set(_is_debug FALSE)
    set(_is_release FALSE)
  else()
    if(NOT EXISTS "${lib}")
      set(lib "-l${lib}")
    endif()
    if(_is_debug)
      set(HPX_PKG_DEBUG_LIBRARIES "${HPX_PKG_DEBUG_LIBRARIES} ${lib}")
    elseif(_is_release)
      set(HPX_PKG_LIBRARIES "${HPX_PKG_LIBRARIES} ${lib}")
    else()
      set(HPX_PKG_LIBRARIES "${HPX_PKG_LIBRARIES} ${lib}")
      set(HPX_PKG_DEBUG_LIBRARIES "${HPX_PKG_DEBUG_LIBRARIES} ${lib}")
    endif()
    set(_is_debug FALSE)
    set(_is_release FALSE)
  endif()
endforeach()

# Get the include directories we need ...
get_directory_property(_INCLUDE_DIRS INCLUDE_DIRECTORIES)
foreach(dir ${_INCLUDE_DIRS})
  if((NOT dir MATCHES "^${CMAKE_BINARY_DIR}.*")
    AND (NOT dir MATCHES "^${hpx_SOURCE_DIR}.*"))
    set(_NEEDED_INCLUDE_DIRS "${_NEEDED_INCLUDE_DIRS} -I${dir}")
    set(_NEEDED_CMAKE_INCLUDE_DIRS ${_NEEDED_CMAKE_INCLUDE_DIRS} "${dir}")
  else()
    set(_NEEDED_BUILD_DIR_INCLUDE_DIRS "${_NEEDED_BUILD_DIR_INCLUDE_DIRS} -I${dir}")
    set(_NEEDED_CMAKE_BUILD_DIR_INCLUDE_DIRS ${_NEEDED_CMAKE_BUILD_DIR_INCLUDE_DIRS} "${dir}")
  endif()
endforeach()

# Configure config for the install dir ...
set(HPX_CONF_INCLUDE_DIRS
  "-I${CMAKE_INSTALL_PREFIX}/include -I${CMAKE_INSTALL_PREFIX}/include/hpx/external ${_NEEDED_INCLUDE_DIRS}"
)
set(HPX_CMAKE_CONF_INCLUDE_DIRS
  "${CMAKE_INSTALL_PREFIX}/include"
  "${CMAKE_INSTALL_PREFIX}/include/hpx/external"
  "${_NEEDED_CMAKE_INCLUDE_DIRS}"
)
set(HPX_CONF_PREFIX ${CMAKE_INSTALL_PREFIX})
configure_file(cmake/templates/HPXConfig.cmake.in
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/HPXConfig.cmake"
  ESCAPE_QUOTES @ONLY)
configure_file(cmake/templates/hpx_application.pc.in
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx_application.pc"
  ESCAPE_QUOTES @ONLY)
configure_file(cmake/templates/hpx_application_debug.pc.in
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx_application_debug.pc"
  ESCAPE_QUOTES @ONLY)
configure_file(cmake/templates/hpx_component.pc.in
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx_component.pc"
  ESCAPE_QUOTES @ONLY)
configure_file(cmake/templates/hpx_component_debug.pc.in
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx_component_debug.pc"
  ESCAPE_QUOTES @ONLY)
# Configure hpxcxx
configure_file(cmake/templates/hpxcxx.in
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpxcxx"
  @ONLY)


# ... and the build dir
set(HPX_CONF_PREFIX "${CMAKE_BINARY_DIR}")
set(HPX_CONF_INCLUDE_DIRS
  "${_NEEDED_BUILD_DIR_INCLUDE_DIRS} ${_NEEDED_INCLUDE_DIRS}"
)
set(HPX_CMAKE_CONF_INCLUDE_DIRS
  ${_NEEDED_CMAKE_BUILD_DIR_INCLUDE_DIRS}
  ${_NEEDED_CMAKE_INCLUDE_DIRS}
)
configure_file(cmake/templates/HPXConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/hpx/HPXConfig.cmake"
  ESCAPE_QUOTES @ONLY)
configure_file(cmake/templates/hpx_application.pc.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/hpx_application.pc"
  ESCAPE_QUOTES @ONLY)
configure_file(cmake/templates/hpx_application_debug.pc.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/hpx_application_debug.pc"
  ESCAPE_QUOTES @ONLY)
configure_file(cmake/templates/hpx_component.pc.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/hpx_component.pc"
  ESCAPE_QUOTES @ONLY)
configure_file(cmake/templates/hpx_component_debug.pc.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/pkgconfig/hpx_component_debug.pc"
  ESCAPE_QUOTES @ONLY)
# Configure hpxcxx
configure_file(cmake/templates/hpxcxx.in
  "${CMAKE_CURRENT_BINARY_DIR}/bin/hpxcxx"
  @ONLY)

# Configure macros for the install dir ...
set(HPX_CMAKE_MODULE_PATH "${CMAKE_INSTALL_PREFIX}/lib/cmake/hpx")
configure_file(cmake/templates/HPXMacros.cmake.in
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/HPXMacros.cmake"
  ESCAPE_QUOTES @ONLY)
# ... and the build dir
set(HPX_CMAKE_MODULE_PATH "${hpx_SOURCE_DIR}/cmake")
configure_file(cmake/templates/HPXMacros.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/hpx/HPXMacros.cmake"
  ESCAPE_QUOTES @ONLY)

install(
  EXPORT HPXTargets
  FILE HPXTargets.cmake
#  NAMESPACE hpx::
  DESTINATION ${LIB}/cmake/hpx
)

install(
  FILES
    "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/HPXConfig.cmake"
    "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/HPXMacros.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/hpx/HPXConfigVersion.cmake"
    DESTINATION ${LIB}/cmake/hpx
  COMPONENT cmake
)

install(
  FILES
    "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx_application.pc"
    "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx_application_debug.pc"
    "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx_component.pc"
    "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpx_component_debug.pc"
  DESTINATION ${LIB}/pkgconfig
  COMPONENT pkgconfig
)

install(
  FILES
    "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/hpxcxx"
  DESTINATION bin
  COMPONENT compiler_wrapper
)
