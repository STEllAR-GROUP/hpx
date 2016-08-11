# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakePackageConfigHelpers)

set(CMAKE_DIR "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" CACHE STRING "directory (in share), where to put FindHPX cmake module")

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}ConfigVersion.cmake"
  VERSION ${HPX_VERSION}
  COMPATIBILITY AnyNewerVersion
)

export(TARGETS ${HPX_EXPORT_TARGETS}
  FILE "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXTargets.cmake"
#  NAMESPACE hpx::
)

if(HPX_WITH_EXPORT_PACKAGE)
  export(PACKAGE ${HPX_PACKAGE_NAME})
endif()

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

set(HPX_PKG_LIBRARY_DIR "")
foreach(dir ${HPX_LIBRARY_DIR})
  set(HPX_PKG_LIBRARY_DIR "${HPX_PKG_LIBRARY_DIR} -L${dir}")
endforeach()

if(HPX_WITH_STATIC_LINKING)
  set(HPX_CONF_LIBRARIES "hpx;${HPX_LIBRARIES}")
  set(HPX_PKG_LIBRARIES "\${libdir}/libhpx.a ${HPX_PKG_LIBRARIES}")
  set(HPX_PKG_DEBUG_LIBRARIES "\${libdir}/libhpxd.a ${HPX_PKG_DEBUG_LIBRARIES}")
else()
  set(HPX_CONF_LIBRARIES "hpx;hpx_init;${HPX_LIBRARIES}")
  if(APPLE)
    set(HPX_PKG_LIBRARIES "\${libdir}/libhpx.dylib \${libdir}/libhpx_init.a ${HPX_PKG_LIBRARIES}")
    set(HPX_PKG_DEBUG_LIBRARIES "\${libdir}/libhpxd.dylib \${libdir}/libhpx_initd.a ${HPX_PKG_DEBUG_LIBRARIES}")
  else()
    set(HPX_PKG_LIBRARIES "\${libdir}/libhpx.so \${libdir}/libhpx_init.a ${HPX_PKG_LIBRARIES}")
    set(HPX_PKG_DEBUG_LIBRARIES "\${libdir}/libhpxd.so \${libdir}/libhpx_initd.a ${HPX_PKG_DEBUG_LIBRARIES}")
  endif()
endif()

# Get the include directories we need ...
get_directory_property(_INCLUDE_DIRS INCLUDE_DIRECTORIES)

# replace all characters with special regex meaning
set(special_chars "^;+;*;?;$;.;-;|;(;);]")
set(binarydir_escaped ${CMAKE_BINARY_DIR})
set(sourcedir_escaped ${PROJECT_SOURCE_DIR})
foreach(special_char ${special_chars})
  string(REPLACE "${special_char}" "\\${special_char}" binarydir_escaped ${binarydir_escaped})
  string(REPLACE "${special_char}" "\\${special_char}" sourcedir_escaped ${sourcedir_escaped})
endforeach()

# '[' has special meaning in lists
string(REPLACE "[" "\\[" binarydir_escaped ${binarydir_escaped})
string(REPLACE "[" "\\[" sourcedir_escaped ${sourcedir_escaped})

foreach(dir ${_INCLUDE_DIRS})
  if((NOT dir MATCHES "^${binarydir_escaped}.*")
    AND (NOT dir MATCHES "^${sourcedir_escaped}.*"))
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
set(HPX_CONF_LIBRARY_DIR ${HPX_LIBRARY_DIR})

configure_file(cmake/templates/${HPX_PACKAGE_NAME}Config.cmake.in
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${HPX_PACKAGE_NAME}Config.cmake"
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
configure_file(cmake/templates/${HPX_PACKAGE_NAME}Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}Config.cmake"
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
set(HPX_CMAKE_MODULE_PATH "${CMAKE_INSTALL_PREFIX}/lib/cmake/${HPX_PACKAGE_NAME}")
configure_file(cmake/templates/HPXMacros.cmake.in
  "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/HPXMacros.cmake"
  ESCAPE_QUOTES @ONLY)
# ... and the build dir
set(HPX_CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
configure_file(cmake/templates/HPXMacros.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/HPXMacros.cmake"
  ESCAPE_QUOTES @ONLY)

install(
  EXPORT HPXTargets
  FILE HPXTargets.cmake
#  NAMESPACE hpx::
  DESTINATION ${LIB}/cmake/${HPX_PACKAGE_NAME}
)

install(
  FILES
    "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/${HPX_PACKAGE_NAME}Config.cmake"
    "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/HPXMacros.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/${HPX_PACKAGE_NAME}/${HPX_PACKAGE_NAME}ConfigVersion.cmake"
    DESTINATION ${LIB}/cmake/${HPX_PACKAGE_NAME}
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
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE
)
