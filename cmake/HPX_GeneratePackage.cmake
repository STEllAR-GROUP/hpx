# Copyright (c) 2014 Thomas Heller
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(CMakePackageConfigHelpers)

set(CMAKE_DIR "cmake-${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" CACHE STRING "directory (in share), where to put FindHPX cmake module")

if(MSVC)
  set(output_dir "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
else()
  set(output_dir "${CMAKE_BINARY_DIR}")
endif()


write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/HPXConfigVersion.cmake"
  VERSION ${HPX_VERSION}
  COMPATIBILITY AnyNewerVersion
)

message("${HPX_EXPORT_TARGETS} <--?")

export(TARGETS hpx_serialization hpx hpx_init
  FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/HPXTargets.cmake"
  NAMESPACE hpx::
)

export(PACKAGE hpx)

configure_file(cmake/templates/HPXConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/HPXConfig.cmake"
  ESCAPE_QUOTES @ONLY)

install(
  EXPORT HPXTargets
  FILE HPXTargets.cmake
  NAMESPACE hpx::
  DESTINATION lib/cmake
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/HPXConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/HPXConfigVersion.cmake"
  DESTINATION lib/cmake
  COMPONENT cmake
)
