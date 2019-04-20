# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

project(HPX.preprocessor CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(HPX_PREPROCESSOR_WITH_TESTS "Include tests for preprocessor" On)

message(STATUS "preprocessor: Configuring")

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/cat.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/config.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/expand.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/nargs.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/stringize.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/strip_parens.hpp)

add_library(hpx_preprocessor INTERFACE)

target_include_directories(hpx_preprocessor INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

# store set of files for later use in constructing source groups
set_property(TARGET hpx_preprocessor PROPERTY INTERFACE_HPX_HEADERS ${headers})

install(TARGETS hpx_preprocessor EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT preprocessor
)
hpx_export_targets(hpx_preprocessor)

install(
  DIRECTORY include/hpx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT preprocessor)

add_subdirectory(tests)

message(STATUS "preprocessor: Configuring done")
