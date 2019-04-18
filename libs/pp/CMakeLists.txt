# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

project(HPX.pp CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(HPX_PP_WITH_TESTS "Include tests for pp" On)

message(STATUS "pp: Configuring")

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/cat.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/config.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/expand.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/nargs.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/stringize.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/pp/strip_parens.hpp)

add_library(hpx_pp INTERFACE)

target_include_directories(hpx_pp INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

set_target_properties(hpx_pp PROPERTIES INTERFACE_HPX_SOURCES "")
set_target_properties(hpx_pp PROPERTIES INTERFACE_HPX_HEADERS "${headers}")

install(TARGETS hpx_pp EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT pp
)
hpx_export_targets(hpx_pp)

install(
  DIRECTORY include/hpx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT pp)

add_subdirectory(tests)

message(STATUS "pp: Configuring done")
