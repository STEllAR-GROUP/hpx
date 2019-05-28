# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

project(HPX.config CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(HPX_AddDefinitions)
include(HPX_Option)

hpx_option(HPX_CONFIG_WITH_TESTS
  BOOL
  "Build HPX config module tests. (default: ON)"
  ON ADVANCED
  CATEGORY "Modules")

message(STATUS "config: Configuring")

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/asio.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/attributes.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/autolink.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/branch_hints.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/compiler_fence.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/compiler_native_tls.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/compiler_specific.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/constexpr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/debug.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/emulate_deleted.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/export_definitions.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/forceinline.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/lambda_capture.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/manual_profiling.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/threads_stack.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/warnings_prefix.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/warnings_suffix.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/config/weak_symbol.hpp
)

set(sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/version.cpp
)

add_library(hpx_config STATIC ${sources} ${headers})

target_link_libraries(hpx_config hpx_preprocessor)
target_include_directories(hpx_config PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)

target_compile_definitions(hpx_config PRIVATE
  $<$<CONFIG:Debug>:DEBUG>
  $<$<CONFIG:Debug>:_DEBUG>
  HPX_MODULE_EXPORTS
)

include(HPX_AddSourceGroup)
add_hpx_source_group(
  NAME hpx
  ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx
  CLASS "Header Files"
  TARGETS ${headers})
add_hpx_source_group(
  NAME hpx
  ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src
  CLASS "Source Files"
  TARGETS ${sources})

set_target_properties(hpx_config PROPERTIES
  FOLDER "Core/Modules")

install(TARGETS hpx_config EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT config
)
hpx_export_targets(hpx_config)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT config)

add_subdirectory(examples)
add_subdirectory(tests)

message(STATUS "config: Configuring done")
