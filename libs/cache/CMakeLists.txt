# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

project(HPX.cache CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(HPX_AddDefinitions)
include(HPX_Option)

hpx_option(HPX_CACHE_WITH_TESTS
  BOOL
  "Build HPX cache module tests. (default: ON)"
  ON ADVANCED
  CATEGORY "Modules")

hpx_option(HPX_CACHE_WITH_DEPRECATION_WARNINGS
  BOOL
  "Enable warnings for deprecated facilities. (default: ${HPX_WITH_DEPRECATION_WARNINGS})"
  ${HPX_WITH_DEPRECATION_WARNINGS} ADVANCED
  CATEGORY "Modules")

if(HPX_CACHE_WITH_DEPRECATION_WARNINGS)
  hpx_add_config_define_namespace(
      DEFINE HPX_CACHE_HAVE_DEPRECATION_WARNINGS
      NAMESPACE CACHE)
endif()

# Added in 1.4.0
hpx_option(HPX_CACHE_WITH_COMPATIBILITY_HEADERS
  BOOL
  "Enable compatibility headers for old headers"
  ON ADVANCED
  CATEGORY "Modules")

if(HPX_CACHE_WITH_COMPATIBILITY_HEADERS)
  hpx_add_config_define_namespace(
      DEFINE HPX_CACHE_HAVE_COMPATIBILITY_HEADERS
      NAMESPACE CACHE)
endif()

message(STATUS "cache: Configuring")

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/local_cache.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/lru_cache.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/entries/entry.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/entries/fifo_entry.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/entries/lfu_entry.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/entries/lru_entry.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/entries/size_entry.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/policies/always.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/statistics/local_full_statistics.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/statistics/local_statistics.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/cache/statistics/no_statistics.hpp
)

if(HPX_CACHE_WITH_COMPATIBILITY_HEADERS)
  set(compat_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/local_cache.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/lru_cache.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/entries/entry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/entries/fifo_entry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/entries/lfu_entry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/entries/lru_entry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/entries/size_entry.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/policies/always.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/statistics/local_full_statistics.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/statistics/local_statistics.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/cache/statistics/no_statistics.hpp
  )
endif()

set(sources
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cache.cpp
)

add_library(hpx_cache STATIC ${sources} ${headers} ${compat_headers})

target_link_libraries(hpx_cache hpx_config)
target_include_directories(hpx_cache PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)

if(HPX_CACHE_WITH_COMPATIBILITY_HEADERS)
  target_include_directories(hpx_cache PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility>)
endif()

target_compile_definitions(hpx_cache PRIVATE
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

if(HPX_CACHE_WITH_COMPATIBILITY_HEADERS)
  add_hpx_source_group(
    NAME hpx
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx
    CLASS "Header Files"
    TARGETS ${compat_headers})
endif()

set_target_properties(hpx_cache PROPERTIES
  FOLDER "Core/Modules")

install(TARGETS hpx_cache EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT cache
)
hpx_export_targets(hpx_cache)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT cache)

if(HPX_CACHE_WITH_COMPATIBILITY_HEADERS)
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT cache)
endif()

write_config_defines_file(
  NAMESPACE CACHE
  FILENAME "${CMAKE_BINARY_DIR}/hpx/cache/config/defines.hpp")

write_config_defines_file(
  NAMESPACE CACHE
  FILENAME "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx/cache/config/defines.hpp")

add_subdirectory(examples)
add_subdirectory(tests)

message(STATUS "cache: Configuring done")
