# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

cmake_minimum_required(VERSION 3.3.2 FATAL_ERROR)

project(HPX.preprocessor CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(HPX_AddDefinitions)
include(HPX_Message)
include(HPX_Option)

hpx_info("  preprocessor")

hpx_option(HPX_PREPROCESSOR_WITH_TESTS
  BOOL
  "Build HPX preprocessor module tests. (default: ${HPX_WITH_TESTS})"
  ${HPX_WITH_TESTS} ADVANCED
  CATEGORY "Modules")

hpx_option(HPX_PREPROCESSOR_WITH_DEPRECATION_WARNINGS
  BOOL
  "Enable warnings for deprecated facilities. (default: ${HPX_WITH_DEPRECATION_WARNINGS})"
  ${HPX_WITH_DEPRECATION_WARNINGS} ADVANCED
  CATEGORY "Modules")
if(HPX_PREPROCESSOR_WITH_DEPRECATION_WARNINGS)
  hpx_add_config_define_namespace(
    DEFINE HPX_PREPROCESSOR_HAVE_DEPRECATION_WARNINGS
    NAMESPACE PREPROCESSOR)
endif()

# Added in 1.3.0
hpx_option(HPX_PREPROCESSOR_WITH_COMPATIBILITY_HEADERS
  BOOL
  "Enable compatibility headers for old headers"
  ON ADVANCED
  CATEGORY "Modules")

if(HPX_PREPROCESSOR_WITH_COMPATIBILITY_HEADERS)
  hpx_add_config_define_namespace(
    DEFINE HPX_PREPROCESSOR_HAVE_COMPATIBILITY_HEADERS
    NAMESPACE PREPROCESSOR)
endif()

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/preprocessor.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/preprocessor/cat.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/preprocessor/config.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/preprocessor/expand.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/preprocessor/nargs.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/preprocessor/stringize.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx/preprocessor/strip_parens.hpp
)

if(HPX_PREPROCESSOR_WITH_COMPATIBILITY_HEADERS)
  set(compat_headers
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/detail/pp/cat.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/detail/pp/config.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/detail/pp/expand.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/detail/pp/nargs.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/detail/pp/stringize.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx/util/detail/pp/strip_parens.hpp
  )
endif()

add_library(hpx_preprocessor STATIC ${headers} ${compat_headers})

target_include_directories(hpx_preprocessor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(HPX_PREPROCESSOR_WITH_COMPATIBILITY_HEADERS)
  target_include_directories(hpx_preprocessor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility>)
endif()

target_compile_definitions(hpx_preprocessor PRIVATE
  $<$<CONFIG:Debug>:DEBUG>
  $<$<CONFIG:Debug>:_DEBUG>
  HPX_MODULE_EXPORTS
)

include(HPX_AddSourceGroup)
add_hpx_source_group(
  NAME hpx_preprocessor
  ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include/hpx
  CLASS "Header Files"
  TARGETS ${headers})

if(HPX_PREPROCESSOR_WITH_COMPATIBILITY_HEADERS)
  add_hpx_source_group(
    NAME hpx_preprocessor
    ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include_compatibility/hpx
    CLASS "Header Files"
    TARGETS ${compat_headers})
endif()

set_target_properties(hpx_preprocessor PROPERTIES
  FOLDER "Core/Modules")

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

if(HPX_PREPROCESSOR_WITH_COMPATIBILITY_HEADERS)
  install(
    DIRECTORY include_compatibility/hpx
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT preprocessor)
endif()

add_subdirectory(src)
add_subdirectory(tests)

write_config_defines_file(
  NAMESPACE PREPROCESSOR
  FILENAME "${CMAKE_BINARY_DIR}/hpx/preprocessor/config/defines.hpp")

write_config_defines_file(
  NAMESPACE PREPROCESSOR
  FILENAME "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/hpx/preprocessor/config/defines.hpp")

