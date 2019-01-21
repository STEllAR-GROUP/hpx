# Copyright (c) 2019 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/asio.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/autolink.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/compiler_fence.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/compiler_specific.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/debug.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/export_definitions.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/lambda_capture.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/version.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/warnings_suffix.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/attributes.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/branch_hints.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/compiler_native_tls.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/constexpr.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/emulate_deleted.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/forceinline.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/manual_profiling.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/threads_stack.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/warnings_prefix.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/config/weak_symbol.hpp
)

set(sources
  ${CMAKE_CURRENT_SOURCE_DIR}/asio.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/autolink.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compiler_fence.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compiler_specific.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/debug.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/export_definitions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lambda_capture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/version.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/warnings_suffix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/attributes.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/branch_hints.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/compiler_native_tls.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/constexpr.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/emulate_deleted.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/forceinline.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/manual_profiling.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/threads_stack.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/warnings_prefix.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/weak_symbol.cpp
)

add_library(hpx.config ${hpx_libs_link_mode} ${sources})

target_link_libraries(hpx.config hpx.pp)
target_include_directories(hpx.config PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)

include(HPX_AddSourceGroup)
add_hpx_source_group(NAME hpx ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx
  CLASS "Header Files" TARGETS ${headers})
add_hpx_source_group(NAME hpx ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx
  CLASS "Source Files" TARGETS ${sources})

install(TARGETS hpx.config EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT config
)
hpx_export_targets(hpx.config)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpx
  COMPONENT config)
