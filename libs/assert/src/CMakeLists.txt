# Copyright (c) 2018 The STE||AR-Group
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(headers
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/assert.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/assert/current_function.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/assert/evaluate_assert.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx/assert/source_location.hpp
)

set(sources
  ${CMAKE_CURRENT_SOURCE_DIR}/assert.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/source_location.cpp
)

add_library(hpx.assert ${hpx_libs_link_mode} ${sources})

target_link_libraries(hpx.assert hpx.pp hpx.config)
target_include_directories(hpx.assert PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>)

include(HPX_AddSourceGroup)
add_hpx_source_group(NAME hpx ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx
  CLASS "Header Files" TARGETS ${headers})
add_hpx_source_group(NAME hpx ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx
  CLASS "Source Files" TARGETS ${sources})

install(TARGETS hpx.assert EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT assert
)
hpx_export_targets(hpx.assert)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/hpx
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hpx
  COMPONENT assert)
