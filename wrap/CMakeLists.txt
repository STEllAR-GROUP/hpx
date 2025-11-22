# Copyright (c) 2007-2025 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2018 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_CXXModules)

set(hpx_wrap_HEADERS
    ""
    CACHE INTERNAL "Headers for libhpx_wrap." FORCE
)
list(APPEND hpx_wrap_HEADERS hpx/hpx_main.hpp hpx/wrap_main.hpp)

set(hpx_wrap_SOURCES
    ""
    CACHE INTERNAL "Sources for libhpx_wrap." FORCE
)
list(APPEND hpx_wrap_SOURCES hpx_wrap.cpp hpx_main.cpp)

set(hpx_auto_wrap_SOURCES
    ""
    CACHE INTERNAL "Sources for libhpx_auto_wrap." FORCE
)
list(APPEND hpx_auto_wrap_SOURCES hpx_auto_wrap.cpp hpx_main.cpp)

# make source groups
set(hpx_wrap_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
list(TRANSFORM hpx_wrap_SOURCES PREPEND ${hpx_wrap_SOURCE_ROOT}/)
list(TRANSFORM hpx_auto_wrap_SOURCES PREPEND ${hpx_wrap_SOURCE_ROOT}/)
add_hpx_source_group(
  NAME hpx_wrap
  CLASS "Source Files"
  ROOT "${hpx_wrap_SOURCE_ROOT}"
  TARGETS ${hpx_wrap_SOURCES}
)
add_hpx_source_group(
  NAME hpx_auto_wrap
  CLASS "Source Files"
  ROOT "${hpx_wrap_SOURCE_ROOT}"
  TARGETS ${hpx_auto_wrap_SOURCES}
)

set(hpx_wrap_HEADER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
list(TRANSFORM hpx_wrap_HEADERS PREPEND ${hpx_wrap_HEADER_ROOT}/)
add_hpx_source_group(
  NAME hpx_wrap
  CLASS "Header Files"
  ROOT "${hpx_wrap_HEADER_ROOT}"
  TARGETS ${hpx_wrap_HEADERS}
)
add_hpx_source_group(
  NAME hpx_auto_wrap
  CLASS "Header Files"
  ROOT "${hpx_wrap_HEADER_ROOT}"
  TARGETS ${hpx_wrap_HEADERS}
)

add_library(
  hpx_wrap STATIC ${hpx_wrap_SOURCES} ${hpx_wrap_HEADERS} ${config_header}
)
add_library(
  hpx_auto_wrap STATIC ${hpx_auto_wrap_SOURCES} ${hpx_wrap_HEADERS} ${config_header}
)

set_target_properties(hpx_wrap PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hpx_wrap PRIVATE hpx_full)
target_link_libraries(hpx_wrap PRIVATE hpx_init)
target_link_libraries(hpx_wrap PRIVATE hpx_private_flags)
target_compile_definitions(hpx_wrap PRIVATE HPX_LIBRARY_EXPORTS)
target_include_directories(
  hpx_wrap PUBLIC $<BUILD_INTERFACE:${hpx_wrap_HEADER_ROOT}>
                  $<INSTALL_INTERFACE:include>
)

set_target_properties(hpx_auto_wrap PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hpx_auto_wrap PRIVATE hpx_full)
target_link_libraries(hpx_auto_wrap PRIVATE hpx_init)
target_link_libraries(hpx_auto_wrap PRIVATE hpx_private_flags)
target_compile_definitions(hpx_auto_wrap PRIVATE HPX_LIBRARY_EXPORTS)
target_include_directories(
  hpx_auto_wrap PUBLIC $<BUILD_INTERFACE:${hpx_wrap_HEADER_ROOT}>
                  $<INSTALL_INTERFACE:include>
)

if(HPX_WITH_CXX_MODULES)
  hpx_configure_module_consumer(hpx_wrap hpx_core_module_if)
  hpx_configure_module_consumer(hpx_auto_wrap hpx_core_module_if)
endif()

set_property(TARGET hpx_wrap PROPERTY FOLDER "Core")
set_property(TARGET hpx_auto_wrap PROPERTY FOLDER "Core")

if(MSVC)
  set_target_properties(
    hpx_wrap
    PROPERTIES COMPILE_PDB_NAME_DEBUG hpx_wrapd
               COMPILE_PDB_NAME_RELWITHDEBINFO hpx_wrap
               COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG
               ${CMAKE_CURRENT_BINARY_DIR}/Debug
               COMPILE_PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO
               ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo
  )
  set_target_properties(
    hpx_auto_wrap
    PROPERTIES COMPILE_PDB_NAME_DEBUG hpx_auto_wrapd
               COMPILE_PDB_NAME_RELWITHDEBINFO hpx_auto_wrap
               COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG
               ${CMAKE_CURRENT_BINARY_DIR}/Debug
               COMPILE_PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO
               ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo
  )
endif()

if(HPX_WITH_DYNAMIC_HPX_MAIN)
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    target_link_libraries(hpx_wrap INTERFACE "-Wl,-wrap=main")
    target_link_libraries(hpx_auto_wrap INTERFACE "-Wl,-wrap=main")
  elseif(APPLE)
    target_link_libraries(hpx_wrap INTERFACE "-Wl,-e,_initialize_main")
    target_link_libraries(hpx_auto_wrap INTERFACE "-Wl,-e,_initialize_main")
  else()
    hpx_error("Dynamic hpx_main is not supported on ${CMAKE_SYSTEM_NAME}.")
  endif()
endif()

install(
  TARGETS hpx_wrap
  EXPORT HPXInternalTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT hpx_init
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT hpx_init
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT hpx_init
)

install(
  TARGETS hpx_auto_wrap
  EXPORT HPXInternalTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT hpx_init
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT hpx_init
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT hpx_init
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT core
)

# install PDB if needed
if(MSVC)
  foreach(cfg DEBUG;RELWITHDEBINFO)
    get_target_property(_pdb_file hpx_wrap COMPILE_PDB_NAME_${cfg})
    get_target_property(_pdb_dir hpx_wrap COMPILE_PDB_OUTPUT_DIRECTORY_${cfg})
    install(
      FILES ${_pdb_dir}/${_pdb_file}.pdb
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      CONFIGURATIONS ${cfg}
      COMPONENT runtime
      OPTIONAL
    )

    get_target_property(_pdb_file_auto hpx_auto_wrap COMPILE_PDB_NAME_${cfg})
    get_target_property(_pdb_dir_auto hpx_auto_wrap COMPILE_PDB_OUTPUT_DIRECTORY_${cfg})
    install(
      FILES ${_pdb_dir_auto}/${_pdb_file_auto}.pdb
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      CONFIGURATIONS ${cfg}
      COMPONENT runtime
      OPTIONAL
    )
  endforeach()
endif()

hpx_export_internal_targets(hpx_wrap)
hpx_export_internal_targets(hpx_auto_wrap)

add_hpx_pseudo_dependencies(core hpx_wrap)
add_hpx_pseudo_dependencies(core hpx_auto_wrap)

if(HPX_WITH_PRECOMPILED_HEADERS)
  target_precompile_headers(hpx_wrap REUSE_FROM hpx_precompiled_headers)
  target_precompile_headers(hpx_auto_wrap REUSE_FROM hpx_precompiled_headers)
endif()
