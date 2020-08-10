# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2018 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(hpx_init_HEADERS
    ""
    CACHE INTERNAL "Headers for libhpx_init." FORCE
)
list(APPEND hpx_init_HEADERS hpx/hpx_main_impl.hpp)
set(hpx_init_SOURCES
    ""
    CACHE INTERNAL "Sources for libhpx_init." FORCE
)
list(
  APPEND
  hpx_init_SOURCES
  hpx_main.cpp
  hpx_main_argc_argv.cpp
  hpx_main_variables_map.cpp
  hpx_user_main.cpp
  hpx_user_main_argc_argv.cpp
  main.cpp
)

# make source groups
set(hpx_init_HEADER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
list(TRANSFORM hpx_init_HEADERS PREPEND ${hpx_init_HEADER_ROOT}/)
add_hpx_source_group(
  NAME hpx_init
  CLASS "Header Files"
  ROOT "${hpx_init_HEADER_ROOT}"
  TARGETS ${hpx_init_HEADERS}
)

set(hpx_init_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
list(TRANSFORM hpx_init_SOURCES PREPEND ${hpx_init_SOURCE_ROOT}/)
add_hpx_source_group(
  NAME hpx_init
  CLASS "Source Files"
  ROOT "${hpx_init_SOURCE_ROOT}"
  TARGETS ${hpx_init_SOURCES}
)

add_library(hpx_init STATIC ${hpx_init_SOURCES} ${hpx_init_HEADERS})
set_target_properties(hpx_init PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hpx_init PRIVATE hpx_core)
target_link_libraries(hpx_init PRIVATE hpx_private_flags)
target_compile_definitions(hpx_init PRIVATE HPX_APPLICATION_EXPORTS)
target_include_directories(
  hpx_init PUBLIC $<BUILD_INTERFACE:${hpx_init_HEADER_ROOT}>
                  $<INSTALL_INTERFACE:include>
)

set_property(TARGET hpx_init PROPERTY FOLDER "Core")

if(MSVC)
  set_target_properties(
    hpx_init
    PROPERTIES COMPILE_PDB_NAME_DEBUG hpx_initd
               COMPILE_PDB_NAME_RELWITHDEBINFO hpx_init
               COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG
               ${CMAKE_CURRENT_BINARY_DIR}/Debug
               COMPILE_PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO
               ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo
  )
endif()

if(HPX_WITH_STACKTRACES AND HPX_WITH_STACKTRACES_STATIC_SYMBOLS)
  # Add -Wl,-E to linker for executables on Linux(Clang or gcc) this will add
  # all symbols to the dynamic symbol table
  if(("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
     AND ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang"
          OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  )
    target_link_libraries(hpx_init INTERFACE "-Wl,-E")
  endif()
endif()

install(
  TARGETS hpx_init
  EXPORT HPXInternalTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT runtime
)

# install PDB if needed
if(MSVC)
  foreach(cfg DEBUG;RELWITHDEBINFO)
    get_target_property(_pdb_file hpx_init COMPILE_PDB_NAME_${cfg})
    get_target_property(_pdb_dir hpx_init COMPILE_PDB_OUTPUT_DIRECTORY_${cfg})
    install(
      FILES ${_pdb_dir}/${_pdb_file}.pdb
      DESTINATION ${CMAKE_INSTALL_LIBDIR}
      CONFIGURATIONS ${cfg}
      OPTIONAL
    )
  endforeach()
endif()

hpx_export_internal_targets(hpx_init)

add_hpx_pseudo_dependencies(core hpx_init)
