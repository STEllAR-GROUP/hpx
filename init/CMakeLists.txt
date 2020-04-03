# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2018 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set(HPX_WITH_DYNAMIC_HPX_MAIN_DEFAULT OFF)
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux" OR APPLE)
  set(HPX_WITH_DYNAMIC_HPX_MAIN_DEFAULT ON)
endif()

hpx_option(HPX_WITH_DYNAMIC_HPX_MAIN BOOL
  "Enable dynamic overload of system ``main()`` (Linux and Apple only, default: ON)"
  ${HPX_WITH_DYNAMIC_HPX_MAIN_DEFAULT} ADVANCED)
if(HPX_WITH_DYNAMIC_HPX_MAIN)
  if(NOT HPX_WITH_DYNAMIC_HPX_MAIN_DEFAULT)
    hpx_error("HPX_WITH_DYNAMIC_HPX_MAIN was set to ON, but the option is only available on Linux and Apple (this is \"${CMAKE_SYSTEM_NAME}\").")
  endif()
  hpx_add_config_define_namespace(DEFINE HPX_HAVE_DYNAMIC_HPX_MAIN NAMESPACE init)
endif()

set(config_header
  "${CMAKE_CURRENT_BINARY_DIR}/include/hpx/init/config/defines.hpp")
write_config_defines_file(NAMESPACE init FILENAME ${config_header})

set(hpx_init_HEADERS "" CACHE INTERNAL "Headers for libhpx_init." FORCE)
list(APPEND hpx_init_HEADERS
  hpx/hpx_main.hpp
  hpx/hpx_main_impl.hpp)

set(hpx_init_SOURCES "" CACHE INTERNAL "Sources for libhpx_init." FORCE)
list(APPEND hpx_init_SOURCES
  hpx_main.cpp
  hpx_main_argc_argv.cpp
  hpx_main_variables_map.cpp
  hpx_main_winsocket.cpp
  hpx_user_main.cpp
  hpx_user_main_argc_argv.cpp
  main.cpp)

if(HPX_WITH_DYNAMIC_HPX_MAIN AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
  list(APPEND hpx_init_SOURCES hpx_wrap.cpp)
endif()

# make source groups
set(hpx_init_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/src)
list(TRANSFORM hpx_init_SOURCES PREPEND ${hpx_init_SOURCE_ROOT}/)
add_hpx_source_group(
  NAME hpx_init
  CLASS "Source Files"
  ROOT "${hpx_init_SOURCE_ROOT}"
  TARGETS ${hpx_init_SOURCES})

set(hpx_init_HEADER_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/include)
list(TRANSFORM hpx_init_HEADERS PREPEND ${hpx_init_HEADER_ROOT}/)
add_hpx_source_group(
  NAME hpx_init
  CLASS "Header Files"
  ROOT "${hpx_init_HEADER_ROOT}"
  TARGETS ${hpx_init_HEADERS})
add_hpx_source_group(
  NAME hpx_init
  CLASS "Header Files"
  ROOT "${CMAKE_CURRENT_BINARY_DIR}/include/hpx"
  TARGETS ${config_header})

add_library(hpx_init STATIC
  ${hpx_init_SOURCES} ${hpx_init_HEADERS})
target_include_directories(hpx_init
  PUBLIC $<BUILD_INTERFACE:${hpx_init_HEADER_ROOT}>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
         $<INSTALL_INTERFACE:include>)
set_target_properties(hpx_init PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(hpx_init PUBLIC hpx)
target_link_libraries(hpx_init PRIVATE hpx_private_flags)
target_link_libraries(hpx_init PRIVATE hpx_dependencies_boost)
target_compile_definitions(hpx_init PRIVATE HPX_APPLICATION_EXPORTS)

set_property(TARGET hpx_init PROPERTY FOLDER "Core")

if(MSVC)
  set_target_properties(hpx_init PROPERTIES
    COMPILE_PDB_NAME_DEBUG hpx_initd
    COMPILE_PDB_NAME_RELWITHDEBINFO hpx_init
    COMPILE_PDB_OUTPUT_DIRECTORY_DEBUG
      ${CMAKE_CURRENT_BINARY_DIR}/Debug
    COMPILE_PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO
      ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo)
endif()

if(HPX_WITH_DYNAMIC_HPX_MAIN AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
  if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
    target_link_libraries(hpx_init INTERFACE "-Wl,-wrap=main")
  elseif(APPLE)
    target_link_libraries(hpx_init INTERFACE "-Wl,-e,_initialize_main")
  else()
    hpx_error("Dynamic hpx_main is not supported on ${CMAKE_SYSTEM_NAME}.")
  endif()
endif()

install(
  TARGETS hpx_init
  EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT runtime
)

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT runtime
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/hpx
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

hpx_export_targets(hpx_init)

add_hpx_pseudo_dependencies(core hpx_init)
