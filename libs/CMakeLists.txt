# Copyright (c) 2007-2021 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2018 Nikunj Gupta
# Copyright (c) 2020 The STE||AR-Group
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

include(HPX_Message)
include(HPX_AddPseudoDependencies)
include(HPX_AddPseudoTarget)

# ##############################################################################
# gather headers and sources

set(hpx_HEADERS)
set(hpx_SOURCES "${PROJECT_SOURCE_DIR}/libs/src/empty.cpp")
set(hpx_external_objects_SOURCES)

if(MSVC)
  # add natvis files to solution (supported starting VS2015)
  if(MSVC14)
    add_hpx_library_sources(
      hpx_natvis_files GLOB GLOBS "${PROJECT_SOURCE_DIR}/tools/VS/*.natvis"
    )
    list(APPEND hpx_external_OBJECTS ${hpx_natvis_files_SOURCES})
    source_group("Natvis Files" FILES ${hpx_natvis_files_SOURCES})
  endif()
endif()

set(hpx_external_SOURCES)
if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
  list(APPEND hpx_external_SOURCES ${hpx_external_objects_SOURCES}
       "${ANDROID_NDK_ROOT}/sources/android/cpufeatures/cpu-features.c"
  )
endif()

foreach(lib "hpx" "hpx_external" "hpx_generated")
  set(${lib}_SOURCES
      ${${lib}_SOURCES}
      CACHE INTERNAL "Sources for lib${lib}." FORCE
  )
  set(${lib}_HEADERS
      ${${lib}_HEADERS}
      CACHE INTERNAL "Headers for lib${lib}." FORCE
  )
endforeach()

# ##############################################################################
# make source groups
add_hpx_source_group(
  NAME hpx
  CLASS "Source Files"
  ROOT "${PROJECT_SOURCE_DIR}/libs/src"
  TARGETS ${hpx_SOURCES}
)

add_hpx_source_group(
  NAME hpx_generated
  CLASS "Source Files"
  ROOT "${PROJECT_BINARY_DIR}/libs"
)

add_hpx_source_group(
  NAME hpx
  CLASS "External Source Files"
  ROOT "${PROJECT_SOURCE_DIR}"
  TARGETS ${hpx_external_SOURCES}
)

if(NOT HPX_WITH_STATIC_LINKING)
  if(MSVC)
    # The MSVC linker can't handle a static library as large as we get when
    # statically linking the main HPX library
    set(hpx_library_link_mode_core SHARED)
  else()
    set(hpx_library_link_mode_core ${hpx_library_link_mode})
  endif()
endif()

# ##############################################################################
# libhpx
add_library(
  hpx_full ${hpx_library_link_mode_core} ${hpx_SOURCES} ${hpx_external_SOURCES}
           ${hpx_external_OBJECTS} ${hpx_HEADERS}
)

if(HPX_WITH_PRECOMPILED_HEADERS)
  target_precompile_headers(hpx_full REUSE_FROM hpx_precompiled_headers)
endif()

target_link_libraries(
  hpx_full
  PUBLIC hpx_public_flags
  PRIVATE hpx_private_flags
)
target_link_libraries(hpx_full PUBLIC hpx_core)
target_link_libraries(hpx_full PUBLIC hpx_dependencies_boost)

# Set the basic search paths for the HPX headers
target_include_directories(
  hpx_full
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}> $<INSTALL_INTERFACE:include>
)

target_link_libraries(hpx_full PUBLIC hpx_base_libraries)

if((NOT HPX_WITH_STATIC_LINKING) AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
                                      OR (APPLE))
)
  set_target_properties(hpx_full PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_compile_definitions(hpx_full PRIVATE HPX_COMPONENT_NAME=hpx HPX_EXPORTS)

# ##############################################################################
# libhpx
if(TARGET APEX::apex)
  # APEX won't get explicitly pulled into libhpx.so any more.  HOWEVER, we do
  # want to add the APEX link commands to all executables, so we use the
  # "INTERFACE" option for target_link_libraries. Because libhpx_apex is a
  # shared object library, we don't need to specify the whole archive.
  target_link_libraries(hpx_full INTERFACE APEX::apex)
endif()

if(TARGET Gperftools::gperftools)
  target_link_libraries(hpx_full PRIVATE Gperftools::gperftools)
endif()

if(TARGET Valgrind::valgrind)
  target_link_libraries(hpx_full PRIVATE Valgrind::valgrind)
endif()

if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
  set_target_properties(
    hpx_full
    PROPERTIES CLEAN_DIRECT_OUTPUT 1
               OUTPUT_NAME hpx
               FOLDER "Core"
  )
else()
  set_target_properties(
    hpx_full
    PROPERTIES VERSION ${HPX_VERSION}
               SOVERSION ${HPX_SOVERSION}
               CLEAN_DIRECT_OUTPUT 1
               OUTPUT_NAME hpx
               FOLDER "Core"
  )
endif()

# ##############################################################################
# Exported targets
# ##############################################################################
set(_library_types
    "STATIC_LIBRARY;MODULE_LIBRARY;SHARED_LIBRARY;OBJECT_LIBRARY;INTERFACE_LIBRARY"
)
set(_is_executable "$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>")
set(_is_library "$<IN_LIST:$<TARGET_PROPERTY:TYPE>,${_library_types}>")

add_library(hpx INTERFACE)
add_library(wrap_main INTERFACE)

target_link_libraries(hpx INTERFACE hpx_full)

# hpx_interface and hpx_interface_wrap_main contain additional interface options
# to be passed to dependent targets. We create these as separate targets to
# easily filter out the generator expressions that can't be handled by the
# pkgconfig file generation.
add_library(hpx_interface INTERFACE)
target_link_libraries(
  hpx_interface INTERFACE $<${_is_executable}:HPXInternal::hpx_init>
)
target_compile_definitions(
  hpx_interface
  INTERFACE
    "$<${_is_executable}:HPX_APPLICATION_NAME_DEFAULT=$<TARGET_PROPERTY:NAME>>"
)
target_compile_definitions(
  hpx_interface
  INTERFACE "$<${_is_executable}:HPX_PREFIX_DEFAULT=\"${HPX_PREFIX}\">"
)
target_compile_definitions(
  hpx_interface INTERFACE "$<${_is_executable}:HPX_APPLICATION_EXPORTS>"
)
target_compile_definitions(
  hpx_interface INTERFACE "$<${_is_library}:HPX_LIBRARY_EXPORTS>"
)

add_library(hpx_interface_wrap_main INTERFACE)
target_link_libraries(
  hpx_interface_wrap_main INTERFACE $<${_is_executable}:HPXInternal::hpx_wrap>
)

target_link_libraries(wrap_main INTERFACE hpx_interface_wrap_main)
target_link_libraries(hpx INTERFACE hpx_interface)

# HPX::component is to be linked privately to all HPX components NOTE: The
# _is_library guard only prevents simple mistakes of linking HPX::component to
# executables. It does not prevent linking it to libraries that are not
# components.
add_library(component INTERFACE)
target_compile_definitions(
  component
  INTERFACE
    "$<${_is_library}:HPX_COMPONENT_NAME_DEFAULT=hpx_$<TARGET_PROPERTY:NAME>>"
)
target_compile_definitions(
  component INTERFACE "$<${_is_library}:HPX_COMPONENT_EXPORTS>"
)

# HPX::plugin is to be linked privately to all HPX plugins NOTE: The _is_library
# guard only prevents simple mistakes of linking HPX::component to executables.
# It does not prevent linking it to libraries that are not components.
add_library(plugin INTERFACE)
target_compile_definitions(
  plugin
  INTERFACE
    "$<${_is_library}:HPX_PLUGIN_NAME_DEFAULT=hpx_$<TARGET_PROPERTY:NAME>>"
)

set(hpx_targets hpx wrap_main plugin component)
set(hpx_internal_targets hpx_full hpx_interface hpx_interface_wrap_main)

install(
  TARGETS ${hpx_targets}
  EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          COMPONENT runtime
          ${_optional}
)

install(
  TARGETS ${hpx_internal_targets}
  EXPORT HPXInternalTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
          COMPONENT runtime
          ${_optional}
)

# install PDB if needed
if(MSVC)
  install(
    FILES $<TARGET_PDB_FILE:hpx_full>
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    CONFIGURATIONS Debug RelWithDebInfo
    OPTIONAL
  )
endif()

hpx_export_targets(${hpx_targets})
hpx_export_internal_targets(${hpx_internal_targets})

foreach(target ${hpx_targets})
  add_hpx_pseudo_dependencies(core ${target})
endforeach()

# add example pseudo targets needed for modules
if(HPX_WITH_EXAMPLES)
  add_hpx_pseudo_target(examples.modules)
  add_hpx_pseudo_dependencies(examples examples.modules)
endif()

# add test pseudo targets needed for modules
if(HPX_WITH_TESTS)
  if(HPX_WITH_TESTS_UNIT)
    add_hpx_pseudo_target(tests.unit.modules)
    add_hpx_pseudo_dependencies(tests.unit tests.unit.modules)
  endif()

  if(HPX_WITH_EXAMPLES AND HPX_WITH_TESTS_EXAMPLES)
    add_hpx_pseudo_target(tests.examples.modules)
    add_hpx_pseudo_dependencies(tests.examples tests.examples.modules)
  endif()

  if(HPX_WITH_TESTS_REGRESSIONS)
    add_hpx_pseudo_target(tests.regressions.modules)
    add_hpx_pseudo_dependencies(tests.regressions tests.regressions.modules)
  endif()

  if(HPX_WITH_TESTS_BENCHMARKS)
    add_hpx_pseudo_target(tests.performance.modules)
    add_hpx_pseudo_dependencies(tests.performance tests.performance.modules)
  endif()

  if(HPX_WITH_TESTS_HEADERS)
    add_hpx_pseudo_target(tests.headers.modules)
    add_hpx_pseudo_dependencies(tests.headers tests.headers.modules)
  endif()
endif()

hpx_info("")
hpx_info("Configuring modules:")

# Generate a file that lists all enabled modules for checks that might be
# optional based on the presence of a module or not
set(MODULE_ENABLED_MODULE_DEFINES)

# variables needed for config_strings_modules.hpp
set(CONFIG_STRINGS_MODULES_INCLUDES)
set(CONFIG_STRINGS_MODULES_ENTRIES)

set(HPX_LIBS
    core full
    CACHE INTERNAL "List of HPX libraries" FORCE
)

foreach(lib ${HPX_LIBS})
  # NOTE: hpx_full (i.e. the target representing libhpx.so) is still created in
  # src/CMakeLists.txt. When all of hpx_full has been modularized the target
  # creation can move here as well.
  if(NOT ${lib} STREQUAL "full")
    string(TOUPPER ${lib} uppercase_lib)
    add_library(hpx_${lib} SHARED src/empty.cpp)
    target_compile_definitions(hpx_${lib} PRIVATE HPX_${uppercase_lib}_EXPORTS)

    set_target_properties(hpx_${lib} PROPERTIES FOLDER "Core")

    install(
      TARGETS hpx_${lib}
      EXPORT HPXInternalTargets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
      RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT ${lib}
    )

    hpx_export_internal_targets(hpx_${lib})
  endif()

  add_subdirectory(${lib})
endforeach()

# add_hpx_module populates HPX_ENABLED_MODULES
foreach(module ${HPX_ENABLED_MODULES})
  string(TOUPPER ${module} uppercase_module)

  set(MODULE_ENABLED_MODULE_DEFINES
      "${MODULE_ENABLED_MODULE_DEFINES}#define HPX_HAVE_MODULE_${uppercase_module}\n"
  )
endforeach()

configure_file(
  "${PROJECT_SOURCE_DIR}/cmake/templates/modules_enabled.hpp.in"
  "${PROJECT_BINARY_DIR}/libs/core/config/include/hpx/config/modules_enabled.hpp"
  @ONLY
)

if(MSVC AND HPX_COROUTINES_WITH_SWAP_CONTEXT_EMULATION)
  target_link_options(hpx_core PRIVATE "/EXPORT:switch_to_fiber")
endif()

if(HPX_WITH_DATAPAR_VC)
  target_link_libraries(hpx_core PUBLIC Vc::vc)
endif()

if(HPX_WITH_DATAPAR_EVE)
  target_link_libraries(hpx_core PUBLIC Eve::eve)
endif()

if(HPX_WITH_ITTNOTIFY)
  target_link_libraries(hpx_core PUBLIC Amplifier::amplifier)
endif()

if(HPX_WITH_MODULES_AS_STATIC_LIBRARIES)
  target_link_libraries(hpx_core PUBLIC hpx_public_flags)
  target_link_libraries(hpx_core PUBLIC hpx_base_libraries)
  target_link_libraries(hpx_core PUBLIC hpx_dependencies_boost)
  target_link_libraries(hpx_core PUBLIC hpx_dependencies_allocator)
  target_link_libraries(hpx_core PUBLIC Hwloc::hwloc)
  target_link_libraries(hpx_core PUBLIC Asio::asio)

  if(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY)
    target_link_libraries(hpx_core PUBLIC Boost::filesystem)
  endif()
endif()
