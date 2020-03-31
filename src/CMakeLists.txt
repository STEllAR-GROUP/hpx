# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2018 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

foreach(lib "hpx" "hpx_external" "hpx_generated")
  set(${lib}_SOURCES "" CACHE INTERNAL "Sources for lib${lib}." FORCE)

  if(MSVC)
    set(${lib}_HEADERS "" CACHE INTERNAL "Headers for lib${lib}." FORCE)
  endif()
endforeach()

################################################################################
# gather sources

# libhpx sources
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/*.cpp"
  APPEND)

add_hpx_library_sources(hpx_generated
  GLOB GLOBS "${PROJECT_BINARY_DIR}/libs/modules.cpp"
  APPEND)

add_hpx_library_sources(hpx_external GLOB APPEND)

if(MSVC)
  if(HPX_WITH_SWAP_CONTEXT_EMULATION)
    enable_language(ASM)
    set(switch_to_fiber_source
      "${PROJECT_SOURCE_DIR}/libs/coroutines/src/switch_to_fiber.asm")
    add_hpx_library_sources(hpx_external_objects
      GLOB GLOBS "${switch_to_fiber_source}")
    add_custom_command(
      OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj"
      COMMAND "${CMAKE_ASM_MASM_COMPILER}" /Fo "${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj" /nologo /c "${switch_to_fiber_source}"
      DEPENDS "${switch_to_fiber_source}"
      VERBATIM)
    set(hpx_external_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/switch_to_fiber.obj"
      CACHE INTERNAL "External object files for HPX." FORCE)
  endif()

  # add natvis files to solution (supported starting VS2015)
  if(MSVC14)
    add_hpx_library_sources(hpx_natvis_files
      GLOB GLOBS "${PROJECT_SOURCE_DIR}/tools/VS/*.natvis")
    set(hpx_external_OBJECTS ${hpx_external_OBJECTS} ${hpx_natvis_files_SOURCES})
    source_group("Natvis Files" FILES ${hpx_natvis_files_SOURCES})
  endif()

endif()

if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
  add_hpx_library_sources(hpx_external GLOB GLOBS "${ANDROID_NDK_ROOT}/sources/android/cpufeatures/cpu-features.c")
endif()

################################################################################
# gather headers

# libhpx headers
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/*.hpp"
  APPEND)

add_hpx_library_headers(hpx_external GLOB APPEND)

################################################################################
# make source groups
add_hpx_source_group(
  NAME hpx CLASS "Source Files"
  ROOT "${PROJECT_SOURCE_DIR}/src"
  TARGETS ${hpx_SOURCES})

add_hpx_source_group(
  NAME hpx_generated CLASS "Source Files"
  ROOT "${PROJECT_BINARY_DIR}/libs"
  TARGETS ${hpx_generated_SOURCES})

add_hpx_source_group(
  NAME hpx CLASS "External Source Files"
  ROOT "${PROJECT_SOURCE_DIR}"
  TARGETS ${hpx_external_SOURCES})

add_hpx_source_group(
  NAME hpx
  CLASS "Header Files"
  ROOT "${PROJECT_SOURCE_DIR}/hpx"
  TARGETS ${hpx_HEADERS})

if(NOT HPX_WITH_STATIC_LINKING)
  if(MSVC)
    # The MSVC linker can't handle a static library as large as we get when
    # statically linking the main HPX library
    set(hpx_library_link_mode_core SHARED)
  else()
    set(hpx_library_link_mode_core ${hpx_library_link_mode})
  endif()
endif()

################################################################################
# libhpx
add_library(hpx ${hpx_library_link_mode_core}
  ${hpx_SOURCES} ${hpx_external_SOURCES} ${hpx_generated_SOURCES}
  ${hpx_external_OBJECTS} ${hpx_HEADERS})

target_link_libraries(hpx PUBLIC hpx_public_flags PRIVATE hpx_private_flags)
target_link_libraries(hpx PUBLIC hpx_dependencies_boost)

# Set the basic search paths for the HPX headers
target_include_directories(hpx PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  )

# FIXME : temporary add public dependencies to the boost libraries which are
# only linked in the modules (which are themselves privately linked to hpx), it
# should be removed when we make the modules object libraries that we publicly
# link to hpx
if(HPX_PROGRAM_OPTIONS_WITH_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
  target_link_libraries(hpx PUBLIC Boost::program_options)
endif()
if(HPX_FILESYSTEM_WITH_BOOST_FILESYSTEM_COMPATIBILITY)
  target_link_libraries(hpx PUBLIC Boost::filesystem)
endif()

target_link_libraries(hpx PUBLIC hpx_dependencies_allocator)

# Hwloc PUBLIC because used by exported target hpx_topology
target_link_libraries(hpx PUBLIC Hwloc::hwloc)

# Enable SIMD via VC if it is available
if (TARGET Vc::vc)
  target_link_libraries(hpx PUBLIC Vc::vc)
endif()

if (TARGET Amplifier::amplifier)
  target_link_libraries(hpx PRIVATE Amplifier::amplifier)
endif()

if (TARGET APEX::apex)
  # APEX won't get explicitly pulled into libhpx.so any more.  HOWEVER,
  # we do want to add the APEX link commands to all executables, so
  # we use the "INTERFACE" option for target_link_libraries.
  # Because libhpx_apex is a shared object library, we don't need to specify
  # the whole archive.
  target_link_libraries(hpx INTERFACE APEX::apex)
endif()

if (TARGET Gperftools::gperftools)
  target_link_libraries(hpx PRIVATE Gperftools::gperftools)
endif()

if (TARGET Valgrind::valgrind)
  target_link_libraries(hpx PRIVATE Valgrind::valgrind)
endif()

if("${HPX_PLATFORM_UC}" STREQUAL "ANDROID")
  set_target_properties(hpx PROPERTIES
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME hpx
    FOLDER "Core")
else()
  set_target_properties(hpx PROPERTIES
    VERSION ${HPX_VERSION}
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME hpx
    FOLDER "Core")
endif()

target_link_libraries(hpx PUBLIC hpx_base_libraries)

# Link to each parcelport plugin
foreach(_parcelport_plugin ${HPX_STATIC_PARCELPORT_PLUGINS})
  target_link_libraries(hpx PRIVATE ${_parcelport_plugin})
endforeach()

# integrate the hpx modules with the main library
foreach(_module ${HPX_LIBS})

  set(_module_target hpx_${_module})
  if(UNIX)
    # add the full archive (some are not used directly in libhpx and need to be
    # forced into libhpx with this option)
    if(APPLE)
      set(_module_target "-Wl,-all_load" "${_module_target}")
    else(APPLE) # not apple, regular linux
      set(_module_target "-Wl,--whole-archive" "${_module_target}" "-Wl,--no-whole-archive")
    endif(APPLE)
  endif(UNIX)

  # add module binaries as PRIVATE dependencies to the core hpx library to avoid
  # dependent applications have to link against those
  target_link_libraries(hpx PRIVATE ${_module_target})

  # add module include directories as PUBLIC to core hpx library to enable
  # compilation against indirectly included headers
  get_target_property(_module_includes hpx_${_module} INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(hpx PUBLIC ${_module_includes})

endforeach()

if((NOT HPX_WITH_STATIC_LINKING) AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
  set_target_properties(hpx PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_compile_definitions(hpx PRIVATE HPX_COMPONENT_NAME=hpx HPX_EXPORTS)

################################################################################
# Emulation of SwapContext on Windows
################################################################################
if(MSVC AND HPX_WITH_SWAP_CONTEXT_EMULATION)
  target_link_options(hpx PRIVATE "/EXPORT:switch_to_fiber")
endif()

set(hpx_targets hpx)

set(_library_types "STATIC_LIBRARY;MODULE_LIBRARY;SHARED_LIBRARY;OBJECT_LIBRARY;INTERFACE_LIBRARY")
set(_is_executable "$<STREQUAL:$<TARGET_PROPERTY:TYPE>,EXECUTABLE>")
set(_is_library "$<IN_LIST:$<TARGET_PROPERTY:TYPE>,${_library_types}>")

# hpx_interface contains additional interface options to be passed to dependent
# targets. We create this as a separate target to easily filter out the
# generator expressions that can't be handled by the pkgconfig file generation.
add_library(hpx_interface INTERFACE)
target_link_libraries(hpx_interface INTERFACE $<${_is_executable}:$<TARGET_NAME_IF_EXISTS:hpx_init>>)
target_link_libraries(hpx_interface INTERFACE $<${_is_executable}:$<TARGET_NAME_IF_EXISTS:HPX::hpx_init>>)
target_compile_definitions(hpx_interface INTERFACE "$<${_is_executable}:HPX_APPLICATION_NAME_DEFAULT=$<TARGET_PROPERTY:NAME>>")
target_compile_definitions(hpx_interface INTERFACE "$<${_is_executable}:HPX_PREFIX_DEFAULT=\"${HPX_PREFIX}\">")
target_compile_definitions(hpx_interface INTERFACE "$<${_is_executable}:HPX_APPLICATION_EXPORTS>")
target_compile_definitions(hpx_interface INTERFACE "$<${_is_library}:HPX_LIBRARY_EXPORTS>")
target_link_libraries(hpx INTERFACE hpx_interface)

# HPX::component is to be linked privately to all HPX components
# NOTE: The _is_library guard only prevents simple mistakes of linking
# HPX::component to executables. It does not prevent linking it to libraries
# that are not components.
add_library(component INTERFACE)
target_compile_definitions(component INTERFACE "$<${_is_library}:HPX_COMPONENT_NAME_DEFAULT=hpx_$<TARGET_PROPERTY:NAME>>")
target_compile_definitions(component INTERFACE "$<${_is_library}:HPX_COMPONENT_EXPORTS>")

# HPX::plugin is to be linked privately to all HPX plugins
# NOTE: The _is_library guard only prevents simple mistakes of linking
# HPX::component to executables. It does not prevent linking it to libraries
# that are not components.
add_library(plugin INTERFACE)
target_compile_definitions(plugin INTERFACE "$<${_is_library}:HPX_PLUGIN_NAME_DEFAULT=hpx_$<TARGET_PROPERTY:NAME>>")

set(hpx_targets ${hpx_targets} hpx_interface plugin component)

install(
  TARGETS ${hpx_targets}
  EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT runtime
  ${_optional}
)

# install PDB if needed
if(MSVC)
  install(
    FILES $<TARGET_PDB_FILE:hpx>
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    CONFIGURATIONS Debug RelWithDebInfo
    OPTIONAL
  )
endif()

hpx_export_targets(${hpx_targets})

foreach(target ${hpx_targets})
  add_hpx_pseudo_dependencies(core ${target})
endforeach()

# Add particular targets for the pkgconfig setup
add_library(hpx_pkgconfig_application INTERFACE)
target_link_libraries(hpx_pkgconfig_application INTERFACE hpx_init)
target_compile_definitions(hpx_pkgconfig_application INTERFACE HPX_APPLICATION_EXPORTS)

add_library(hpx_pkgconfig_component INTERFACE)
target_compile_definitions(hpx_pkgconfig_component INTERFACE HPX_COMPONENT_EXPORTS)
target_link_libraries(hpx_pkgconfig_component INTERFACE hpx)
