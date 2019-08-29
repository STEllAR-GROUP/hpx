# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2018 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

foreach(lib "hpx" "hpx_init")
  set(${lib}_SOURCES "" CACHE INTERNAL "Sources for lib${lib}." FORCE)

  if(MSVC)
    set(${lib}_HEADERS "" CACHE INTERNAL "Headers for lib${lib}." FORCE)
  endif()
endforeach()

################################################################################
# gather sources

# libhpx sources
add_hpx_library_sources(hpx
  GLOB GLOBS "${PROJECT_SOURCE_DIR}/src/*.cpp"
  EXCLUDE "(.*(hpx_main|hpx_user).*[.]cpp)|main.cpp|hpx_wrap.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB GLOBS "${PROJECT_SOURCE_DIR}/src/pre_main.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/runtime/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/performance_counters/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/util/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/util/detail/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/lcos/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/compute/*.cpp"
  APPEND)
add_hpx_library_sources(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/src/compat/*.cpp"
  APPEND)

add_hpx_library_sources(hpx_generated
  GLOB GLOBS "${CMAKE_BINARY_DIR}/libs/modules.cpp"
  APPEND)

# libhpx_wrap sources
if(HPX_WITH_DYNAMIC_HPX_MAIN AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
  add_hpx_library_sources (hpx_wrap
      GLOB GLOBS "${PROJECT_SOURCE_DIR}/src/hpx_wrap.cpp")
endif()

# libhpx_init sources
add_hpx_library_sources(hpx_init
  GLOB GLOBS "${PROJECT_SOURCE_DIR}/src/hpx_*.cpp"
  EXCLUDE "hpx_init[.]cpp|hpx_wrap.cpp")
add_hpx_library_sources(hpx_init
  GLOB GLOBS "${PROJECT_SOURCE_DIR}/src/main.cpp" APPEND)

add_hpx_library_sources(hpx_external GLOB APPEND)

if(MSVC)
  if(HPX_WITH_SWAP_CONTEXT_EMULATION)
    enable_language(ASM)
    set(switch_to_fiber_source
      "${PROJECT_SOURCE_DIR}/src/runtime/threads/coroutines/switch_to_fiber.asm")
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
  GLOB GLOBS "${PROJECT_SOURCE_DIR}/hpx/*.hpp"
  EXCLUDE ".*hpx_main.*[.]hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/runtime/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/config/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/include/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/lcos/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/performance_counters/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/traits/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/util/*.h*"
  EXCLUDE "(.*_binary_.*|[io]chunk_.*)[.]hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB GLOBS "${PROJECT_SOURCE_DIR}/hpx/plugins/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/parallel/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/compute/*.hpp"
  APPEND)
add_hpx_library_headers(hpx
  GLOB_RECURSE GLOBS "${PROJECT_SOURCE_DIR}/hpx/compat/*.hpp"
  APPEND)

# libhpx_init headers
add_hpx_library_headers(hpx_init
  GLOB GLOBS "${PROJECT_SOURCE_DIR}/hpx/hpx_main*.hpp")

add_hpx_library_headers(hpx_external GLOB APPEND)

################################################################################
# make source groups
add_hpx_source_group(
  NAME hpx CLASS "Source Files"
  ROOT "${PROJECT_SOURCE_DIR}/src"
  TARGETS ${hpx_SOURCES})

add_hpx_source_group(
  NAME hpx_generated CLASS "Source Files"
  ROOT "${CMAKE_BINARY_DIR}/libs"
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

if(HPX_WITH_DYNAMIC_HPX_MAIN AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
  add_hpx_source_group(
    NAME hpx_wrap
    CLASS "Source Files"
    ROOT "${PROECT_SOURCE_DIR}/src"
    TARGETS "${hpx_wrap_SOURCES}")
endif()

add_hpx_source_group(
  NAME hpx_init
  CLASS "Source Files"
  ROOT "${PROJECT_SOURCE_DIR}/src"
  TARGETS ${hpx_init_SOURCES})

add_hpx_source_group(
  NAME hpx_init
  CLASS "Header Files"
  ROOT "${PROJECT_SOURCE_DIR}/hpx"
  TARGETS ${hpx_init_HEADERS})

if(NOT HPX_WITH_STATIC_LINKING)
  if(MSVC)
    # The MSVC linker can't handle a static library as large as we get when
    # statically linking the main HPX library
    set(hpx_library_link_mode_core SHARED)
  else()
    set(hpx_library_link_mode_core ${hpx_library_link_mode})
  endif()
endif()

# Add static parcelport sources ...
add_static_parcelports()
set(hpx_HEADERS ${hpx_HEADERS} ${HPX_STATIC_PARCELPORT_PLUGINS_HEADERS})
set(hpx_SOURCES ${hpx_SOURCES} ${HPX_STATIC_PARCELPORT_PLUGINS_SOURCES})
#
# set_source_files_properties and set_property appear to be broken when setting
# INCLUDE_DIRECTORIES, so work around by adding compile flags for the include
# locations needed by parcelport plugin sources
#
foreach(_plugin ${HPX_STATIC_PARCELPORT_PLUGINS})
  if (_parcelport_${_plugin}_INCLUDE_DIRECTORIES)
    string (REPLACE ";" " -I" _INCLUDE_STRING "${_parcelport_${_plugin}_INCLUDE_DIRECTORIES}")
    set_property(
      SOURCE
        ${_parcelport_${_plugin}_SOURCES}
      APPEND_STRING PROPERTY COMPILE_FLAGS
        " -I${_INCLUDE_STRING}"
    )
  endif()
endforeach()

################################################################################
# libhpx
if(HPX_WITH_STATIC_LINKING)
  set(hpx_SOURCES ${hpx_SOURCES} ${hpx_init_SOURCES} ${hpx_generated_SOURCES})
endif()
if(HPX_WITH_DEFAULT_TARGETS)
  if(HPX_WITH_CUDA)
    cuda_add_library(hpx ${hpx_library_link_mode_core}
      ${hpx_SOURCES} ${hpx_external_SOURCES} ${hpx_generated_SOURCES}
      ${hpx_external_OBJECTS} ${hpx_HEADERS})
  else()
    add_library(hpx ${hpx_library_link_mode_core}
      ${hpx_SOURCES} ${hpx_external_SOURCES} ${hpx_generated_SOURCES}
      ${hpx_external_OBJECTS} ${hpx_HEADERS})
  endif()
else()
  if(HPX_WITH_CUDA)
    cuda_add_library(hpx ${hpx_library_link_mode_core} EXCLUDE_FROM_ALL
      ${hpx_SOURCES} ${hpx_external_SOURCES} ${hpx_generated_SOURCES}
      ${hpx_external_OBJECTS} ${hpx_HEADERS})
  else()
    add_library(hpx ${hpx_library_link_mode_core} EXCLUDE_FROM_ALL
      ${hpx_SOURCES} ${hpx_external_SOURCES} ${hpx_generated_SOURCES}
      ${hpx_external_OBJECTS} ${hpx_HEADERS})
  endif()
endif()

# Default unnamed config (not Debug/Release/etc) are in this var
get_property(_temp_flags GLOBAL PROPERTY HPX_CMAKE_FLAGS_CXX_)
target_compile_options(hpx PRIVATE ${_temp_flags})
target_compile_definitions(hpx PUBLIC $<$<CONFIG:Debug>:HPX_DEBUG>)

# message("Adding ${_temp_flags}")

# Could potentially use CMAKE_CONFIGURATION_TYPES in case a user defined config exists
foreach(_config "DEBUG" "RELEASE" "RELWITHDEBINFO" "MINSIZEREL")
  get_property(_temp_flags GLOBAL PROPERTY HPX_CMAKE_FLAGS_CXX_${_config})
  target_compile_options(hpx PRIVATE $<$<CONFIG:${_config}>:${_temp_flags}>)
#  message("Adding $<$<CONFIG:${_config}>:${_temp_flags}>")
endforeach()

#if(HPX_WITH_STATIC_LINKING)
#  add_dependencies(hpx static_component_data_hpp)
#endif()

target_link_libraries(hpx PUBLIC hpx::boost)
target_include_directories(hpx PUBLIC ${Boost_INCLUDE_DIRS})

# Set the basic search paths for the HPX headers
target_include_directories(hpx PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<INSTALL_INTERFACE:include>
  )

target_link_libraries(hpx PUBLIC hpx::allocator)

target_link_libraries(hpx PRIVATE hpx::hwloc)

# Enable SIMD via VC if it is available
if (TARGET hpx::vc)
  target_link_libraries(hpx PRIVATE hpx::vc)
endif()

if (TARGET hpx::amplifier)
  target_link_libraries(hpx PRIVATE hpx::amplifier)
endif()

if (TARGET hpx::apex)
  target_link_libraries(hpx PRIVATE hpx::apex)
  # System has been removed when passing at set_property for cmake < 3.11 in
  # HPX_SetupApex.cmake so the property get is not SYSTEM_INTERFACE_INC_DIR
  # anymore, we can put it back once we require cmake >= 3.11
  get_target_property(_apex_inc_dir hpx::apex INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(hpx INTERFACE "${_apex_inc_dir}")
endif()

if (TARGET hpx::gperftools)
  target_link_libraries(hpx PRIVATE hpx::gperftools)
endif()

if (TARGET hpx::valgrind)
  target_link_libraries(hpx PRIVATE hpx::valgrind)
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

target_link_libraries(hpx PUBLIC ${HPX_LIBRARIES})
if(HPX_ADDITIONAL_RUNTIME_DEPENDENCIES)
  target_link_libraries(hpx PUBLIC ${HPX_ADDITIONAL_RUNTIME_DEPENDENCIES})
endif()
if(HPX_ADDITIONAL_PRIVATE_RUNTIME_DEPENDENCIES)
  target_link_libraries(hpx PRIVATE ${HPX_ADDITIONAL_PRIVATE_RUNTIME_DEPENDENCIES})
endif()

foreach(_plugin ${HPX_STATIC_PARCELPORT_PLUGINS})
  if (_parcelport_${_plugin}_LIBRARIES)
    target_link_libraries(hpx PUBLIC ${_parcelport_${_plugin}_LIBRARIES})
    hpx_export_targets(${_parcelport_${_plugin}_LIBRARIES})
  endif()
endforeach()

# integrate the hpx modules with the main library
foreach(_module ${HPX_LIBS})
  set(_module_target hpx_${_module})
  if(UNIX OR APPLE)
    # add the full archive (some are not used directly in libhpx and need to be
    # forced into libhpx with this option)
    set(_module_target "-Wl,--whole-archive" "${_module_target}" "-Wl,--no-whole-archive")
  endif()

  # add module binaries as PRIVATE dependencies to the core hpx library to
  # avoid dependent applicatons have to link against those
  target_link_libraries(hpx PRIVATE ${_module_target})

  # add module include directories as PUBLIC to core hpx library to enable
  # compilation against indirectly included headers
  get_target_property(_module_includes hpx_${_module} INTERFACE_INCLUDE_DIRECTORIES)
  target_include_directories(hpx PUBLIC ${_module_includes})
endforeach()

target_compile_definitions(hpx PUBLIC HPX_ENABLE_ASSERT_HANDLER PRIVATE
  HPX_COMPONENT_NAME=hpx HPX_EXPORTS)

################################################################################
# Emulation of SwapContext on Windows
################################################################################
if(MSVC)
  if(HPX_WITH_SWAP_CONTEXT_EMULATION)
    set_property(TARGET hpx APPEND
      PROPERTY LINK_FLAGS
      "/EXPORT:switch_to_fiber")
  endif()
endif()

set(hpx_targets ${hpx_targets} hpx)

################################################################################
# libhpx_init

if(NOT HPX_WITH_STATIC_LINKING)
  if(HPX_WITH_DEFAULT_TARGETS)
    if(HPX_WITH_CUDA)
      cuda_add_library(hpx_init STATIC
        ${hpx_init_SOURCES} ${hpx_init_HEADERS})
    else()
      add_library(hpx_init STATIC
        ${hpx_init_SOURCES} ${hpx_init_HEADERS})
    endif()
  else()
    if(HPX_WITH_CUDA)
      cuda_add_library(hpx_init STATIC EXCLUDE_FROM_ALL
        ${hpx_init_SOURCES} ${hpx_init_HEADERS})
    else()
      add_library(hpx_init STATIC EXCLUDE_FROM_ALL
        ${hpx_init_SOURCES} ${hpx_init_HEADERS})
    endif()
  endif()
  target_include_directories(hpx_init PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>)
  set_target_properties(hpx_init PROPERTIES POSITION_INDEPENDENT_CODE ON)

  foreach(_module ${HPX_LIBS})
    # add module include directories as PRIVATE to the hpx_init library to
    # enable its compilation against indirectly included headers
    get_target_property(_module_includes hpx_${_module} INTERFACE_INCLUDE_DIRECTORIES)
    target_include_directories(hpx_init PRIVATE ${_module_includes})
  endforeach()

  # Set the basic search paths for the generated HPX headers
  target_include_directories(hpx_init
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
      $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
      $<INSTALL_INTERFACE:include>)
  target_link_libraries(hpx_init PRIVATE hpx::boost)

  target_compile_definitions(hpx_init PUBLIC HPX_ENABLE_ASSERT_HANDLER PRIVATE
    HPX_APPLICATION_EXPORTS)

  set_property(TARGET hpx_init PROPERTY FOLDER "Core")
  set(hpx_targets ${hpx_targets} hpx_init)
endif()

################################################################################
# libhpx_wrap

if(HPX_WITH_DYNAMIC_HPX_MAIN AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
  if(NOT HPX_WITH_STATIC_LINKING)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    if(HPX_WITH_DEFAULT_TARGETS)
      if(HPX_WITH_CUDA)
        cuda_add_library(hpx_wrap STATIC
          ${hpx_wrap_SOURCES})
      else()
        add_library(hpx_wrap STATIC
          ${hpx_wrap_SOURCES})
      endif()
    else()
      if(HPX_WITH_CUDA)
        cuda_add_library(hpx_wrap STATIC EXCLUDE_FROM_ALL
          ${hpx_wrap_SOURCES})
      else()
        add_library(hpx_wrap STATIC EXCLUDE_FROM_ALL
          ${hpx_wrap_SOURCES})
      endif()
    endif()

    target_compile_definitions(hpx_wrap PUBLIC HPX_ENABLE_ASSERT_HANDLER PRIVATE
      HPX_APPLICATION_EXPORTS)

    # Set the basic search paths for the generated HPX headers
    target_include_directories(hpx_wrap
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>)
    target_link_libraries(hpx_wrap PUBLIC hpx)
    if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
      set_property(TARGET hpx_wrap APPEND PROPERTY INTERFACE_LINK_OPTIONS -Wl,-wrap=main)
    elseif(APPLE)
      set_property(TARGET hpx_wrap APPEND PROPERTY INTERFACE_LINK_OPTIONS -Wl,-e,_initialize_main)
    else()
      hpx_error("hpx_main is not supported on ${CMAKE_SYSTEM_NAME}.")
    endif()

    foreach(_module ${HPX_LIBS})
      # add module include directories as PRIVATE to the hpx_wrap library to
      # enable its compilation against indirectly included headers
      get_target_property(_module_includes hpx_${_module} INTERFACE_INCLUDE_DIRECTORIES)
      target_include_directories(hpx_wrap PRIVATE ${_module_includes})
    endforeach()

    set_property(TARGET hpx_wrap PROPERTY FOLDER "Core")
    set(hpx_targets ${hpx_targets} hpx_wrap)
  endif()

endif()

foreach(_keyword PUBLIC;PRIVATE)
  get_property(HPX_TARGET_COMPILE_DEFINITIONS_VAR
    GLOBAL PROPERTY HPX_TARGET_COMPILE_DEFINITIONS_${_keyword})
  foreach(_flag ${HPX_TARGET_COMPILE_DEFINITIONS_VAR})
    target_compile_definitions(hpx ${_keyword} ${_flag})
    if(HPX_WITH_DYNAMIC_HPX_MAIN AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
      target_compile_definitions(hpx_wrap ${_keyword} ${_flag})
    endif()
    if(NOT HPX_WITH_STATIC_LINKING)
      target_compile_definitions(hpx_init ${_keyword} ${_flag})
    endif()
    foreach(_module ${HPX_LIBS})
      target_compile_definitions(hpx_${_module} ${_keyword} ${_flag})
    endforeach()
  endforeach()

  get_property(HPX_TARGET_COMPILE_OPTIONS_VAR
    GLOBAL PROPERTY HPX_TARGET_COMPILE_OPTIONS_${_keyword})
  foreach(_flag ${HPX_TARGET_COMPILE_OPTIONS_VAR})
    target_compile_options(hpx ${_keyword} ${_flag})
    target_compile_options(hpx_internal_flags INTERFACE ${_flag})
    if(HPX_WITH_DYNAMIC_HPX_MAIN AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux") OR (APPLE)))
      target_compile_options(hpx_wrap ${_keyword} ${_flag})
    endif()
    if(NOT HPX_WITH_STATIC_LINKING)
      target_compile_options(hpx_init ${_keyword} ${_flag})
    endif()
    foreach(_module ${HPX_LIBS})
      target_compile_options(hpx_${_module} ${_keyword} ${_flag})
    endforeach()
  endforeach()
endforeach()

set(_optional)
if(HPX_WITH_DEFAULT_TARGETS)
  set(_optional OPTIONAL)
endif()

install(
  TARGETS
    ${hpx_targets}
  EXPORT HPXTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  COMPONENT runtime
  ${_optional}
)

hpx_export_targets(${hpx_targets})

foreach(target ${hpx_targets})
  add_hpx_pseudo_dependencies(core ${target})
endforeach()

# Add particular targets for the pkgconfig setup
add_library(hpx_application INTERFACE)
target_link_libraries(hpx_application INTERFACE hpx_wrap)
target_compile_definitions(hpx_application INTERFACE HPX_APPLICATION_EXPORTS)
add_library(hpx::application ALIAS hpx_application)

add_library(hpx_component INTERFACE)
target_compile_definitions(hpx_component INTERFACE HPX_COMPONENT_EXPORTS)
target_link_libraries(hpx_component INTERFACE hpx)
add_library(hpx::component ALIAS hpx_component)
