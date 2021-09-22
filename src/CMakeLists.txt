# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2018 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ##############################################################################
# gather sources

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(hpx_SOURCES
      runtime/parcelset/detail/parcel_await.cpp
      runtime/parcelset/detail/parcel_route_handler.cpp
      runtime/parcelset/detail/per_action_data_counter.cpp
      runtime/parcelset/locality.cpp
      runtime/parcelset/parcel.cpp
      runtime/parcelset/parcelhandler.cpp
      runtime/parcelset/parcelport.cpp
      runtime/parcelset/put_parcel.cpp
      runtime/set_parcel_write_handler.cpp
  )
  list(TRANSFORM hpx_SOURCES PREPEND ${PROJECT_SOURCE_DIR}/src/)
else()
  set(hpx_SOURCES ${PROJECT_SOURCE_DIR}/libs/src/dummy.cpp)
endif()

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

# ##############################################################################
# gather headers

# List all headers (local and distributed)
set(hpx_HEADERS)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  set(hpx_HEADERS
      hpx/plugins/binary_filter_factory_base.hpp
      hpx/plugins/binary_filter_factory.hpp
      hpx/plugins/message_handler_factory_base.hpp
      hpx/plugins/message_handler_factory.hpp
      hpx/plugins/parcel/coalescing_counter_registry.hpp
      hpx/plugins/parcel/coalescing_message_handler.hpp
      hpx/plugins/parcel/coalescing_message_handler_registration.hpp
      hpx/plugins/parcel/message_buffer.hpp
      hpx/plugins/parcelport_factory_base.hpp
      hpx/plugins/parcelport_factory.hpp
      hpx/plugins/parcelport/mpi/header.hpp
      hpx/plugins/parcelport/mpi/locality.hpp
      hpx/plugins/parcelport/mpi/receiver_connection.hpp
      hpx/plugins/parcelport/mpi/receiver.hpp
      hpx/plugins/parcelport/mpi/sender_connection.hpp
      hpx/plugins/parcelport/mpi/sender.hpp
      hpx/plugins/parcelport/mpi/tag_provider.hpp
      hpx/plugins/parcelport/tcp/connection_handler.hpp
      hpx/plugins/parcelport/tcp/locality.hpp
      hpx/plugins/parcelport/tcp/receiver.hpp
      hpx/plugins/parcelport/tcp/sender.hpp
      hpx/plugins/plugin_factory_base.hpp
      hpx/plugins/plugin_registry.hpp
      hpx/plugins/unique_plugin_name.hpp
      hpx/runtime/message_handler_fwd.hpp
      hpx/runtime/parcelset/connection_cache.hpp
      hpx/runtime/parcelset/decode_parcels.hpp
      hpx/runtime/parcelset/detail/call_for_each.hpp
      hpx/runtime/parcelset/detail/data_point.hpp
      hpx/runtime/parcelset/detail/gatherer.hpp
      hpx/runtime/parcelset/detail/parcel_await.hpp
      hpx/runtime/parcelset/detail/parcel_route_handler.hpp
      hpx/runtime/parcelset/detail/per_action_data_counter.hpp
      hpx/runtime/parcelset/encode_parcels.hpp
      hpx/runtime/parcelset_fwd.hpp
      hpx/runtime/parcelset/locality.hpp
      hpx/runtime/parcelset/parcel_buffer.hpp
      hpx/runtime/parcelset/parcelhandler.hpp
      hpx/runtime/parcelset/parcel.hpp
      hpx/runtime/parcelset/parcelport_connection.hpp
      hpx/runtime/parcelset/parcelport.hpp
      hpx/runtime/parcelset/parcelport_impl.hpp
      hpx/runtime/parcelset/policies/message_handler.hpp
      hpx/runtime/parcelset/put_parcel.hpp
      hpx/runtime/set_parcel_write_handler.hpp
      hpx/traits/action_message_handler.hpp
      hpx/traits/action_serialization_filter.hpp
  )
  list(TRANSFORM hpx_HEADERS PREPEND ${PROJECT_SOURCE_DIR}/)
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
  ROOT "${PROJECT_SOURCE_DIR}/src"
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

add_hpx_source_group(
  NAME hpx
  CLASS "Header Files"
  ROOT "${PROJECT_SOURCE_DIR}/hpx"
  TARGETS ${hpx_HEADERS}
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

if(HPX_WITH_UNITY_BUILD)
  set_target_properties(hpx_full PROPERTIES UNITY_BUILD ON)
endif()

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

target_link_libraries(hpx_full PUBLIC hpx_base_libraries)

# Link to each parcelport plugin
hpx_warn("HPX_STATIC_PARCELPORT_PLUGINS=\"${HPX_STATIC_PARCELPORT_PLUGINS}\"")
foreach(_parcelport_plugin ${HPX_STATIC_PARCELPORT_PLUGINS})
  target_link_libraries(hpx_full PRIVATE ${_parcelport_plugin})
endforeach()

if((NOT HPX_WITH_STATIC_LINKING) AND (("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
                                      OR (APPLE))
)
  set_target_properties(hpx_full PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_compile_definitions(hpx_full PRIVATE HPX_COMPONENT_NAME=hpx HPX_EXPORTS)

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
