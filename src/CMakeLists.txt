# Copyright (c) 2007-2017 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
# Copyright (c)      2018 Nikunj Gupta
#
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

# ##############################################################################
# gather sources

set(hpx_SOURCES runtime/threads/thread_pool_suspension_helpers.cpp)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  list(
    APPEND
    hpx_SOURCES
    lcos/base_lco.cpp
    lcos/base_lco_with_value_1.cpp
    lcos/base_lco_with_value_2.cpp
    lcos/base_lco_with_value_3.cpp
    lcos/base_lco_with_value.cpp
    lcos/promise.cpp
    runtime/agas/addressing_service.cpp
    runtime/agas/big_boot_barrier.cpp
    runtime/agas/component_namespace.cpp
    runtime/agas/detail/bootstrap_component_namespace.cpp
    runtime/agas/detail/bootstrap_locality_namespace.cpp
    runtime/agas/detail/hosted_component_namespace.cpp
    runtime/agas/detail/hosted_locality_namespace.cpp
    runtime/agas/interface.cpp
    runtime/agas/locality_namespace.cpp
    runtime/agas/server/component_namespace_server.cpp
    runtime/agas/server/locality_namespace_server.cpp
    runtime/agas/server/route.cpp
    runtime/agas/server/symbol_namespace_server.cpp
    runtime/agas/symbol_namespace.cpp
    runtime/applier/applier.cpp
    runtime/applier/bind_naming_wrappers.cpp
    runtime/components/binpacking_distribution_policy.cpp
    runtime/components/component_registry.cpp
    runtime/components/component_type.cpp
    runtime/components/console_error_sink.cpp
    runtime/components/console_logging.cpp
    runtime/components/runtime_support.cpp
    runtime/components/server/component_base.cpp
    runtime/components/server/console_error_sink_server.cpp
    runtime/components/server/console_error_sink_singleton.cpp
    runtime/components/server/console_logging_server.cpp
    runtime/components/server/destroy_component.cpp
    runtime/components/server/runtime_support_server.cpp
    runtime/components/server/wrapper_heap.cpp
    runtime/components/stubs/runtime_support_stubs.cpp
    runtime/get_locality_name.cpp
    runtime/parcelset/detail/parcel_await.cpp
    runtime/parcelset/detail/parcel_route_handler.cpp
    runtime/parcelset/detail/per_action_data_counter.cpp
    runtime/parcelset/locality.cpp
    runtime/parcelset/parcel.cpp
    runtime/parcelset/parcelhandler.cpp
    runtime/parcelset/parcelport.cpp
    runtime/parcelset/put_parcel.cpp
    runtime/set_parcel_write_handler.cpp
    runtime/threads/threadmanager_counters.cpp
    runtime/trigger_lco.cpp
    runtime_distributed.cpp
    state.cpp
    util/activate_counters.cpp
    util/generate_unique_ids.cpp
    util/init_logging.cpp
    util/one_size_heap_list.cpp
    util/pool_timer.cpp
    util/query_counters.cpp
    util/register_locks_globally.cpp
  )
endif()

list(TRANSFORM hpx_SOURCES PREPEND ${PROJECT_SOURCE_DIR}/src/)

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
set(hpx_HEADERS hpx/lcos_fwd.hpp hpx/traits/managed_component_policies.hpp
                hpx/traits/rma_memory_region_traits.hpp
)

if(HPX_WITH_DISTRIBUTED_RUNTIME)
  list(
    APPEND
    hpx_HEADERS
    hpx/components/containers/container_distribution_policy.hpp
    hpx/components_fwd.hpp
    hpx/lcos/base_lco.hpp
    hpx/lcos/base_lco_with_value.hpp
    hpx/lcos/detail/promise_base.hpp
    hpx/lcos/detail/promise_lco.hpp
    hpx/lcos/object_semaphore.hpp
    hpx/lcos/packaged_action.hpp
    hpx/lcos/promise.hpp
    hpx/lcos/server/object_semaphore.hpp
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
    hpx/runtime_fwd.hpp
    hpx/runtime/actions/continuation2_impl.hpp
    hpx/runtime/actions/continuation_impl.hpp
    hpx/runtime/actions/lambda_to_action.hpp
    hpx/runtime/actions/make_continuation.hpp
    hpx/runtime/agas/addressing_service.hpp
    hpx/runtime/agas/big_boot_barrier.hpp
    hpx/runtime/agas/component_namespace.hpp
    hpx/runtime/agas/detail/bootstrap_component_namespace.hpp
    hpx/runtime/agas/detail/bootstrap_locality_namespace.hpp
    hpx/runtime/agas/detail/hosted_component_namespace.hpp
    hpx/runtime/agas/detail/hosted_locality_namespace.hpp
    hpx/runtime/agas/interface.hpp
    hpx/runtime/agas/locality_namespace.hpp
    hpx/runtime/agas/namespace_action_code.hpp
    hpx/runtime/agas/server/component_namespace.hpp
    hpx/runtime/agas/server/locality_namespace.hpp
    hpx/runtime/agas/server/symbol_namespace.hpp
    hpx/runtime/agas/symbol_namespace.hpp
    hpx/runtime/basename_registration_fwd.hpp
    hpx/runtime/basename_registration.hpp
    hpx/runtime/components/binpacking_distribution_policy.hpp
    hpx/runtime/components/client_base.hpp
    hpx/runtime/components/client.hpp
    hpx/runtime/components/colocating_distribution_policy.hpp
    hpx/runtime/components/component_commandline_base.hpp
    hpx/runtime/components/component_commandline.hpp
    hpx/runtime/components/component_factory_base.hpp
    hpx/runtime/components/component_factory.hpp
    hpx/runtime/components/component_registry.hpp
    hpx/runtime/components/component_startup_shutdown_base.hpp
    hpx/runtime/components/component_startup_shutdown.hpp
    hpx/runtime/components/console_error_sink.hpp
    hpx/runtime/components/console_logging.hpp
    hpx/runtime/components/copy_component.hpp
    hpx/runtime/components/create_component_helpers.hpp
    hpx/runtime/components/default_distribution_policy.hpp
    hpx/runtime/components/derived_component_factory.hpp
    hpx/runtime/components_fwd.hpp
    hpx/runtime/components/make_client.hpp
    hpx/runtime/components/migrate_component.hpp
    hpx/runtime/components/new.hpp
    hpx/runtime/components/runtime_support.hpp
    hpx/runtime/components/server/abstract_component_base.hpp
    hpx/runtime/components/server/abstract_migration_support.hpp
    hpx/runtime/components/server/component_base.hpp
    hpx/runtime/components/server/component_database.hpp
    hpx/runtime/components/server/component_heap.hpp
    hpx/runtime/components/server/component.hpp
    hpx/runtime/components/server/console_error_sink.hpp
    hpx/runtime/components/server/console_error_sink_singleton.hpp
    hpx/runtime/components/server/console_logging.hpp
    hpx/runtime/components/server/copy_component.hpp
    hpx/runtime/components/server/create_component_fwd.hpp
    hpx/runtime/components/server/create_component.hpp
    hpx/runtime/components/server/destroy_component.hpp
    hpx/runtime/components/server/distributed_metadata_base.hpp
    hpx/runtime/components/server/executor_component.hpp
    hpx/runtime/components/server/fixed_component_base.hpp
    hpx/runtime/components/server/invoke_function.hpp
    hpx/runtime/components/server/locking_hook.hpp
    hpx/runtime/components/server/managed_component_base.hpp
    hpx/runtime/components/server/migrate_component.hpp
    hpx/runtime/components/server/migration_support.hpp
    hpx/runtime/components/server/runtime_support.hpp
    hpx/runtime/components/server/simple_component_base.hpp
    hpx/runtime/components/server/wrapper_heap.hpp
    hpx/runtime/components/server/wrapper_heap_list.hpp
    hpx/runtime/components/stubs/runtime_support.hpp
    hpx/runtime/components/stubs/stub_base.hpp
    hpx/runtime/components/target_distribution_policy.hpp
    hpx/runtime/components/unwrapping_result_policy.hpp
    hpx/runtime/find_here.hpp
    hpx/runtime/find_all_localities.hpp
    hpx/runtime/get_locality_name.hpp
    hpx/runtime/get_ptr.hpp
    hpx/runtime/message_handler_fwd.hpp
    hpx/runtime/naming/resolver_client.hpp
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
    hpx/runtime/runtime_fwd.hpp
    hpx/runtime/set_parcel_write_handler.hpp
    hpx/runtime/threads/threadmanager_counters.hpp
    hpx/runtime/trigger_lco.hpp
    hpx/state.hpp
    hpx/traits/action_message_handler.hpp
    hpx/traits/action_serialization_filter.hpp
    hpx/traits/is_client.hpp
    hpx/traits/is_continuation.hpp
    hpx/traits/is_distribution_policy.hpp
    hpx/util/activate_counters.hpp
    hpx/util/bind_action.hpp
    hpx/util/connection_cache.hpp
    hpx/util/functional/colocated_helpers.hpp
    hpx/util/functional/segmented_iterator_helpers.hpp
    hpx/util/generate_unique_ids.hpp
    hpx/util/init_logging.hpp
    hpx/util/one_size_heap_list.hpp
    hpx/util/pool_timer.hpp
    hpx/util/query_counters.hpp
    hpx/util/register_locks_globally.hpp
    hpx/util/remove_local_destinations.hpp
    hpx/util/storage/tuple.hpp
    hpx/util/wrapper_heap_base.hpp
  )
endif()

list(TRANSFORM hpx_HEADERS PREPEND ${PROJECT_SOURCE_DIR}/)

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

target_link_libraries(
  hpx_full
  PUBLIC hpx_public_flags
  PRIVATE hpx_private_flags
)
target_link_libraries(hpx_full PUBLIC hpx_core hpx_parallelism)
target_link_libraries(hpx_full PUBLIC hpx_dependencies_boost)

# Set the basic search paths for the HPX headers
target_include_directories(
  hpx_full
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}> $<INSTALL_INTERFACE:include>
)

# FIXME : temporary add public dependencies to the boost libraries which are
# only linked in the modules (which are themselves privately linked to hpx), it
# should be removed when we make the modules object libraries that we publicly
# link to hpx
if(HPX_PROGRAM_OPTIONS_WITH_BOOST_PROGRAM_OPTIONS_COMPATIBILITY)
  target_link_libraries(hpx_full PUBLIC Boost::program_options)
endif()

# Enable SIMD via VC if it is available
if(TARGET Vc::vc)
  target_link_libraries(hpx_full PUBLIC Vc::vc)
endif()

if(TARGET Amplifier::amplifier)
  target_link_libraries(hpx_full PRIVATE Amplifier::amplifier)
endif()

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
