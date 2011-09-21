# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
  
add_subdirectory(components)

add_library(hpx_init STATIC hpx_init.cpp)
  
set_property(TARGET hpx_init APPEND
  PROPERTY COMPILE_DEFINITIONS
  "HPX_COMPONENT_EXPORTS")

###############################################################################
# POSIX 
if(UNIX)
  foreach(lib ${hpx_CORE})
    set(${lib}_SOURCES "" CACHE INTERNAL "Sources for lib${lib}." FORCE)
  endforeach()

  add_hpx_library_sources(hpx
    exception.cpp
    exception_list.cpp
    pre_main.cpp
    state.cpp
    version.cpp
    runtime.cpp)

  add_subdirectory(lcos)
  add_subdirectory(performance_counters)
  add_subdirectory(runtime)
  add_subdirectory(util)

  # Boost Chrono sources (as long as this library is not in Boost)
  if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
    add_hpx_library_sources(hpx
      ${hpx_SOURCE_DIR}/external/chrono/libs/chrono/src/chrono.cpp)
  endif()
 
  # Add Boost.Backtrace sources if stack traces are enabled 
  if(HPX_STACKTRACES)
    add_hpx_library_sources(hpx
      ${hpx_SOURCE_DIR}/external/backtrace/libs/backtrace/src/backtrace.cpp)
  endif()

  if(HPX_WARNINGS)
    hpx_use_flag_if_available(Wextra)

    # We never want trampolines
    hpx_use_flag_if_available(Wtrampolines)

    # These are usually benign and can't be suppressed because of
    # interface requirements
    hpx_use_flag_if_available(Wno-unused-parameter)
    hpx_use_flag_if_available(Wno-unused-but-set-parameter)

    # There are ignored qualifiers in Boost, so we have to ignore them
    hpx_use_flag_if_available(Wno-ignored-qualifiers)

    # Be extra strict about format checks
    # Boost.Logging is built on fprintf, sadly
    hpx_use_flag_if_available(Wformat=2 NAME Wformat_2)
    hpx_use_flag_if_available(Wno-format-nonliteral) 

    # Self initialization is dangerous
    hpx_use_flag_if_available(Winit-self)

    # For portability
    hpx_use_flag_if_available(Wdouble-promotion)

    # Warn about casting that violates qualifiers or alignment
    hpx_use_flag_if_available(Wcast-qual)
    hpx_use_flag_if_available(Wcast-align)

    hpx_use_flag_if_available(Werror=trampolines NAME Werror_trampolines)
    hpx_use_flag_if_available(Werror=parentheses NAME Werror_parentheses)
    hpx_use_flag_if_available(Werror=reorder NAME Werror_reorder)
    hpx_use_flag_if_available(Werror=return-type NAME Werror_return-type)
    hpx_use_flag_if_available(Werror=sequence-point NAME Werror_sequence-point)
    hpx_use_flag_if_available(Werror=uninitialized NAME Werror_uninitialized)
    hpx_use_flag_if_available(Werror=format NAME Werror_format)
    hpx_use_flag_if_available(Werror=missing-braces NAME Werror_missing-braces)
    hpx_use_flag_if_available(Werror=sign-compare NAME Werror_sign-compare)
  endif()

  add_hpx_library_sources(hpx
    ${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/swapcontext.cpp)

  #############################################################################
  # libhpx_serialization
  add_library(hpx_serialization SHARED ${hpx_serialization_SOURCES})

  set_target_properties(hpx_serialization PROPERTIES
    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib/hpx"
    VERSION ${HPX_VERSION}
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME hpx_serialization)

  target_link_libraries(hpx_serialization
    ${hpx_MALLOC_LIBRARY}
    ${BOOST_serialization_LIBRARY}
    ${BOOST_system_LIBRARY})

  get_target_property(path hpx_serialization LOCATION)
  get_filename_component(name "${path}" NAME)

  set_property(TARGET hpx_serialization APPEND
    PROPERTY COMPILE_DEFINITIONS
    "HPX_LIBRARY=\"${name}\""
    "HPX_COMPONENT_NAME=hpx"
    "HPX_EXPORTS"
    "BOOST_COROUTINE_EXPORTS")

  #############################################################################
  # libhpx
  add_library(hpx SHARED ${hpx_SOURCES})
  set_target_properties(hpx PROPERTIES
    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib/hpx"
    VERSION ${HPX_VERSION}
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME hpx)

  target_link_libraries(hpx hpx_serialization
    ${hpx_MALLOC_LIBRARY} ${BOOST_FOUND_LIBRARIES})
 
  get_target_property(path hpx LOCATION)
  get_filename_component(name "${path}" NAME)
  set_property(TARGET hpx APPEND
    PROPERTY COMPILE_DEFINITIONS
    "HPX_LIBRARY=\"${name}\""
    "HPX_COMPONENT_NAME=hpx"
    "HPX_EXPORTS"
    "BOOST_ENABLE_ASSERT_HANDLER"
    "BOOST_COROUTINE_EXPORTS")

  # libhpx is a component
  set_property(TARGET hpx 
               PROPERTY LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/lib/hpx")
  set_property(TARGET hpx 
               PROPERTY ARCHIVE_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/lib/hpx")
  set_property(TARGET hpx 
               PROPERTY RUNTIME_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/lib/hpx")

  foreach(lib ${hpx_CORE})
    get_target_property(path ${lib} LOCATION)
    get_filename_component(name "${path}" NAME)
    hpx_library_install(core ${name})
    hpx_library_install(core ${name}.${HPX_SOVERSION})
    hpx_library_install(core ${name}.${HPX_VERSION})
  endforeach()
else()
###############################################################################
# MSVC 
  # global
  set(hpxlib_global_HEADERS
      ${hpx_SOURCE_DIR}/hpx/config.hpp
      ${hpx_SOURCE_DIR}/hpx/exception.hpp
      ${hpx_SOURCE_DIR}/hpx/exception_list.hpp
      ${hpx_SOURCE_DIR}/hpx/hpx.hpp
      ${hpx_SOURCE_DIR}/hpx/hpx_fwd.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime.hpp
      ${hpx_SOURCE_DIR}/hpx/version.hpp)
  source_group("Header Files" 
      FILES ${hpxlib_global_HEADERS})
  
  # config
  set(hpxlib_config_HEADERS
      ${hpx_SOURCE_DIR}/hpx/config/defaults.hpp
      ${hpx_SOURCE_DIR}/hpx/config/export_definitions.hpp
      ${hpx_SOURCE_DIR}/hpx/config/warnings_prefix.hpp
      ${hpx_SOURCE_DIR}/hpx/config/warnings_suffix.hpp)
  source_group("Header Files\\config" 
      FILES ${hpxlib_config_HEADERS})
  
  # include
  set(hpxlib_include_HEADERS
      ${hpx_SOURCE_DIR}/hpx/include/actions.hpp
      ${hpx_SOURCE_DIR}/hpx/include/applier.hpp
      ${hpx_SOURCE_DIR}/hpx/include/components.hpp
      ${hpx_SOURCE_DIR}/hpx/include/lcos.hpp
      ${hpx_SOURCE_DIR}/hpx/include/naming.hpp
      ${hpx_SOURCE_DIR}/hpx/include/parcelset.hpp
      ${hpx_SOURCE_DIR}/hpx/include/runtime.hpp
      ${hpx_SOURCE_DIR}/hpx/include/threadmanager.hpp
      ${hpx_SOURCE_DIR}/hpx/include/util.hpp)
  source_group("Header Files\\include" 
      FILES ${hpxlib_include_HEADERS})
  
  # lcos
  set(hpxlib_lcos_HEADERS
      ${hpx_SOURCE_DIR}/hpx/lcos/barrier.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/queue.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/object_semaphore.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/stubs/barrier.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/stubs/queue.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/stubs/object_semaphore.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/server/barrier.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/server/queue.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/server/object_semaphore.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/base_lco.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/condition.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/contin.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/local_counting_semaphore.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/local_dataflow_variable.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/local_shared_mutex.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/eager_future.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/eager_future_constructors.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/eager_future_constructors_direct.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/future_callback.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/future_value.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/future_value_full_empty_bit.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/future_value_promise.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/future_wait.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/lazy_future.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/lazy_future_constructors.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/lazy_future_constructors_direct.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/local_barrier.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/mutex.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/recursive_mutex.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/thunk.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/thunk_client_constructors.hpp
      ${hpx_SOURCE_DIR}/hpx/lcos/thunk_constructors.hpp)
  source_group("Header Files\\lcos" 
      FILES ${hpxlib_lcos_HEADERS})
  
  # Performance counters
  set(hpxlib_performance_counters_HEADERS
      ${hpx_SOURCE_DIR}/hpx/performance_counters/counters.hpp
      ${hpx_SOURCE_DIR}/hpx/performance_counters/registry.hpp)
  source_group("Header Files\\performance_counters" 
      FILES ${hpxlib_performance_counters_HEADERS})
  
  set(hpxlib_performance_counters_stubs_HEADERS
      ${hpx_SOURCE_DIR}/hpx/performance_counters/stubs/performance_counter.hpp)
  source_group("Header Files\\performance_counters\\stubs" 
      FILES ${hpxlib_performance_counters_stubs_HEADERS})
  
  set(hpxlib_performance_counters_server_HEADERS
      ${hpx_SOURCE_DIR}/hpx/performance_counters/server/base_performance_counter.hpp
      ${hpx_SOURCE_DIR}/hpx/performance_counters/server/raw_counter.hpp)
  source_group("Header Files\\performance_counters\\server" 
      FILES ${hpxlib_performance_counters_server_HEADERS})
  
  # runtime
  set(hpxlib_runtime_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/get_lva.hpp)
  source_group("Header Files\\runtime" 
      FILES ${hpxlib_runtime_HEADERS})
  
  # actions 
  set(hpxlib_actions_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_constructors.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_manager.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_support.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/component_action.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/component_action_implementations.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/continuation.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/continuation_impl.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/manage_object_action.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/plain_action.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/actions/plain_action_implementations.hpp)
  source_group("Header Files\\runtime\\actions" 
      FILES ${hpxlib_actions_HEADERS})

  if("${HPX_AGAS_VERSION}" STREQUAL "2")
    # AGAS
    set(hpxlib_agas_HEADERS
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/traits.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/traits_fwd.hpp)
    source_group("Header Files\\runtime\\agas" 
        FILES ${hpxlib_agas_HEADERS})
  
    set(hpxlib_agas_client_HEADERS
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/client/legacy_agent.hpp)
    source_group("Header Files\\runtime\\agas\\client" 
        FILES ${hpxlib_agas_client_HEADERS})
  
    set(hpxlib_agas_database_HEADERS
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/database/table.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/database/traits.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/database/traits_fwd.hpp)
    source_group("Header Files\\runtime\\agas\\database" 
        FILES ${hpxlib_agas_database_HEADERS})
  
    set(hpxlib_agas_database_backend_HEADERS
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/database/backend/std_map.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/database/backend/default.hpp)
    source_group("Header Files\\Runtime\\AGAS\\Database\\Backend" 
        FILES ${hpxlib_agas_database_backend_HEADERS})
  
    set(hpxlib_agas_network_HEADERS
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/network/full_gva.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/network/table.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/network/traits.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/network/traits_fwd.hpp)
    source_group("Header Files\\runtime\\agas\\network" 
        FILES ${hpxlib_agas_network_HEADERS})
  
    set(hpxlib_agas_network_backend_HEADERS
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/network/backend/default.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/network/backend/ib.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/network/backend/tcpip.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/network/backend/udpip.hpp)
    source_group("Header Files\\runtime\\agas\\network\\backend" 
        FILES ${hpxlib_agas_network_backend_HEADERS})
  
    set(hpxlib_agas_namespace_HEADERS
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/namespace/component.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/namespace/primary.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/namespace/symbol.hpp)
    source_group("Header Files\\runtime\\agas\\namespace" 
        FILES ${hpxlib_agas_namespace_HEADERS})
  
    set(hpxlib_agas_namespace_server_HEADERS
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/namespace/server/component.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/namespace/server/primary.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/namespace/server/symbol.hpp)
    source_group("Header Files\\runtime\\agas\\namespace\\server" 
        FILES ${hpxlib_agas_namespace_server_HEADERS})
  
    set(hpxlib_agas_namespace_stubs_HEADERS
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/namespace/stubs/component.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/namespace/stubs/primary.hpp
        ${hpx_SOURCE_DIR}/hpx/runtime/agas/namespace/stubs/symbol.hpp)
    source_group("Header Files\\runtime\\agas\\namespace\\stubs" 
        FILES ${hpxlib_agas_namespace_stubs_HEADERS})
  endif()

  # applier
  set(hpxlib_applier_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/applier/applier.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply_helper.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply_helper_implementations.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply_implementations.hpp)
  source_group("Header Files\\runtime\\applier" 
      FILES ${hpxlib_applier_HEADERS})

  # components
  set(hpxlib_components_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/components/client_base.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/component_type.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory_base.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory_one.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/component_registry.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/component_registry_base.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/component_startup_shutdown.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/component_startup_shutdown_base.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/constructor_argument.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/console_error_sink.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/console_logging.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/constructor_argument.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/derived_component_factory.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/derived_component_factory_one.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/memory_block.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/plain_component_factory.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/runtime_support.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/unique_component_name.hpp)
  source_group("Header Files\\runtime\\components" 
      FILES ${hpxlib_components_HEADERS})

  set(hpxlib_components_server_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/console_error_sink.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/console_error_sink_singleton.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/console_logging.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/manage_component.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/managed_component_base.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/memory.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/memory_block.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/plain_function.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/runtime_support.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/simple_component_base.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/wrapper_heap.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/server/wrapper_heap_list.hpp)
  source_group("Header Files\\runtime\\components\\server" 
      FILES ${hpxlib_components_server_HEADERS})

  set(hpxlib_components_stubs_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/memory_block.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/runtime_support.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/stub_base.hpp)
  source_group("Header Files\\runtime\\components\\stubs" 
      FILES ${hpxlib_components_stubs_HEADERS})

  # naming
  set(hpxlib_naming_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/address.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/bulk_resolver_client.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/locality.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/name.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/resolver_client.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/resolver_client_connection.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/resolver_server.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/detail/resolver_do_undo.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/connection.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/reply.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/request.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/request_handler.hpp)
  source_group("Header Files\\runtime\\naming" 
      FILES ${hpxlib_naming_HEADERS})

  # parcelset
  set(hpxlib_parcelset_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcel.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelhandler.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelhandler_queue_base.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelport.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelport_connection.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/policies/global_parcelhandler_queue.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/parcelport_queue.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/parcelport_server_connection.hpp)
  source_group("Header Files\\parcelset" 
      FILES ${hpxlib_parcelset_HEADERS})

  # threadmanager
  set(hpxlib_threadmanager_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread_affinity.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread_helpers.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread_init_data.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/threadmanager.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/detail/tagged_thread_state.hpp)
  source_group("Header Files\\runtime\\threads" 
      FILES ${hpxlib_threadmanager_HEADERS})

  set(hpxlib_threadmanager_policies_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/abp_queue_scheduler.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/callback_notifier.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/global_queue_scheduler.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/local_queue_scheduler.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/local_priority_queue_scheduler.hpp
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/thread_queue.hpp)
  source_group("Header Files\\runtime\\threads\\policies" 
      FILES ${hpxlib_threadmanager_policies_HEADERS})

  set(hpxlib_threadmanager_detail_HEADERS
      ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/tagged_thread_state.hpp)
  source_group("Header Files\\runtime\\threads\\detail" 
      FILES ${hpxlib_threadmanager_detail_HEADERS})

  # util
  set(hpxlib_util_HEADERS
      ${hpx_SOURCE_DIR}/hpx/util/asio_util.hpp
      ${hpx_SOURCE_DIR}/hpx/util/block_profiler.hpp
      ${hpx_SOURCE_DIR}/hpx/util/connection_cache.hpp
      ${hpx_SOURCE_DIR}/hpx/util/container_device.hpp
      ${hpx_SOURCE_DIR}/hpx/util/filesystem_compatibility.hpp
      ${hpx_SOURCE_DIR}/hpx/util/full_empty_memory.hpp
      ${hpx_SOURCE_DIR}/hpx/util/full_empty_store.hpp
      ${hpx_SOURCE_DIR}/hpx/util/future.hpp
      ${hpx_SOURCE_DIR}/hpx/util/generate_unique_ids_v1.hpp
      ${hpx_SOURCE_DIR}/hpx/util/generate_unique_ids_v2.hpp
      ${hpx_SOURCE_DIR}/hpx/util/high_resolution_timer.hpp
      ${hpx_SOURCE_DIR}/hpx/util/ini.hpp
      ${hpx_SOURCE_DIR}/hpx/util/init_ini_data.hpp
      ${hpx_SOURCE_DIR}/hpx/util/itt_notify.hpp
      ${hpx_SOURCE_DIR}/hpx/util/io_service_pool.hpp
      ${hpx_SOURCE_DIR}/hpx/util/locking_helpers.hpp
      ${hpx_SOURCE_DIR}/hpx/util/logging.hpp
      ${hpx_SOURCE_DIR}/hpx/util/map_hostnames.hpp
      ${hpx_SOURCE_DIR}/hpx/util/one_size_heap_list.hpp
      ${hpx_SOURCE_DIR}/hpx/util/pbs_environment.hpp
      ${hpx_SOURCE_DIR}/hpx/util/performance_counters.hpp
      ${hpx_SOURCE_DIR}/hpx/util/random.hpp
      ${hpx_SOURCE_DIR}/hpx/util/runtime_configuration.hpp
      ${hpx_SOURCE_DIR}/hpx/util/safe_bool.hpp
      ${hpx_SOURCE_DIR}/hpx/util/serialize_asio_address_v4.hpp
      ${hpx_SOURCE_DIR}/hpx/util/serialize_asio_address_v6.hpp
      ${hpx_SOURCE_DIR}/hpx/util/serialize_asio_basic_endpoint.hpp
      ${hpx_SOURCE_DIR}/hpx/util/serialize_exception.hpp
      ${hpx_SOURCE_DIR}/hpx/util/serialize_sequence.hpp
      ${hpx_SOURCE_DIR}/hpx/util/spinlock_pool.hpp
      ${hpx_SOURCE_DIR}/hpx/util/static.hpp
      ${hpx_SOURCE_DIR}/hpx/util/stringstream.hpp
      ${hpx_SOURCE_DIR}/hpx/util/time_logger.hpp
      ${hpx_SOURCE_DIR}/hpx/util/unlock_lock.hpp
      ${hpx_SOURCE_DIR}/hpx/util/unmangle_name.hpp
      ${hpx_SOURCE_DIR}/hpx/util/value_logger.hpp)
  source_group("Header Files\\util" 
      FILES ${hpxlib_util_HEADERS})
  
  set(hpxlib_HEADERS 
      ${hpxlib_global_HEADERS}
      ${hpxlib_config_HEADERS}
      ${hpxlib_include_HEADERS}
      ${hpxlib_lcos_HEADERS}
      ${hpxlib_agas_HEADERS}
      ${hpxlib_performance_counters_HEADERS}
      ${hpxlib_performance_counters_stubs_HEADERS}
      ${hpxlib_performance_counters_server_HEADERS}
      ${hpxlib_runtime_HEADERS}
      ${hpxlib_actions_HEADERS}
      ${hpxlib_applier_HEADERS}
      ${hpxlib_components_HEADERS}
      ${hpxlib_components_server_HEADERS}
      ${hpxlib_components_stubs_HEADERS}
      ${hpxlib_naming_HEADERS}
      ${hpxlib_parcelset_HEADERS}
      ${hpxlib_threadmanager_HEADERS}
      ${hpxlib_threadmanager_policies_HEADERS}
      ${hpxlib_util_HEADERS})
  
  # global
  set(hpxlib_global_SOURCES
      ${hpx_SOURCE_DIR}/src/exception.cpp
      ${hpx_SOURCE_DIR}/src/exception_list.cpp
      ${hpx_SOURCE_DIR}/src/pre_main.cpp
      ${hpx_SOURCE_DIR}/src/state.cpp
      ${hpx_SOURCE_DIR}/src/version.cpp
      ${hpx_SOURCE_DIR}/src/runtime.cpp)
  source_group("Source Files" 
      FILES ${hpxlib_global_SOURCES})
  
  # lcos
  set(hpxlib_lcos_SOURCES
      ${hpx_SOURCE_DIR}/src/lcos/local_counting_semaphore.cpp
      ${hpx_SOURCE_DIR}/src/lcos/barrier/barrier.cpp
      ${hpx_SOURCE_DIR}/src/lcos/base_lco.cpp)
  source_group("Source Files\\lcos" 
      FILES ${hpxlib_lcos_SOURCES})
  
  # performance counters
  set(hpxlib_performance_counters_SOURCES
      ${hpx_SOURCE_DIR}/src/performance_counters/counters.cpp
      ${hpx_SOURCE_DIR}/src/performance_counters/manage_counter.cpp
      ${hpx_SOURCE_DIR}/src/performance_counters/manage_counter_type.cpp
      ${hpx_SOURCE_DIR}/src/performance_counters/registry.cpp)
  source_group("Source Files\\performance_counters" 
      FILES ${hpxlib_performance_counters_SOURCES})
  
  set(hpxlib_performance_counters_stubs_SOURCES
      ${hpx_SOURCE_DIR}/src/performance_counters/stubs/performance_counter.cpp)
  source_group("Source Files\\performance_counters\\stubs" 
      FILES ${hpxlib_performance_counters_stubs_SOURCES})
  
  set(hpxlib_performance_counters_server_SOURCES
      ${hpx_SOURCE_DIR}/src/performance_counters/server/raw_counter.cpp)
  source_group("Source Files\\performance_counters\\server" 
      FILES ${hpxlib_performance_counters_server_SOURCES})

  # actions 
  set(hpxlib_actions_SOURCES
      ${hpx_SOURCE_DIR}/src/runtime/actions/action_manager.cpp
      ${hpx_SOURCE_DIR}/src/runtime/actions/continuation.cpp)
  source_group("Source Files\\runtime\\actions" 
      FILES ${hpxlib_actions_SOURCES})

  # AGAS 
  if("${HPX_AGAS_VERSION}" STREQUAL "2")
    set(hpxlib_agas_SOURCES
        ${hpx_SOURCE_DIR}/src/runtime/agas/router/legacy.cpp
        ${hpx_SOURCE_DIR}/src/runtime/agas/router/big_boot_barrier.cpp
        ${hpx_SOURCE_DIR}/src/runtime/agas/stdmap_tcpip/component_module.cpp
        ${hpx_SOURCE_DIR}/src/runtime/agas/stdmap_tcpip/symbol_namespace.cpp
        ${hpx_SOURCE_DIR}/src/runtime/agas/stdmap_tcpip/component_namespace.cpp
        ${hpx_SOURCE_DIR}/src/runtime/agas/stdmap_tcpip/primary_namespace.cpp)
    source_group("Source Files\\runtime\\agas" 
        FILES ${hpxlib_agas_SOURCES})
  endif()

  # applier
  set(hpxlib_applier_SOURCES
      ${hpx_SOURCE_DIR}/src/runtime/applier/bind_naming_wrappers.cpp
      ${hpx_SOURCE_DIR}/src/runtime/applier/applier.cpp)
  source_group("Source Files\\runtime\\applier" 
      FILES ${hpxlib_applier_SOURCES})

  # components
  set(hpxlib_components_SOURCES
      ${hpx_SOURCE_DIR}/src/runtime/components/console_error_sink.cpp
      ${hpx_SOURCE_DIR}/src/runtime/components/console_logging.cpp
      ${hpx_SOURCE_DIR}/src/runtime/components/component_type.cpp)
  source_group("Source Files\\runtime\\components" 
      FILES ${hpxlib_components_SOURCES})

  set(hpxlib_components_server_SOURCES
      ${hpx_SOURCE_DIR}/src/runtime/components/server/console_error_sink.cpp
      ${hpx_SOURCE_DIR}/src/runtime/components/server/console_error_sink_singleton.cpp
      ${hpx_SOURCE_DIR}/src/runtime/components/server/console_logging.cpp
      ${hpx_SOURCE_DIR}/src/runtime/components/server/memory.cpp
      ${hpx_SOURCE_DIR}/src/runtime/components/server/memory_block.cpp
      ${hpx_SOURCE_DIR}/src/runtime/components/server/runtime_support.cpp)
  source_group("Source Files\\runtime\\components\\server" 
      FILES ${hpxlib_components_server_SOURCES})
  
  # naming 
  set(hpxlib_naming_SOURCES
      ${hpx_SOURCE_DIR}/src/runtime/naming/address.cpp
      ${hpx_SOURCE_DIR}/src/runtime/naming/bulk_resolver_client.cpp
      ${hpx_SOURCE_DIR}/src/runtime/naming/locality.cpp
      ${hpx_SOURCE_DIR}/src/runtime/naming/name.cpp
      ${hpx_SOURCE_DIR}/src/runtime/naming/resolver_client.cpp
      ${hpx_SOURCE_DIR}/src/runtime/naming/resolver_server.cpp
      ${hpx_SOURCE_DIR}/src/runtime/naming/detail/resolver_do_undo.cpp
      ${hpx_SOURCE_DIR}/src/runtime/naming/server/reply.cpp
      ${hpx_SOURCE_DIR}/src/runtime/naming/server/request.cpp
      ${hpx_SOURCE_DIR}/src/runtime/naming/server/request_handler.cpp)
  source_group("Source Files\\runtime\\naming" 
      FILES ${hpxlib_naming_SOURCES})
  
  # parcelset
  set(hpxlib_parcelset_SOURCES
      ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcel.cpp
      ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelport.cpp
      ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelport_connection.cpp)

  if("${HPX_AGAS_VERSION}" STREQUAL "2")
    set(hpxlib_parcelset_SOURCES ${hpxlib_parcelset_SOURCES}
        ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelhandler_v2.cpp)
  else()
    set(hpxlib_parcelset_SOURCES ${hpxlib_parcelset_SOURCES}
        ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelhandler_v1.cpp)
  endif()    

  source_group("Source Files\\runtime\\parcelset" 
      FILES ${hpxlib_parcelset_SOURCES})
  
  # threadmanager
  set(hpxlib_threadmanager_SOURCES
      ${hpx_SOURCE_DIR}/src/runtime/threads/thread.cpp
      ${hpx_SOURCE_DIR}/src/runtime/threads/thread_helpers.cpp
      ${hpx_SOURCE_DIR}/src/runtime/threads/threadmanager.cpp)
  source_group("Source Files\\runtime\\threads" 
      FILES ${hpxlib_threadmanager_SOURCES})

  # util 
  set(hpxlib_util_SOURCES
      ${hpx_SOURCE_DIR}/src/util/asio_util.cpp
      ${hpx_SOURCE_DIR}/src/util/ini.cpp
      ${hpx_SOURCE_DIR}/src/util/init_ini_data.cpp
      ${hpx_SOURCE_DIR}/src/util/itt_notify.cpp
      ${hpx_SOURCE_DIR}/src/util/io_service_pool.cpp
      ${hpx_SOURCE_DIR}/src/util/logging.cpp
      ${hpx_SOURCE_DIR}/src/util/performance_counters.cpp
      ${hpx_SOURCE_DIR}/src/util/runtime_configuration.cpp
      ${hpx_SOURCE_DIR}/src/util/serialize_exception.cpp)

  if("${HPX_AGAS_VERSION}" STREQUAL "2")
    set(hpxlib_util_SOURCES ${hpxlib_util_SOURCES}
      ${hpx_SOURCE_DIR}/src/util/generate_unique_ids_v2.cpp)
  else()
    set(hpxlib_util_SOURCES ${hpxlib_util_SOURCES}
      ${hpx_SOURCE_DIR}/src/util/generate_unique_ids_v1.cpp)
  endif()   

  source_group("Source Files\\util" 
      FILES ${hpxlib_util_SOURCES})
  set(hpxlib_SOURCES
    ${hpxlib_global_SOURCES}
    ${hpxlib_agas_SOURCES}
    ${hpxlib_lcos_SOURCES}
    ${hpxlib_performance_counters_SOURCES}
    ${hpxlib_performance_counters_stubs_SOURCES}
    ${hpxlib_performance_counters_server_SOURCES}
    ${hpxlib_actions_SOURCES}
    ${hpxlib_applier_SOURCES}
    ${hpxlib_components_SOURCES}
    ${hpxlib_components_server_SOURCES}
    ${hpxlib_naming_SOURCES}
    ${hpxlib_parcelset_SOURCES}
    ${hpxlib_threadmanager_SOURCES}
    ${hpxlib_util_SOURCES})
  
  if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
    set(hpxlib_SOURCES ${hpxlib_SOURCES} 
      ${hpx_SOURCE_DIR}/external/chrono/libs/chrono/src/chrono.cpp)
  endif()

  # Add Boost.Backtrace sources if stack traces are enabled 
  if(HPX_STACKTRACES)
    set(hpxlib_SOURCES ${hpxlib_SOURCES} 
      ${hpx_SOURCE_DIR}/external/backtrace/libs/backtrace/src/backtrace.cpp)
  endif()

  set(hpxserializationlib_HEADERS
      ${hpx_SOURCE_DIR}/hpx/hpx_fwd.hpp
      ${hpx_SOURCE_DIR}/hpx/util/binary_portable_iarchive.hpp
      ${hpx_SOURCE_DIR}/hpx/util/binary_portable_oarchive.hpp
      ${hpx_SOURCE_DIR}/hpx/util/portable_binary_archive.hpp
      ${hpx_SOURCE_DIR}/hpx/util/portable_binary_iarchive.hpp
      ${hpx_SOURCE_DIR}/hpx/util/portable_binary_oarchive.hpp)
  source_group("Header Files\\serialization" 
      FILES ${hpxserializationlib_HEADERS})
  
  set(hpxserializationlib_SOURCES
      util/binary_portable_archives.cpp
      util/portable_binary_iarchive.cpp
      util/portable_binary_oarchive.cpp)
  source_group("Source Files\\serialization" 
      FILES ${hpxserializationlib_SOURCES})

  # libhpx_serialization
  add_library(hpx_serialization SHARED ${hpxserializationlib_SOURCES}
                                       ${hpxserializationlib_HEADERS})

  set_target_properties(hpx_serialization PROPERTIES
    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib/hpx"
    VERSION ${HPX_VERSION}
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME hpx_serialization)

  get_target_property(path hpx_serialization LOCATION)
  get_filename_component(name "${path}" NAME)
  set_property(TARGET hpx_serialization APPEND
    PROPERTY COMPILE_DEFINITIONS
    "HPX_LIBRARY=\"${name}\""
    "HPX_COMPONENT_NAME=hpx"
    "HPX_EXPORTS"
    "BOOST_COROUTINE_EXPORTS")

  # libhpx
  add_library(hpx SHARED ${hpxlib_SOURCES}
                         ${hpxlib_HEADERS})

  target_link_libraries(hpx hpx_serialization)

  if(HPX_STACKTRACES)
    target_link_libraries(hpx dbghelp)
  endif()

  set_target_properties(hpx PROPERTIES
    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib/hpx"
    VERSION ${HPX_VERSION}
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1
    OUTPUT_NAME hpx)

  get_target_property(path hpx LOCATION)
  get_filename_component(name "${path}" NAME)
  set_property(TARGET hpx APPEND
    PROPERTY COMPILE_DEFINITIONS
    "HPX_LIBRARY=\"${name}\""
    "HPX_COMPONENT_NAME=hpx"
    "HPX_EXPORTS"
    "BOOST_ENABLE_ASSERT_HANDLER"
    "BOOST_COROUTINE_EXPORTS")

  set_property(TARGET hpx 
               PROPERTY LIBRARY_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/lib/hpx")
  set_property(TARGET hpx 
               PROPERTY ARCHIVE_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/lib/hpx")
  set_property(TARGET hpx 
               PROPERTY RUNTIME_OUTPUT_DIRECTORY
               "${CMAKE_BINARY_DIR}/lib/hpx")

  install(TARGETS hpx
      ARCHIVE DESTINATION lib/hpx
      LIBRARY DESTINATION lib/hpx
      COMPONENT core
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                  GROUP_READ GROUP_EXECUTE
                  WORLD_READ WORLD_EXECUTE)

  install(TARGETS hpx_serialization
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      COMPONENT core
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                  GROUP_READ GROUP_EXECUTE
                  WORLD_READ WORLD_EXECUTE)
  
  set_property(TARGET hpx APPEND
    PROPERTY COMPILE_FLAGS
    "-bigobj")
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set_property(TARGET hpx APPEND
      PROPERTY COMPILE_FLAGS
      "-DHPX_EMULATE_SWAP_CONTEXT=1")
  endif()
endif()

