# Copyright (c) 2007-2009 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


###############################################################################
# global
set (hpxlib_global_HEADERS
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
set (hpxlib_config_HEADERS
    ${hpx_SOURCE_DIR}/hpx/config/defaults.hpp
    ${hpx_SOURCE_DIR}/hpx/config/export_definitions.hpp
    ${hpx_SOURCE_DIR}/hpx/config/warnings_prefix.hpp
    ${hpx_SOURCE_DIR}/hpx/config/warnings_suffix.hpp)
source_group("Header Files\\config" 
    FILES ${hpxlib_config_HEADERS})

# include
set (hpxlib_include_HEADERS
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

# LCO's
set (hpxlib_lcos_HEADERS
    ${hpx_SOURCE_DIR}/hpx/lcos/barrier.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/base_lco.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/condition.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/counting_semaphore.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/eager_future.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/eager_future_constructors.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/eager_future_constructors_direct.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/future_callback.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/future_value.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/future_value_full_empty_bit.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/future_value_promise.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/future_wait.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/lazy_future.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/lazy_future_get_results.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/lazy_future_get_results_direct.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/mutex.hpp)
source_group("Header Files\\lcos" 
    FILES ${hpxlib_lcos_HEADERS})

# Performance counters
set (hpxlib_performance_counters_HEADERS
    ${hpx_SOURCE_DIR}/hpx/performance_counters/counters.hpp
    ${hpx_SOURCE_DIR}/hpx/performance_counters/performance_counter.hpp
    ${hpx_SOURCE_DIR}/hpx/performance_counters/registry.hpp)
source_group("Header Files\\performance_counters" 
    FILES ${hpxlib_performance_counters_HEADERS})

set (hpxlib_performance_counters_stubs_HEADERS
    ${hpx_SOURCE_DIR}/hpx/performance_counters/stubs/performance_counter.hpp)
source_group("Header Files\\performance_counters\\stubs" 
    FILES ${hpxlib_performance_counters_stubs_HEADERS})

set (hpxlib_performance_counters_server_HEADERS
    ${hpx_SOURCE_DIR}/hpx/performance_counters/server/base_performance_counter.hpp
    ${hpx_SOURCE_DIR}/hpx/performance_counters/server/raw_counter.hpp)
source_group("Header Files\\performance_counters\\server" 
    FILES ${hpxlib_performance_counters_server_HEADERS})

# runtime
set (hpxlib_runtime_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/get_lva.hpp)
source_group("Header Files\\runtime" 
    FILES ${hpxlib_runtime_HEADERS})

# actions 
set (hpxlib_actions_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_constructors.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_manager.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/action_support.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/component_action.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/component_action_implementations.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/continuation.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/continuation_impl.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/plain_action.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/actions/plain_action_implementations.hpp)
source_group("Header Files\\runtime\\actions" 
    FILES ${hpxlib_actions_HEADERS})

# applier
set (hpxlib_applier_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/applier.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply_helper.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply_helper_implementations.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/applier/apply_implementations.hpp)
source_group("Header Files\\runtime\\applier" 
    FILES ${hpxlib_applier_HEADERS})

# components
set (hpxlib_components_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/components/client_base.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_type.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory_base.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/console_error_sink.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/console_logging.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/derived_component_factory.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/memory_block.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/runtime_support.hpp)
source_group("Header Files\\runtime\\components" 
    FILES ${hpxlib_components_HEADERS})

set (hpxlib_components_server_HEADERS
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

set (hpxlib_components_stubs_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/memory_block.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/runtime_support.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/stubs/stub_base.hpp)
source_group("Header Files\\runtime\\components\\stubs" 
    FILES ${hpxlib_components_stubs_HEADERS})

# naming
set (hpxlib_naming_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/address.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/full_address.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/locality.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/name.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/resolver_client.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/resolver_client_connection.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/resolver_server.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/connection.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/reply.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/request.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/naming/server/request_handler.hpp)
source_group("Header Files\\runtime\\naming" 
    FILES ${hpxlib_naming_HEADERS})

# parcelset
set (hpxlib_parcelset_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcel.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelhandler.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelport.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelport_connection.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/parcelhandler_queue.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/parcelport_queue.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/parcelport_server_connection.hpp)
source_group("Header Files\\parcelset" 
    FILES ${hpxlib_parcelset_HEADERS})

# threadmanager
set (hpxlib_threadmanager_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread_affinity.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/thread_helpers.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/threadmanager.hpp)
source_group("Header Files\\runtime\\threadmanager" 
    FILES ${hpxlib_threadmanager_HEADERS})

set (hpxlib_threadmanager_policies_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/global_queue_scheduler.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/callback_notifier.hpp)
source_group("Header Files\\runtime\\threadmanager\\policies" 
    FILES ${hpxlib_threadmanager_policies_HEADERS})

# util
set (hpxlib_util_HEADERS
    ${hpx_SOURCE_DIR}/hpx/util/asio_util.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_oarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/block_profiler.hpp
    ${hpx_SOURCE_DIR}/hpx/util/connection_cache.hpp
    ${hpx_SOURCE_DIR}/hpx/util/container_device.hpp
    ${hpx_SOURCE_DIR}/hpx/util/full_empty_memory.hpp
    ${hpx_SOURCE_DIR}/hpx/util/full_empty_store.hpp
    ${hpx_SOURCE_DIR}/hpx/util/future.hpp
    ${hpx_SOURCE_DIR}/hpx/util/generate_unique_ids.hpp
    ${hpx_SOURCE_DIR}/hpx/util/high_resolution_timer.hpp
    ${hpx_SOURCE_DIR}/hpx/util/ini.hpp
    ${hpx_SOURCE_DIR}/hpx/util/init_ini_data.hpp
    ${hpx_SOURCE_DIR}/hpx/util/io_service_pool.hpp
    ${hpx_SOURCE_DIR}/hpx/util/logging.hpp
    ${hpx_SOURCE_DIR}/hpx/util/one_size_heap_list.hpp
    ${hpx_SOURCE_DIR}/hpx/util/performance_counters.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_archive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_oarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/runtime_configuration.hpp
    ${hpx_SOURCE_DIR}/hpx/util/safe_bool.hpp
    ${hpx_SOURCE_DIR}/hpx/util/serialize_exception.hpp
    ${hpx_SOURCE_DIR}/hpx/util/serialize_sequence.hpp
    ${hpx_SOURCE_DIR}/hpx/util/static.hpp
    ${hpx_SOURCE_DIR}/hpx/util/time_logger.hpp
    ${hpx_SOURCE_DIR}/hpx/util/unlock_lock.hpp
    ${hpx_SOURCE_DIR}/hpx/util/util.hpp
    ${hpx_SOURCE_DIR}/hpx/util/value_logger.hpp)
source_group("Header Files\\util" 
    FILES ${hpxlib_util_HEADERS})

set (hpxlib_HEADERS 
    ${hpxlib_global_HEADERS}
    ${hpxlib_config_HEADERS}
    ${hpxlib_include_HEADERS}
    ${hpxlib_lcos_HEADERS}
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

###############################################################################
# global
set (hpxlib_global_SOURCES
    ${hpx_SOURCE_DIR}/src/exception_list.cpp
    ${hpx_SOURCE_DIR}/src/runtime.cpp)
source_group ("Source Files" 
    FILES ${hpxlib_global_SOURCES})

# lcos
set (hpxlib_lcos_SOURCES
    ${hpx_SOURCE_DIR}/src/lcos/base_lco.cpp)
source_group ("Source Files\\lcos" 
    FILES ${hpxlib_lcos_SOURCES})

# performance counters
set (hpxlib_performance_counters_SOURCES
    ${hpx_SOURCE_DIR}/src/performance_counters/counters.cpp
    ${hpx_SOURCE_DIR}/src/performance_counters/registry.cpp)
source_group ("Source Files\\performance_counters" 
    FILES ${hpxlib_performance_counters_SOURCES})

set (hpxlib_performance_counters_stubs_SOURCES
    ${hpx_SOURCE_DIR}/src/performance_counters/stubs/performance_counter.cpp)
source_group ("Source Files\\performance_counters\\stubs" 
    FILES ${hpxlib_performance_counters_stubs_SOURCES})

set (hpxlib_performance_counters_server_SOURCES
    ${hpx_SOURCE_DIR}/src/performance_counters/server/raw_counter.cpp)
source_group ("Source Files\\performance_counters\\server" 
    FILES ${hpxlib_performance_counters_server_SOURCES})

# actions 
set (hpxlib_actions_SOURCES
    ${hpx_SOURCE_DIR}/src/runtime/actions/action_manager.cpp
    ${hpx_SOURCE_DIR}/src/runtime/actions/continuation.cpp)
source_group ("Source Files\\runtime\\actions" 
    FILES ${hpxlib_actions_SOURCES})

# applier
set (hpxlib_applier_SOURCES
    ${hpx_SOURCE_DIR}/src/runtime/applier/applier.cpp)
source_group ("Source Files\\runtime\\applier" 
    FILES ${hpxlib_applier_SOURCES})

# components
set (hpxlib_components_SOURCES
    ${hpx_SOURCE_DIR}/src/runtime/components/component_type.cpp)
source_group ("Source Files\\runtime\\components" 
    FILES ${hpxlib_components_SOURCES})

set (hpxlib_components_server_SOURCES
    ${hpx_SOURCE_DIR}/src/runtime/components/server/console_error_sink.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/console_error_sink_singleton.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/console_logging.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/memory.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/memory_block.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/plain_function.cpp
    ${hpx_SOURCE_DIR}/src/runtime/components/server/runtime_support.cpp)
source_group ("Source Files\\runtime\\components\\server" 
    FILES ${hpxlib_components_server_SOURCES})

# naming 
set (hpxlib_naming_SOURCES
    ${hpx_SOURCE_DIR}/src/runtime/naming/address.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/full_address.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/locality.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/name.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/resolver_client.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/resolver_server.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/server/reply.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/server/request.cpp
    ${hpx_SOURCE_DIR}/src/runtime/naming/server/request_handler.cpp)
source_group ("Source Files\\runtime\\naming" 
    FILES ${hpxlib_naming_SOURCES})

# parcelset
set (hpxlib_parcelset_SOURCES
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcel.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelhandler.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelport.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelport_connection.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/server/parcelhandler_queue.cpp)
source_group ("Source Files\\runtime\\parcelset" 
    FILES ${hpxlib_parcelset_SOURCES})

# threadmanager
set (hpxlib_threadmanager_SOURCES
    ${hpx_SOURCE_DIR}/src/runtime/threads/thread.cpp
    ${hpx_SOURCE_DIR}/src/runtime/threads/threadmanager.cpp)
source_group ("Source Files\\runtime\\threadmanager" 
    FILES ${hpxlib_threadmanager_SOURCES})

# util 
set (hpxlib_util_SOURCES
    ${hpx_SOURCE_DIR}/src/util/asio_util.cpp
    ${hpx_SOURCE_DIR}/src/util/ini.cpp
    ${hpx_SOURCE_DIR}/src/util/io_service_pool.cpp
    ${hpx_SOURCE_DIR}/src/util/logging.cpp
    ${hpx_SOURCE_DIR}/src/util/performance_counters.cpp
    ${hpx_SOURCE_DIR}/src/util/runtime_configuration.cpp)
source_group ("Source Files\\util" 
    FILES ${hpxlib_util_SOURCES})

set (hpxlib_SOURCES
    ${hpxlib_global_SOURCES}
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

# add source file from coroutine library (needed for gcc only)
if (CMAKE_COMPILER_IS_GNUCXX)
    SET(hpxlib_SOURCES 
        ${hpxlib_SOURCES} 
        ${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/swapcontext.cpp)
endif()

# Boost Chrono sources (as long as this library is not in Boost)
set (hpxlib_SOURCES 
     ${hpxlib_SOURCES}
     ${hpx_SOURCE_DIR}/external/chrono/libs/chrono/src/chrono.cpp)

# yes, the main shared library contains components as well
add_definitions(-DHPX_COMPONENT_NAME=hpx)
add_definitions(-DHPX_EXPORTS)
add_definitions(-DBOOST_COROUTINE_EXPORTS)

# separate shared module containing our portable serialization archives
set (hpxserializationlib_HEADERS
    ${hpx_SOURCE_DIR}/hpx/hpx_fwd.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_oarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_archive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_oarchive.hpp)

set (hpxserializationlib_SOURCES
    util/binary_portable_archives.cpp
    util/portable_binary_iarchive.cpp
    util/portable_binary_oarchive.cpp)

###############################################################################
# build instructions for serialization support library
add_library(hpx_serialization SHARED 
    ${hpxserializationlib_SOURCES} ${hpxserializationlib_HEADERS})
set_target_properties(hpx_serialization PROPERTIES
    VERSION ${HPX_VERSION}      # create *nix style library versions + symbolic links
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1       # allow creating static and shared libs without conflicts
    OUTPUT_NAME hpx_serialization)
target_link_libraries(hpx_serialization ${Boost_LIBRARIES})

# build instructions for main shared HPX library
add_library(hpx SHARED 
    ${hpxlib_SOURCES} ${hpxlib_HEADERS})
set_target_properties(hpx PROPERTIES
    VERSION ${HPX_VERSION}      # create *nix style library versions + symbolic links
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1       # allow creating static and shared libs without conflicts
    OUTPUT_NAME hpx)
target_link_libraries(hpx hpx_serialization ${Boost_LIBRARIES})

# installation instructions
install(TARGETS hpx hpx_serialization
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

