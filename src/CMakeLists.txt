# Copyright (c) 2007-2011 Hartmut Kaiser
# Copyright (c)      2011 Bryce Lelbach
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

###############################################################################
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

# LCO's
set(hpxlib_lcos_HEADERS
    ${hpx_SOURCE_DIR}/hpx/lcos/barrier.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/stubs/barrier.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/server/barrier.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/base_lco.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/condition.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/contin.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/counting_semaphore.hpp
    ${hpx_SOURCE_DIR}/hpx/lcos/dataflow_variable.hpp
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
    ${hpx_SOURCE_DIR}/hpx/lcos/thunk_constructors.hpp
    )
source_group("Header Files\\lcos" 
    FILES ${hpxlib_lcos_HEADERS})

# Performance counters
set(hpxlib_performance_counters_HEADERS
    ${hpx_SOURCE_DIR}/hpx/performance_counters/counters.hpp
    ${hpx_SOURCE_DIR}/hpx/performance_counters/performance_counter.hpp
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

# nedmalloc
set(hpxlib_nedmalloc_HEADERS
    ${hpx_SOURCE_DIR}/hpx/util/nedmalloc_allocator.hpp
    ${hpx_SOURCE_DIR}/external/nedmalloc/hpx_nedmalloc/dlmalloc.h
    ${hpx_SOURCE_DIR}/external/nedmalloc/hpx_nedmalloc/nedmalloc.h)
source_group("Header Files\\memory\\nedmalloc" 
    FILES ${hpxlib_nedmalloc_HEADERS})

# jemalloc
set(hpxlib_jemalloc_HEADERS
    ${hpx_SOURCE_DIR}/hpx/util/jemalloc_allocator.hpp
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/arena.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/base.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/chunk.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/chunk_dss.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/chunk_mmap.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/chunk_swap.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/ckh.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/ctl.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/extent.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/hash.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/huge.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/jemalloc.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/jemalloc_defs.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/jemalloc_internal.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/mb.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/mutex.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/random.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/prof.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/ql.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/qr.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/rb.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/rtree.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/stats.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/tcache.h
    ${hpx_SOURCE_DIR}/external/jemalloc/hpx_jemalloc/zone.h)     
source_group("Header Files\\memory\\jemalloc" 
    FILES ${hpxlib_jemalloc_HEADERS})

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
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory_base.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_factory_one.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_registry_base.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/component_registry.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/constructor_argument.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/console_error_sink.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/console_logging.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/derived_component_factory.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/memory_block.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/plain_component_factory.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/unique_component_name.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/components/runtime_support.hpp)
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
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelport.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/parcelport_connection.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/parcelset/server/parcelhandler_queue.hpp
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
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/threadmanager_impl.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/detail/tagged_thread_state.hpp)
source_group("Header Files\\runtime\\threads" 
    FILES ${hpxlib_threadmanager_HEADERS})

set(hpxlib_threadmanager_policies_HEADERS
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/callback_notifier.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/global_queue_scheduler.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/local_queue_scheduler.hpp
    ${hpx_SOURCE_DIR}/hpx/runtime/threads/policies/thread_queue.hpp)
source_group("Header Files\\runtime\\threads\\policies" 
    FILES ${hpxlib_threadmanager_policies_HEADERS})

# util
#set(hpxutillib_HEADERS
set(hpxlib_util_HEADERS
    ${hpx_SOURCE_DIR}/hpx/util/asio_util.hpp
    ${hpx_SOURCE_DIR}/hpx/util/allocator.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_oarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/block_profiler.hpp
    ${hpx_SOURCE_DIR}/hpx/util/connection_cache.hpp
    ${hpx_SOURCE_DIR}/hpx/util/container_device.hpp
    ${hpx_SOURCE_DIR}/hpx/util/filesystem_compatibility.hpp
    ${hpx_SOURCE_DIR}/hpx/util/full_empty_memory.hpp
    ${hpx_SOURCE_DIR}/hpx/util/full_empty_store.hpp
    ${hpx_SOURCE_DIR}/hpx/util/future.hpp
    ${hpx_SOURCE_DIR}/hpx/util/generate_unique_ids.hpp
    ${hpx_SOURCE_DIR}/hpx/util/high_resolution_timer.hpp
    ${hpx_SOURCE_DIR}/hpx/util/ini.hpp
    ${hpx_SOURCE_DIR}/hpx/util/init_ini_data.hpp
    ${hpx_SOURCE_DIR}/hpx/util/itt_notify.hpp
    ${hpx_SOURCE_DIR}/hpx/util/io_service_pool.hpp
    ${hpx_SOURCE_DIR}/hpx/util/locking_helpers.hpp
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
    ${hpx_SOURCE_DIR}/hpx/util/spinlock_pool.hpp
    ${hpx_SOURCE_DIR}/hpx/util/static.hpp
    ${hpx_SOURCE_DIR}/hpx/util/system_allocator.hpp
    ${hpx_SOURCE_DIR}/hpx/util/time_logger.hpp
    ${hpx_SOURCE_DIR}/hpx/util/unlock_lock.hpp
    ${hpx_SOURCE_DIR}/hpx/util/util.hpp
    ${hpx_SOURCE_DIR}/hpx/util/value_logger.hpp)
source_group("Header Files\\util" 
#    FILES ${hpxutillib_HEADERS})
    FILES ${hpxlib_util_HEADERS})

set(hpxlib_HEADERS 
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
#    ${hpxutillib_HEADERS})
    ${hpxlib_util_HEADERS})

set(hpxmalloclib_HEADERS)
if("${HPX_MALLOC}" STREQUAL "jemalloc")
  set(hpxmalloclib_HEADERS ${hpxlib_jemalloc_HEADERS})
  list(APPEND hpxlib_HEADERS
       ${hpxlib_jemalloc_HEADERS}
       ${hpx_SOURCE_DIR}/hpx/util/jemalloc_allocator.hpp)
elseif("${HPX_MALLOC}" STREQUAL "nedmalloc")
  set(hpxmalloclib_HEADERS ${hpxlib_nedmalloc_HEADERS})
  list(APPEND hpxlib_HEADERS
       ${hpxlib_nedmalloc_HEADERS}
       ${hpx_SOURCE_DIR}/hpx/util/nedmalloc_allocator.hpp)
endif()

###############################################################################
# global
set(hpxlib_global_SOURCES
    ${hpx_SOURCE_DIR}/src/exception.cpp
    ${hpx_SOURCE_DIR}/src/exception_list.cpp
    ${hpx_SOURCE_DIR}/src/runtime.cpp)
source_group("Source Files" 
    FILES ${hpxlib_global_SOURCES})

# lcos
set(hpxlib_lcos_SOURCES
    ${hpx_SOURCE_DIR}/src/lcos/base_lco.cpp)
source_group("Source Files\\lcos" 
    FILES ${hpxlib_lcos_SOURCES})

# performance counters
set(hpxlib_performance_counters_SOURCES
    ${hpx_SOURCE_DIR}/src/performance_counters/counters.cpp
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

# applier
set(hpxlib_applier_SOURCES
    ${hpx_SOURCE_DIR}/src/runtime/applier/applier.cpp)
source_group("Source Files\\runtime\\applier" 
    FILES ${hpxlib_applier_SOURCES})

# components
set(hpxlib_components_SOURCES
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
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelhandler.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelport.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/parcelport_connection.cpp
    ${hpx_SOURCE_DIR}/src/runtime/parcelset/server/parcelhandler_queue.cpp)
source_group("Source Files\\runtime\\parcelset" 
    FILES ${hpxlib_parcelset_SOURCES})

# threadmanager
set(hpxlib_threadmanager_SOURCES
    ${hpx_SOURCE_DIR}/src/runtime/threads/thread.cpp
    ${hpx_SOURCE_DIR}/src/runtime/threads/threadmanager.cpp)
source_group("Source Files\\runtime\\threads" 
    FILES ${hpxlib_threadmanager_SOURCES})
 
# nedmalloc
set(hpxlib_nedmalloc_SOURCES
    ${hpx_SOURCE_DIR}/external/nedmalloc/src/nedmalloc.c)
source_group("Source Files\\memory\\nedmalloc" 
    FILES ${hpxlib_nedmalloc_SOURCES})

# jemalloc
set(hpxlib_jemalloc_SOURCES
    ${hpx_SOURCE_DIR}/external/jemalloc/src/arena.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/base.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/chunk.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/chunk_dss.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/chunk_mmap.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/chunk_swap.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/ckh.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/ctl.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/extent.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/hash.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/huge.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/jemalloc.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/mb.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/mutex.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/prof.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/rtree.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/stats.c
    ${hpx_SOURCE_DIR}/external/jemalloc/src/tcache.c)

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
    list(APPEND hpxlib_jemalloc_SOURCES
        ${hpx_SOURCE_DIR}/external/jemalloc/src/zone.c)
endif()

source_group("Source Files\\memory\\jemalloc" 
    FILES ${hpxlib_jemalloc_SOURCES})

# util 
#set(hpxutillib_SOURCES
set(hpxlib_util_SOURCES
    ${hpx_SOURCE_DIR}/src/util/asio_util.cpp
    ${hpx_SOURCE_DIR}/src/util/ini.cpp
    ${hpx_SOURCE_DIR}/src/util/init_ini_data.cpp
    ${hpx_SOURCE_DIR}/src/util/itt_notify.cpp
    ${hpx_SOURCE_DIR}/src/util/io_service_pool.cpp
    ${hpx_SOURCE_DIR}/src/util/logging.cpp
    ${hpx_SOURCE_DIR}/src/util/performance_counters.cpp
    ${hpx_SOURCE_DIR}/src/util/runtime_configuration.cpp)
source_group("Source Files\\util" 
#    FILES ${hpxutillib_SOURCES})
    FILES ${hpxlib_util_SOURCES})

set(hpxlib_SOURCES
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
    # TODO: remove this when hpx_util is split into it's own shared library
    ${hpxlib_util_SOURCES})

set(hpxmalloclib_SOURCES)
if("${HPX_MALLOC}" STREQUAL "jemalloc")
  set(hpxmalloclib_SOURCES ${hpxlib_jemalloc_SOURCES})
  list(APPEND hpxlib_SOURCES ${hpxlib_jemalloc_SOURCES})
elseif("${HPX_MALLOC}" STREQUAL "nedmalloc")
  set(hpxmalloclib_SOURCES ${hpxlib_nedmalloc_SOURCES})
  list(APPEND hpxlib_SOURCES ${hpxlib_nedmalloc_SOURCES})
endif()

# add source file from coroutine library 
if(NOT MSVC)
    set(hpxlib_SOURCES 
        ${hpxlib_SOURCES} 
        ${hpx_SOURCE_DIR}/external/coroutine/libs/coroutine/src/swapcontext.cpp)
endif()

# Boost Chrono sources (as long as this library is not in Boost)
if(HPX_INTERNAL_CHRONO AND ${BOOST_MINOR_VERSION} LESS 47)
    set(hpxlib_SOURCES 
        ${hpxlib_SOURCES}
        ${hpx_SOURCE_DIR}/external/chrono/libs/chrono/src/chrono.cpp)
endif()

# yes, the main shared library contains components as well
add_definitions(-DHPX_COMPONENT_NAME=hpx)
add_definitions(-DHPX_EXPORTS)
add_definitions(-DBOOST_COROUTINE_EXPORTS)

# separate shared module containing our portable serialization archives
set(hpxserializationlib_HEADERS
    ${hpx_SOURCE_DIR}/hpx/hpx_fwd.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/binary_portable_oarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_archive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_iarchive.hpp
    ${hpx_SOURCE_DIR}/hpx/util/portable_binary_oarchive.hpp)

set(hpxserializationlib_SOURCES
    util/binary_portable_archives.cpp
    util/portable_binary_iarchive.cpp
    util/portable_binary_oarchive.cpp)

###############################################################################
# build instructions for serialization support library
add_library(hpx_serialization SHARED 
    ${hpxserializationlib_SOURCES} ${hpxserializationlib_HEADERS})
set_target_properties(hpx_serialization PROPERTIES
    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
    VERSION ${HPX_VERSION}      # create *nix style library versions + symbolic links
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1       # allow creating static and shared libs without conflicts
    OUTPUT_NAME hpx_serialization)
target_link_libraries(hpx_serialization ${BOOST_FOUND_LIBRARIES})

###############################################################################
# build instructions for utility library
#add_library(hpx_util SHARED 
#    ${hpxutillib_SOURCES} ${hpxutillib_HEADERS})
#set_target_properties(hpx_util PROPERTIES
#    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
#    VERSION ${HPX_VERSION}      # create *nix style library versions + symbolic links
#    SOVERSION ${HPX_SOVERSION}
#    CLEAN_DIRECT_OUTPUT 1       # allow creating static and shared libs without conflicts
#    OUTPUT_NAME hpx_util)
#target_link_libraries(hpx_util ${BOOST_FOUND_LIBRARIES})

###############################################################################
# build instructions for the optional malloc preload shared object
if(HPX_MALLOC_PRELOAD_SO)
  add_library(hpx_malloc SHARED
      ${hpxmalloclib_SOURCES} ${hpxmalloclib_HEADERS})
  set_target_properties(hpx_malloc PROPERTIES
      INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
      VERSION ${HPX_VERSION}      # create *nix style library versions + symbolic links
      SOVERSION ${HPX_SOVERSION}
      CLEAN_DIRECT_OUTPUT 1       # allow creating static and shared libs without conflicts
      OUTPUT_NAME hpx_malloc)
  set_property(TARGET hpx_malloc APPEND
               PROPERTY COMPILE_DEFINITIONS "HPX_MALLOC_PRELOAD_SO")
  install(TARGETS hpx_malloc 
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
      COMPONENT core
      PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                  GROUP_READ GROUP_EXECUTE
                  WORLD_READ WORLD_EXECUTE)
endif()

# build instructions for main shared HPX library
add_library(hpx SHARED 
    ${hpxlib_SOURCES} ${hpxlib_HEADERS})
set_target_properties(hpx PROPERTIES
    INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib"
    VERSION ${HPX_VERSION}      # create *nix style library versions + symbolic links
    SOVERSION ${HPX_SOVERSION}
    CLEAN_DIRECT_OUTPUT 1       # allow creating static and shared libs without conflicts
    OUTPUT_NAME hpx)
#target_link_libraries(hpx hpx_util hpx_serialization ${BOOST_FOUND_LIBRARIES})
target_link_libraries(hpx hpx_serialization ${BOOST_FOUND_LIBRARIES})

# installation instructions
#install(TARGETS hpx hpx_util hpx_serialization
install(TARGETS hpx hpx_serialization
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    COMPONENT core
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE)

