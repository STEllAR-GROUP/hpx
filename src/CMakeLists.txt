# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (hpxlib_HEADERS
    # global
    ../hpx/config.hpp
    ../hpx/defaults.hpp
    ../hpx/exception.hpp
    ../hpx/hpx.hpp
    ../hpx/hpx_fwd.hpp
    ../hpx/version.hpp
    # config
    ../hpx/config/defaults.hpp
    ../hpx/config/export_definitions.hpp
    ../hpx/config/warnings_prefix.hpp
    ../hpx/config/warnings_suffix.hpp
    # include
    ../hpx/include/actions.hpp
    ../hpx/include/applier.hpp
    ../hpx/include/components.hpp
    ../hpx/include/lcos.hpp
    ../hpx/include/naming.hpp
    ../hpx/include/parcelset.hpp
    ../hpx/include/runtime.hpp
    ../hpx/include/threadmanager.hpp
    ../hpx/include/util.hpp
    # LCO's
    ../hpx/lcos/barrier.hpp
    ../hpx/lcos/base_lco.hpp
    ../hpx/lcos/condition.hpp
    ../hpx/lcos/eager_future.hpp
    ../hpx/lcos/eager_future_constructors.hpp
    ../hpx/lcos/eager_future_constructors_direct.hpp
    ../hpx/lcos/lazy_future.hpp
    ../hpx/lcos/lazy_future_get_results.hpp
    ../hpx/lcos/lazy_future_get_results_direct.hpp
    ../hpx/lcos/mutex.hpp
    ../hpx/lcos/simple_future.hpp
    # runtime
    ../hpx/runtime/get_lva.hpp
    ../hpx/runtime/runtime.hpp
    # actions 
    ../hpx/runtime/actions/action.hpp
    ../hpx/runtime/actions/action_constructors.hpp
    ../hpx/runtime/actions/action_implementations.hpp
    ../hpx/runtime/actions/continuation.hpp
    ../hpx/runtime/actions/continuation_impl.hpp
    ../hpx/runtime/actions/action_manager.hpp
    # applier
    ../hpx/runtime/applier/applier.hpp
    ../hpx/runtime/applier/applier_implementations.hpp
    ../hpx/runtime/applier/apply_helper.hpp
    ../hpx/runtime/applier/apply_helper_implementations.hpp
    # components
    ../hpx/runtime/components/component_type.hpp
    ../hpx/runtime/components/component_factory_base.hpp
    ../hpx/runtime/components/component_factory.hpp
    ../hpx/runtime/components/runtime_support.hpp
    ../hpx/runtime/components/server/manage_component.hpp
    ../hpx/runtime/components/server/managed_component_base.hpp
    ../hpx/runtime/components/server/memory.hpp
    ../hpx/runtime/components/server/runtime_support.hpp
    ../hpx/runtime/components/server/simple_component_base.hpp
    ../hpx/runtime/components/server/wrapper_heap.hpp
    ../hpx/runtime/components/server/wrapper_heap_list.hpp
    ../hpx/runtime/components/stubs/runtime_support.hpp
    # naming
    ../hpx/runtime/naming/address.hpp
    ../hpx/runtime/naming/locality.hpp
    ../hpx/runtime/naming/name.hpp
    ../hpx/runtime/naming/resolver_client.hpp
    ../hpx/runtime/naming/resolver_client_connection.hpp
    ../hpx/runtime/naming/resolver_server.hpp
    ../hpx/runtime/naming/server/connection.hpp
    ../hpx/runtime/naming/server/reply.hpp
    ../hpx/runtime/naming/server/request.hpp
    ../hpx/runtime/naming/server/request_handler.hpp
    # parcelset
    ../hpx/runtime/parcelset/connection_cache.hpp
    ../hpx/runtime/parcelset/parcel.hpp
    ../hpx/runtime/parcelset/parcelhandler.hpp
    ../hpx/runtime/parcelset/parcelport.hpp
    ../hpx/runtime/parcelset/parcelport_connection.hpp
    ../hpx/runtime/parcelset/server/parcelhandler_queue.hpp
    ../hpx/runtime/parcelset/server/parcelport_queue.hpp
    ../hpx/runtime/parcelset/server/parcelport_server_connection.hpp
    # threadmanager
    ../hpx/runtime/threads/thread.hpp
    ../hpx/runtime/threads/threadmanager.hpp
    # util
    ../hpx/util/asio_util.hpp
    ../hpx/util/container_device.hpp
    ../hpx/util/find_msb.hpp
    ../hpx/util/full_empty_memory.hpp
    ../hpx/util/full_empty_store.hpp
    ../hpx/util/future.hpp
    ../hpx/util/generate_unique_ids.hpp
    ../hpx/util/high_resolution_timer.hpp
    ../hpx/util/ini.hpp
    ../hpx/util/init_ini_data.hpp
    ../hpx/util/io_service_pool.hpp
    ../hpx/util/logging.hpp
    ../hpx/util/one_size_heap_list.hpp
    ../hpx/util/portable_binary_iarchive.hpp
    ../hpx/util/portable_binary_oarchive.hpp
    ../hpx/util/safe_bool.hpp
    ../hpx/util/serialize_sequence.hpp
    ../hpx/util/static.hpp
    ../hpx/util/unlock_lock.hpp
    ../hpx/util/util.hpp
)

set (hpxlib_SOURCES
    # lcos
    lcos/base_lco.cpp
    # runtime
    runtime/runtime.cpp
    # action manager
    runtime/actions/action_manager.cpp
    runtime/actions/continuation.cpp
    # applier
    runtime/applier/applier.cpp
    # components
    runtime/components/server/memory.cpp
    runtime/components/server/runtime_support.cpp
    # naming 
    runtime/naming/address.cpp
    runtime/naming/locality.cpp
    runtime/naming/name.cpp
    runtime/naming/resolver_client.cpp
    runtime/naming/resolver_server.cpp
    runtime/naming/server/reply.cpp
    runtime/naming/server/request.cpp
    runtime/naming/server/request_handler.cpp
    # parcelset
    runtime/parcelset/connection_cache.cpp
    runtime/parcelset/parcel.cpp
    runtime/parcelset/parcelhandler.cpp
    runtime/parcelset/parcelport.cpp
    runtime/parcelset/server/parcelhandler_queue.cpp
    runtime/parcelset/server/parcelport_queue.cpp
    # threadmanager
    runtime/threads/threadmanager.cpp
    # util 
    util/ini.cpp
    util/io_service_pool.cpp
    util/logging.cpp
    # source file from coroutine library
    ../external/coroutine/libs/coroutine/src/swapcontext.cpp
)

add_definitions(-DHPX_EXPORTS)
add_library (hpx SHARED ${hpxlib_SOURCES} ${hpxlib_HEADERS})

