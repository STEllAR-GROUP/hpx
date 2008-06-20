# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (hpxlib_HEADERS
    # global
    ../hpx/config.hpp
    ../hpx/exception.hpp
    ../hpx/hpx.hpp
    ../hpx/hpx_fwd.hpp
    # components
    ../hpx/components/accumulator.hpp
    ../hpx/components/action.hpp
    ../hpx/components/action_constructors.hpp
    ../hpx/components/action_implementations.hpp
    ../hpx/components/component_type.hpp
    ../hpx/components/continuation.hpp
    ../hpx/components/factory.hpp
    ../hpx/components/server/accumulator.hpp
    ../hpx/components/server/factory.hpp
    ../hpx/components/server/manage_component.hpp
    ../hpx/components/server/wrapper.hpp
    ../hpx/components/server/wrapper_heap.hpp
    ../hpx/components/server/wrapper_heap_list.hpp
    ../hpx/components/stubs/accumulator.hpp
    ../hpx/components/stubs/factory.hpp
    # include
    ../hpx/include/action_manager.hpp
    ../hpx/include/applier.hpp
    ../hpx/include/components.hpp
    ../hpx/include/naming.hpp
    ../hpx/include/parcelset.hpp
    ../hpx/include/runtime.hpp
    ../hpx/include/threadmanager.hpp
    ../hpx/include/util.hpp
    # LCO's
    ../hpx/lcos/base_lco.hpp
    ../hpx/lcos/full_empty_memory.hpp
    ../hpx/lcos/full_empty_store.hpp
    ../hpx/lcos/simple_future.hpp
    # runtime
    ../hpx/runtime/runtime.hpp
    # action manager
    ../hpx/runtime/action_manager/action_manager.hpp
    # applier
    ../hpx/runtime/applier/applier.hpp
    ../hpx/runtime/applier/applier_implementations.hpp
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
    ../hpx/runtime/threadmanager/px_thread.hpp
    ../hpx/runtime/threadmanager/threadmanager.hpp
    # util
    ../hpx/util/asio_util.hpp
    ../hpx/util/container_device.hpp
    ../hpx/util/future.hpp
    ../hpx/util/generate_unique_ids.hpp
    ../hpx/util/high_resolution_timer.hpp
    ../hpx/util/io_service_pool.hpp
    ../hpx/util/logging.hpp
    ../hpx/util/one_size_heap.hpp
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
    # components
    components/server/factory.cpp
    # runtime
    # action manager
    runtime/action_manager/action_manager.cpp
    # applier
    runtime/applier/applier.cpp
    # naming 
    runtime/naming/resolver_client.cpp
    runtime/naming/resolver_server.cpp
    runtime/naming/server/request_handler.cpp
    # parcelset
    runtime/parcelset/connection_cache.cpp
    runtime/parcelset/parcel.cpp
    runtime/parcelset/parcelhandler.cpp
    runtime/parcelset/parcelport.cpp
    runtime/parcelset/server/parcelhandler_queue.cpp
    runtime/parcelset/server/parcelport_queue.cpp
    # threadmanager
    runtime/threadmanager/threadmanager.cpp
    # util 
    util/io_service_pool.cpp
    util/logging.cpp
)

add_library (hpx STATIC ${hpxlib_SOURCES} ${hpxlib_HEADERS})

