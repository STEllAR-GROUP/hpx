# Copyright (c) 2007-2008 Hartmut Kaiser
#
# Distributed under the Boost Software License, Version 1.0. (See accompanying 
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

set (hpxlib_HEADERS
    # global
    ../hpx/components.hpp
    ../hpx/config.hpp
    ../hpx/exception.hpp
    ../hpx/hpx.hpp
    ../hpx/hpx_fwd.hpp
    ../hpx/naming.hpp
    ../hpx/parcelset.hpp
    ../hpx/threadmanager.hpp
    ../hpx/util.hpp
    # action manager
    ../hpx/action_manager/action_manager.hpp
    # applier
    ../hpx/applier/applier.hpp
    # components
    ../hpx/components/action.hpp
    ../hpx/components/action_constructors.hpp
    ../hpx/components/action_implementations.hpp
    ../hpx/components/component_type.hpp
    ../hpx/components/accumulator.hpp
    ../hpx/components/factory.hpp
    ../hpx/components/server/accumulator.hpp
    ../hpx/components/server/factory.hpp
    ../hpx/components/server/manage_component.hpp
    ../hpx/components/server/manage_component_implementations.hpp
    # naming
    ../hpx/naming/address.hpp
    ../hpx/naming/locality.hpp
    ../hpx/naming/name.hpp
    ../hpx/naming/namespace.hpp
    ../hpx/naming/resolver_client.hpp
    ../hpx/naming/resolver_client_connection.hpp
    ../hpx/naming/resolver_server.hpp
    ../hpx/naming/server/connection.hpp
    ../hpx/naming/server/reply.hpp
    ../hpx/naming/server/request.hpp
    ../hpx/naming/server/request_handler.hpp
    # parcelset
    ../hpx/parcelset/connection_cache.hpp
    ../hpx/parcelset/parcel.hpp
    ../hpx/parcelset/parcelhandler.hpp
    ../hpx/parcelset/parcelport.hpp
    ../hpx/parcelset/parcelport_connection.hpp
    ../hpx/parcelset/server/parcelhandler_queue.hpp
    ../hpx/parcelset/server/parcelport_queue.hpp
    ../hpx/parcelset/server/parcelport_server_connection.hpp
    # threadmanager
    ../hpx/threadmanager/px_thread.hpp
    ../hpx/threadmanager/threadmanager.hpp
    # util
    ../hpx/util/asio_util.hpp
    ../hpx/util/container_device.hpp
    ../hpx/util/logging.hpp
    ../hpx/util/future.hpp
    ../hpx/util/generate_unique_ids.hpp
    ../hpx/util/high_resolution_timer.hpp
    ../hpx/util/io_service_pool.hpp
    ../hpx/util/one_size_heap.hpp
    ../hpx/util/one_size_heap_list.hpp
    ../hpx/util/portable_binary_iarchive.hpp
    ../hpx/util/portable_binary_oarchive.hpp
    ../hpx/util/safe_bool.hpp
    ../hpx/util/serialize_sequence.hpp
    ../hpx/util/util.hpp
)

set (hpxlib_SOURCES
    # components
    components/server/factory.cpp
    # naming 
    naming/resolver_client.cpp
    naming/resolver_server.cpp
    naming/server/request_handler.cpp
    # parcelset
    parcelset/connection_cache.cpp
    parcelset/parcel.cpp
    parcelset/parcelhandler.cpp
    parcelset/parcelport.cpp
    parcelset/server/parcelhandler_queue.cpp
    parcelset/server/parcelport_queue.cpp
    # threadmanager
    threadmanager/threadmanager.cpp
    # util 
    util/logging.cpp
    util/io_service_pool.cpp
)

add_library (hpx STATIC ${hpxlib_SOURCES} ${hpxlib_HEADERS})

