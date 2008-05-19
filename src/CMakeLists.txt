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
    ../hpx/naming.hpp
    ../hpx/parcelset.hpp
    ../hpx/util.hpp
    # components
    ../hpx/components/action.hpp
    ../hpx/components/component_type.hpp
    ../hpx/components/accumulator.hpp
    ../hpx/components/server/accumulator.hpp
    # naming
    ../hpx/naming/address.hpp
    ../hpx/naming/locality.hpp
    ../hpx/naming/name.hpp
    ../hpx/naming/namespace.hpp
    ../hpx/naming/resolver_client.hpp
    ../hpx/naming/resolver_server.hpp
    # parcelset
    ../hpx/parcelset/parcel.hpp
    ../hpx/parcelset/parcelport.hpp
    ../hpx/parcelset/parcelhandler.hpp
    ../hpx/parcelset/server/parcelport_connection.hpp
    ../hpx/parcelset/server/parcelhandler_queue.hpp
    ../hpx/parcelset/server/parcelport_queue.hpp
    # util
    ../hpx/util/asio_util.hpp
    ../hpx/util/container_device.hpp
    ../hpx/util/dgas_logging.hpp
    ../hpx/util/generate_unique_ids.hpp
    ../hpx/util/high_resolution_timer.hpp
    ../hpx/util/io_service_pool.hpp
    ../hpx/util/portable_binary_iarchive.hpp
    ../hpx/util/portable_binary_oarchive.hpp
    ../hpx/util/safe_bool.hpp
    ../hpx/util/serialize_sequence.hpp
    ../hpx/util/util.hpp
)

set (hpxlib_SOURCES
    # naming 
    naming/resolver_client.cpp
    naming/resolver_server.cpp
    naming/server/request_handler.cpp
    # parcelset
    parcelset/parcel.cpp
    parcelset/parcelhandler.cpp
    parcelset/parcelport.cpp
    parcelset/server/parcelhandler_queue.cpp
    parcelset/server/parcelport_queue.cpp
    # util 
    util/dgas_logging.cpp
    util/io_service_pool.cpp
)

add_library (hpx STATIC ${hpxlib_SOURCES} ${hpxlib_HEADERS})

