//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/format.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/parcelset/parcelhandler.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/manage_counter_type.hpp>
#include <hpx/performance_counters/parcelhandler_counter_types.hpp>

#include <cstdint>
#include <string>

namespace hpx::performance_counters {

    ///////////////////////////////////////////////////////////////////////////
    void register_parcelhandler_counter_types(
        parcelset::parcelhandler& ph, std::string const& pp_type)
    {
        if (!ph.is_networking_enabled())
        {
            return;
        }

        using placeholders::_1;
        using placeholders::_2;

        using parcelset::parcelhandler;

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        hpx::function<std::int64_t(std::string const&, bool)> num_parcel_sends(
            hpx::bind_front(
                &parcelhandler::get_action_parcel_send_count, &ph, pp_type));
        hpx::function<std::int64_t(std::string const&, bool)>
            num_parcel_receives(hpx::bind_front(
                &parcelhandler::get_action_parcel_receive_count, &ph, pp_type));
#else
        hpx::function<std::int64_t(bool)> num_parcel_sends(hpx::bind_front(
            &parcelhandler::get_parcel_send_count, &ph, pp_type));
        hpx::function<std::int64_t(bool)> num_parcel_receives(hpx::bind_front(
            &parcelhandler::get_parcel_receive_count, &ph, pp_type));
#endif

        hpx::function<std::int64_t(bool)> num_message_sends(hpx::bind_front(
            &parcelhandler::get_message_send_count, &ph, pp_type));
        hpx::function<std::int64_t(bool)> num_message_receives(hpx::bind_front(
            &parcelhandler::get_message_receive_count, &ph, pp_type));

        hpx::function<std::int64_t(bool)> sending_time(
            hpx::bind_front(&parcelhandler::get_sending_time, &ph, pp_type));
        hpx::function<std::int64_t(bool)> receiving_time(
            hpx::bind_front(&parcelhandler::get_receiving_time, &ph, pp_type));

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        hpx::function<std::int64_t(std::string const&, bool)>
            sending_serialization_time(hpx::bind_front(
                &parcelhandler::get_action_sending_serialization_time, &ph,
                pp_type));
        hpx::function<std::int64_t(std::string const&, bool)>
            receiving_serialization_time(hpx::bind_front(
                &parcelhandler::get_action_receiving_serialization_time, &ph,
                pp_type));
#else
        hpx::function<std::int64_t(bool)> sending_serialization_time(
            hpx::bind_front(
                &parcelhandler::get_sending_serialization_time, &ph, pp_type));
        hpx::function<std::int64_t(bool)> receiving_serialization_time(
            hpx::bind_front(&parcelhandler::get_receiving_serialization_time,
                &ph, pp_type));
#endif

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
        hpx::function<std::int64_t(std::string const&, bool)> data_sent(
            hpx::bind_front(
                &parcelhandler::get_action_data_sent, &ph, pp_type));
        hpx::function<std::int64_t(std::string const&, bool)> data_received(
            hpx::bind_front(
                &parcelhandler::get_action_data_received, &ph, pp_type));
#else
        hpx::function<std::int64_t(bool)> data_sent(
            hpx::bind_front(&parcelhandler::get_data_sent, &ph, pp_type));
        hpx::function<std::int64_t(bool)> data_received(
            hpx::bind_front(&parcelhandler::get_data_received, &ph, pp_type));
#endif

        hpx::function<std::int64_t(bool)> data_raw_sent(
            hpx::bind_front(&parcelhandler::get_raw_data_sent, &ph, pp_type));
        hpx::function<std::int64_t(bool)> data_raw_received(hpx::bind_front(
            &parcelhandler::get_raw_data_received, &ph, pp_type));

        hpx::function<std::int64_t(bool)> buffer_allocate_time_sent(
            hpx::bind_front(
                &parcelhandler::get_buffer_allocate_time_sent, &ph, pp_type));
        hpx::function<std::int64_t(bool)> buffer_allocate_time_received(
            hpx::bind_front(&parcelhandler::get_buffer_allocate_time_received,
                &ph, pp_type));

        performance_counters::generic_counter_type_data const counter_types[] =
            {
                {hpx::util::format("/parcels/count/{}/sent", pp_type),
                    performance_counters::counter_monotonically_increasing,
                    hpx::util::format(
                        "returns the number of parcels sent using the {} "
                        "connection type for the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                    hpx::bind(
                        &performance_counters::per_action_data_counter_creator,
                        _1, HPX_MOVE(num_parcel_sends), _2),
                    &performance_counters::per_action_data_counter_discoverer,
#else
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(num_parcel_sends), _2),
                    &performance_counters::locality_counter_discoverer,
#endif
                    ""},
                {hpx::util::format("/parcels/count/{}/received", pp_type),
                    performance_counters::counter_monotonically_increasing,
                    hpx::util::format(
                        "returns the number of parcels received using the {} "
                        "connection type for the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                    hpx::bind(
                        &performance_counters::per_action_data_counter_creator,
                        _1, HPX_MOVE(num_parcel_receives), _2),
                    &performance_counters::per_action_data_counter_discoverer,
#else
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(num_parcel_receives), _2),
                    &performance_counters::locality_counter_discoverer,
#endif
                    ""},
                {hpx::util::format("/messages/count/{}/sent", pp_type),
                    performance_counters::counter_monotonically_increasing,
                    hpx::util::format(
                        "returns the number of messages sent using the {} "
                        "connection type for the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(num_message_sends), _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {hpx::util::format("/messages/count/{}/received", pp_type),
                    performance_counters::counter_monotonically_increasing,
                    hpx::util::format(
                        "returns the number of messages received using the {} "
                        "connection type for the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(num_message_receives), _2),
                    &performance_counters::locality_counter_discoverer, ""},

                {hpx::util::format("/data/time/{}/sent", pp_type),
                    performance_counters::counter_elapsed_time,
                    hpx::util::format(
                        "returns the total time between the start of each "
                        "asynchronous write and the invocation of the write "
                        "callback using the {} connection type for the "
                        "referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(sending_time), _2),
                    &performance_counters::locality_counter_discoverer, "ns"},
                {hpx::util::format("/data/time/{}/received", pp_type),
                    performance_counters::counter_elapsed_time,
                    hpx::util::format(
                        "returns the total time between the start of each "
                        "asynchronous read and the invocation of the read "
                        "callback using the {} connection type for the "
                        "referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(receiving_time), _2),
                    &performance_counters::locality_counter_discoverer, "ns"},
                {hpx::util::format("/serialize/time/{}/sent", pp_type),
                    performance_counters::counter_elapsed_time,
                    hpx::util::format(
                        "returns the total time required to serialize all sent "
                        "parcels using the {} connection type for the "
                        "referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                    hpx::bind(
                        &performance_counters::per_action_data_counter_creator,
                        _1, HPX_MOVE(sending_serialization_time), _2),
                    &performance_counters::per_action_data_counter_discoverer,
#else
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(sending_serialization_time), _2),
                    &performance_counters::locality_counter_discoverer,
#endif
                    "ns"},
                {hpx::util::format("/serialize/time/{}/received", pp_type),
                    performance_counters::counter_elapsed_time,
                    hpx::util::format(
                        "returns the total time required to de-serialize all "
                        "received parcels using the {} connection type for the "
                        "referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                    hpx::bind(
                        &performance_counters::per_action_data_counter_creator,
                        _1, HPX_MOVE(receiving_serialization_time), _2),
                    &performance_counters::per_action_data_counter_discoverer,
#else
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(receiving_serialization_time), _2),
                    &performance_counters::locality_counter_discoverer,
#endif
                    "ns"},

                {hpx::util::format("/data/count/{}/sent", pp_type),
                    performance_counters::counter_monotonically_increasing,
                    hpx::util::format(
                        "returns the amount of (uncompressed) parcel argument "
                        "data sent using the {} connection type by the "
                        "referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(data_raw_sent), _2),
                    &performance_counters::locality_counter_discoverer,
                    "bytes"},
                {hpx::util::format("/data/count/{}/received", pp_type),
                    performance_counters::counter_monotonically_increasing,
                    hpx::util::format(
                        "returns the amount of (uncompressed) parcel argument "
                        "data received using the {} connection type by the "
                        "referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(data_raw_received), _2),
                    &performance_counters::locality_counter_discoverer,
                    "bytes"},
                {hpx::util::format("/serialize/count/{}/sent", pp_type),
                    performance_counters::counter_monotonically_increasing,
                    hpx::util::format(
                        "returns the amount of parcel data (including headers, "
                        "possibly compressed) sent using the {} connection "
                        "type by the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                    hpx::bind(
                        &performance_counters::per_action_data_counter_creator,
                        _1, HPX_MOVE(data_sent), _2),
                    &performance_counters::per_action_data_counter_discoverer,
#else
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(data_sent), _2),
                    &performance_counters::locality_counter_discoverer,
#endif
                    "bytes"},
                {hpx::util::format("/serialize/count/{}/received", pp_type),
                    performance_counters::counter_monotonically_increasing,
                    hpx::util::format(
                        "returns the amount of parcel data (including headers, "
                        "possibly compressed) received using the {} connection "
                        "type by the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                    hpx::bind(
                        &performance_counters::per_action_data_counter_creator,
                        _1, HPX_MOVE(data_received), _2),
                    &performance_counters::per_action_data_counter_discoverer,
#else
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(data_received), _2),
                    &performance_counters::locality_counter_discoverer,
#endif
                    "bytes"},
                {hpx::util::format(
                     "/parcels/time/{}/buffer_allocate/received", pp_type),
                    performance_counters::counter_elapsed_time,
                    hpx::util::format(
                        "returns the time needed to allocate the buffers for "
                        "serializing using the {} connection type",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(buffer_allocate_time_received), _2),
                    &performance_counters::locality_counter_discoverer, "ns"},
                {hpx::util::format(
                     "/parcels/time/{}/buffer_allocate/sent", pp_type),
                    performance_counters::counter_elapsed_time,
                    hpx::util::format(
                        "returns the time needed to allocate the buffers for "
                        "serializing using the {} connection type",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(buffer_allocate_time_sent), _2),
                    &performance_counters::locality_counter_discoverer, "ns"},
            };

        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types) / sizeof(counter_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    // register connection specific performance counters related to connection
    // caches
    void register_connection_cache_counter_types(
        parcelset::parcelhandler& ph, std::string const& pp_type)
    {
        if (!ph.is_networking_enabled())
        {
            return;
        }

        using hpx::placeholders::_1;
        using hpx::placeholders::_2;

        using parcelset::parcelhandler;
        using parcelset::parcelport;

        hpx::function<std::int64_t(bool)> cache_insertions(
            hpx::bind_front(&parcelhandler::get_connection_cache_statistics,
                &ph, pp_type, parcelport::connection_cache_insertions));
        hpx::function<std::int64_t(bool)> cache_evictions(
            hpx::bind_front(&parcelhandler::get_connection_cache_statistics,
                &ph, pp_type, parcelport::connection_cache_evictions));
        hpx::function<std::int64_t(bool)> cache_hits(
            hpx::bind_front(&parcelhandler::get_connection_cache_statistics,
                &ph, pp_type, parcelport::connection_cache_hits));
        hpx::function<std::int64_t(bool)> cache_misses(
            hpx::bind_front(&parcelhandler::get_connection_cache_statistics,
                &ph, pp_type, parcelport::connection_cache_misses));
        hpx::function<std::int64_t(bool)> cache_reclaims(
            hpx::bind_front(&parcelhandler::get_connection_cache_statistics,
                &ph, pp_type, parcelport::connection_cache_reclaims));

        performance_counters::generic_counter_type_data const
            connection_cache_types[] = {
                {hpx::util::format(
                     "/parcelport/count/{}/cache-insertions", pp_type),
                    performance_counters::counter_raw,
                    hpx::util::format(
                        "returns the number of cache insertions while "
                        "accessing the connection cache for the {} connection "
                        "type on the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(cache_insertions), _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {hpx::util::format(
                     "/parcelport/count/{}/cache-evictions", pp_type),
                    performance_counters::counter_raw,
                    hpx::util::format(
                        "returns the number of cache evictions while accessing "
                        "the connection cache for the {} connection type on "
                        "the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(cache_evictions), _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {hpx::util::format("/parcelport/count/{}/cache-hits", pp_type),
                    performance_counters::counter_raw,
                    hpx::util::format(
                        "returns the number of cache hits while accessing the "
                        "connection cache for the {} connection type on the "
                        "referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(cache_hits), _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {hpx::util::format(
                     "/parcelport/count/{}/cache-misses", pp_type),
                    performance_counters::counter_raw,
                    hpx::util::format(
                        "returns the number of cache misses while accessing "
                        "the connection cache for the {} connection type on "
                        "the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(cache_misses), _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {hpx::util::format(
                     "/parcelport/count/{}/cache-reclaims", pp_type),
                    performance_counters::counter_raw,
                    hpx::util::format(
                        "returns the number of cache reclaims while accessing "
                        "the connection cache for the {} connection type on "
                        "the referenced locality",
                        pp_type),
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        HPX_MOVE(cache_reclaims), _2),
                    &performance_counters::locality_counter_discoverer, ""}};

        performance_counters::install_counter_types(connection_cache_types,
            sizeof(connection_cache_types) / sizeof(connection_cache_types[0]));
    }

    ///////////////////////////////////////////////////////////////////////////
    void register_parcelhandler_counter_types(parcelset::parcelhandler& ph)
    {
        // register connection specific counters
        ph.enum_parcelports([&](std::string const& type) -> bool {
            register_parcelhandler_counter_types(ph, type);
            register_connection_cache_counter_types(ph, type);
            return true;
        });

        using placeholders::_1;
        using placeholders::_2;

        using parcelset::parcelhandler;

        // register common counters
        hpx::function<std::int64_t(bool)> incoming_queue_length(
            hpx::bind_front(&parcelhandler::get_incoming_queue_length, &ph));
        hpx::function<std::int64_t(bool)> outgoing_queue_length(
            hpx::bind_front(&parcelhandler::get_outgoing_queue_length, &ph));
        hpx::function<std::int64_t(bool)> outgoing_routed_count(
            hpx::bind_front(&parcelhandler::get_parcel_routed_count, &ph));

        performance_counters::generic_counter_type_data const counter_types[] =
            {{"/parcelqueue/length/receive", performance_counters::counter_raw,
                 "returns the number current length of the queue of incoming "
                 "parcels",
                 HPX_PERFORMANCE_COUNTER_V1,
                 hpx::bind(&performance_counters::locality_raw_counter_creator,
                     _1, incoming_queue_length, _2),
                 &performance_counters::locality_counter_discoverer, ""},
                {"/parcelqueue/length/send", performance_counters::counter_raw,
                    "returns the number current length of the queue of "
                    "outgoing parcels",
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        outgoing_queue_length, _2),
                    &performance_counters::locality_counter_discoverer, ""},
                {"/parcels/count/routed",
                    performance_counters::counter_monotonically_increasing,
                    "returns the number of (outbound) parcel routed through "
                    "the responsible AGAS service",
                    HPX_PERFORMANCE_COUNTER_V1,
                    hpx::bind(
                        &performance_counters::locality_raw_counter_creator, _1,
                        outgoing_routed_count, _2),
                    &performance_counters::locality_counter_discoverer, ""}};

        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types) / sizeof(counter_types[0]));
    }
}    // namespace hpx::performance_counters

#endif
