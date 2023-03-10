//  Copyright (c) 2007-2021 Hartmut Kaiser
//  Copyright (c) 2014-2021 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING)
#include <hpx/assert.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/functional.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/modules/runtime_local.hpp>
#include <hpx/modules/serialization.hpp>
#include <hpx/modules/timing.hpp>

#include <hpx/components_base/agas_interface.hpp>
#include <hpx/naming_base/id_type.hpp>
#include <hpx/parcelset_base/detail/data_point.hpp>
#include <hpx/parcelset_base/detail/parcel_route_handler.hpp>
#include <hpx/parcelset_base/parcel_interface.hpp>

#if ASIO_HAS_BOOST_THROW_EXCEPTION != 0
#include <boost/exception/exception.hpp>
#endif

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <system_error>
#include <utility>
#include <vector>

namespace hpx::parcelset {

    template <typename Buffer>
    std::vector<serialization::serialization_chunk> decode_chunks(
        Buffer& buffer)
    {
        using transmission_chunk_type =
            typename Buffer::transmission_chunk_type;

        std::vector<serialization::serialization_chunk> chunks;

        std::size_t num_zero_copy_chunks = static_cast<std::size_t>(
            static_cast<std::uint32_t>(buffer.num_chunks_.first));
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
        HPX_ASSERT(num_zero_copy_chunks == buffer.chunks_.size());
        parcelset::data_point& data = buffer.data_point_;
        data.num_zchunks_ += buffer.chunks_.size();
        data.num_zchunks_per_msg_max_ =
            (std::max)(data.num_zchunks_per_msg_max_,
                (std::int64_t) buffer.chunks_.size());
        for (auto& chunk : buffer.chunks_)
        {
            data.size_zchunks_total_ += chunk.size();
            data.size_zchunks_max_ =
                (std::max)(data.size_zchunks_max_, (std::int64_t) chunk.size());
        }
#endif

        if (num_zero_copy_chunks != 0)
        {
            // decode chunk information
            std::size_t num_non_zero_copy_chunks = static_cast<std::size_t>(
                static_cast<std::uint32_t>(buffer.num_chunks_.second));

            chunks.resize(num_zero_copy_chunks + num_non_zero_copy_chunks);

            // place the zero-copy chunks at their spots first
            for (std::size_t i = 0; i != num_zero_copy_chunks; ++i)
            {
                transmission_chunk_type& c = buffer.transmission_chunks_[i];
                std::size_t first = static_cast<std::size_t>(
                    static_cast<std::uint64_t>(c.first));
                std::size_t second = static_cast<std::size_t>(
                    static_cast<std::uint64_t>(c.second));

                HPX_ASSERT(buffer.chunks_[i].size() == second);

                chunks[first] = serialization::create_pointer_chunk(
                    buffer.chunks_[i].data(), second);
            }

            std::size_t index = 0;
            for (std::size_t i = num_zero_copy_chunks;
                 i != num_zero_copy_chunks + num_non_zero_copy_chunks; ++i)
            {
                transmission_chunk_type& c = buffer.transmission_chunks_[i];
                std::size_t first = static_cast<std::size_t>(
                    static_cast<std::uint64_t>(c.first));
                std::size_t second = static_cast<std::size_t>(
                    static_cast<std::uint64_t>(c.second));

                // find next free entry
                while (chunks[index].size_ != 0)
                    ++index;

                // place the index based chunk at the right spot
                chunks[index] =
                    serialization::create_index_chunk(first, second);
                ++index;
            }
#if defined(HPX_DEBUG)
            // make sure that all spots have been populated
            for (std::size_t i = 0;
                 i != num_zero_copy_chunks + num_non_zero_copy_chunks; ++i)
            {
                HPX_ASSERT(chunks[i].size_ != 0);
            }
#endif
        }

        return chunks;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Parcelport, typename Buffer>
    void decode_message_with_chunks(Parcelport& pp, Buffer buffer,
        std::size_t parcel_count,
        std::vector<serialization::serialization_chunk>& chunks,
        std::size_t num_thread = -1)
    {
        std::size_t inbound_data_size = static_cast<std::size_t>(
            static_cast<std::uint64_t>(buffer.data_size_));

        // protect from unhandled exceptions bubbling up
        try
        {
            try
            {
                // mark start of serialization
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
                hpx::chrono::high_resolution_timer timer;
                std::int64_t overall_add_parcel_time = 0;
                parcelset::data_point& data = buffer.data_point_;
#endif
                {
                    std::vector<parcelset::parcel> deferred_parcels;
                    // De-serialize the parcel data
                    serialization::input_archive archive(
                        buffer.data_, inbound_data_size, &chunks);

                    if (parcel_count == 0)
                    {
                        archive >> parcel_count;    //-V128
                    }
                    if (parcel_count > 1)
                    {
                        deferred_parcels.reserve(parcel_count);
                    }

                    for (std::size_t i = 0; i != parcel_count; ++i)
                    {
                        bool deferred_schedule = parcel_count > 1;

#if defined(HPX_HAVE_PARCELPORT_COUNTERS) &&                                   \
    defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                        std::size_t archive_pos = archive.current_pos();
                        std::int64_t serialize_time =
                            timer.elapsed_nanoseconds();
#endif
                        // de-serialize parcel and add it to incoming parcel queue
                        parcelset::parcel p;

                        // deferred_schedule will be set to false if the action
                        // to be loaded is a non direct action. If we only got
                        // one parcel to decode, deferred_schedule will be
                        // preset to false and the direct action will be called
                        // directly
                        bool migrated = p.load_schedule(
                            archive, num_thread, deferred_schedule);

#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
                        std::int64_t add_parcel_time =
                            timer.elapsed_nanoseconds();
#endif

#if defined(HPX_HAVE_PARCELPORT_COUNTERS) &&                                   \
    defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                        parcelset::data_point action_data;
                        action_data.bytes_ =
                            archive.current_pos() - archive_pos;
                        action_data.serialization_time_ =
                            add_parcel_time - serialize_time;
                        action_data.num_parcels_ = 1;
                        pp.add_received_data(p.get_action_name(), action_data);
#endif
                        // make sure this parcel ended up on the right locality
                        std::uint32_t here = agas::get_locality_id();
                        if (hpx::get_runtime_ptr() &&
                            here != naming::invalid_locality_id &&
                            (naming::get_locality_id_from_gid(
                                 p.destination_locality()) != here))
                        {
                            HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                                "hpx::parcelset::decode_message",
                                "parcel destination does not match locality "
                                "which received the parcel ({}), {}",
                                here, p);
                            return;
                        }

                        if (migrated)
                        {
                            agas::route(HPX_MOVE(p),
                                &parcelset::detail::parcel_route_handler,
                                threads::thread_priority::normal);
                        }
                        else if (deferred_schedule)
                        {
                            // If we got a direct action
                            deferred_parcels.emplace_back(HPX_MOVE(p));
                        }

                        // be sure not to measure add_parcel as serialization time
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
                        overall_add_parcel_time +=
                            timer.elapsed_nanoseconds() - add_parcel_time;
#endif
                    }

                    // complete received data with parcel count
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
                    data.num_parcels_ = parcel_count;
                    data.raw_bytes_ = archive.bytes_read();
#endif
                    if (!deferred_parcels.empty())
                    {
                        for (std::size_t i = 1; i != deferred_parcels.size();
                             ++i)
                        {
                            auto f = [num_thread](parcelset::parcel&& p) {
                                if (p.schedule_action(num_thread))
                                {
                                    // route this parcel as the object
                                    // was migrated
                                    agas::route(HPX_MOVE(p),
                                        &parcelset::detail::
                                            parcel_route_handler,
                                        threads::thread_priority::normal);
                                }
                            };

                            // schedule all but the first parcel on a new thread.
                            hpx::threads::thread_init_data data(
                                hpx::threads::make_thread_function_nullary(
                                    util::deferred_call(HPX_MOVE(f),
                                        HPX_MOVE(deferred_parcels[i]))),
                                "schedule_parcel",
                                threads::thread_priority::boost,
                                threads::thread_schedule_hint(
                                    static_cast<std::int16_t>(num_thread)),
                                threads::thread_stacksize::default_,
                                threads::thread_schedule_state::pending, true);
                            hpx::threads::register_thread(data);
                        }

                        // If we are the first deferred parcel, we don't need to spin
                        // up a new thread...
                        if (deferred_parcels[0].schedule_action(num_thread))
                        {
                            // route this parcel as the object was migrated
                            agas::route(HPX_MOVE(deferred_parcels[0]),
                                &parcelset::detail::parcel_route_handler,
                                threads::thread_priority::normal);
                        }
                    }
                }

                // store the time required for serialization
#if defined(HPX_HAVE_PARCELPORT_COUNTERS)
                data.serialization_time_ =
                    timer.elapsed_nanoseconds() - overall_add_parcel_time;
                pp.add_received_data(data);
#else
                HPX_UNUSED(pp);
#endif
            }
            catch (hpx::exception const& e)
            {
                LPT_(error).format(
                    "decode_message: caught hpx::exception: {}", e.what());
                hpx::report_error(std::current_exception());
            }
            catch (std::system_error const& e)
            {
                LPT_(error).format(
                    "decode_message: caught std::system_error: {}", e.what());
                hpx::report_error(std::current_exception());
            }
#if ASIO_HAS_BOOST_THROW_EXCEPTION != 0
            catch (boost::exception const&)
            {
                LPT_(error).format("decode_message: caught boost::exception.");
                hpx::report_error(std::current_exception());
            }
#endif
            catch (std::exception const& e)
            {
                // We have to repackage all exceptions thrown by the
                // serialization library as otherwise we will loose the
                // e.what() description of the problem, due to slicing.
                hpx::throw_with_info(
                    hpx::exception(hpx::error::serialization_error, e.what()));
            }
        }
        catch (...)
        {
            LPT_(error).format("decode_message: caught unknown exception.");
            hpx::report_error(std::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Parcelport, typename Buffer>
    void decode_message(Parcelport& pp, Buffer buffer, std::size_t parcel_count,
        std::size_t num_thread = -1)
    {
        std::vector<serialization::serialization_chunk> chunks(
            decode_chunks(buffer));
        decode_message_with_chunks(
            pp, HPX_MOVE(buffer), parcel_count, chunks, num_thread);
    }

    template <typename Parcelport, typename Buffer>
    void decode_parcel(
        Parcelport& parcelport, Buffer buffer, std::size_t num_thread)
    {
        decode_message(parcelport, HPX_MOVE(buffer), 1, num_thread);
    }

    template <typename Parcelport, typename Buffer>
    void decode_parcels(
        Parcelport& parcelport, Buffer buffer, std::size_t num_thread)
    {
        decode_message(parcelport, HPX_MOVE(buffer), 0, num_thread);
    }
}    // namespace hpx::parcelset

#endif
