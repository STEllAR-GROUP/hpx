//  Copyright (c) 2007-2013 Hartmut Kaiser
//  Copyright (c) 2014-2015 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARCELSET_DECODE_PARCELS_HPP
#define HPX_PARCELSET_DECODE_PARCELS_HPP

#include <hpx/config.hpp>
#include <hpx/exception.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/runtime/naming/name.hpp>
#include <hpx/runtime/naming/resolver_client.hpp>
#include <hpx/runtime/parcelset/parcel.hpp>
#include <hpx/runtime/parcelset/detail/parcel_route_handler.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/logging.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <utility>
#include <vector>

namespace hpx { namespace parcelset
{
    template <typename Buffer>
    std::vector<serialization::serialization_chunk> decode_chunks(Buffer & buffer)
    {
        typedef typename Buffer::transmission_chunk_type transmission_chunk_type;

        std::vector<serialization::serialization_chunk> chunks;

        std::size_t num_zero_copy_chunks =
            static_cast<std::size_t>(
                static_cast<std::uint32_t>(buffer.num_chunks_.first));

        if (num_zero_copy_chunks != 0)
        {
            // decode chunk information
            std::size_t num_non_zero_copy_chunks =
                static_cast<std::size_t>(
                    static_cast<std::uint32_t>(buffer.num_chunks_.second));

            chunks.resize(num_zero_copy_chunks + num_non_zero_copy_chunks);

            // place the zero-copy chunks at their spots first
            for (std::size_t i = 0; i != num_zero_copy_chunks; ++i)
            {
                transmission_chunk_type& c = buffer.transmission_chunks_[i];
                std::uint64_t first = c.first, second = c.second;

                HPX_ASSERT(buffer.chunks_[i].size() == second);

                chunks[first] = serialization::create_pointer_chunk(
                        buffer.chunks_[i].data(), second);
            }

            std::size_t index = 0;
            for (std::size_t i = num_zero_copy_chunks;
                 i != num_zero_copy_chunks + num_non_zero_copy_chunks;
                 ++i)
            {
                transmission_chunk_type& c = buffer.transmission_chunks_[i];
                std::uint64_t first = c.first, second = c.second;

                // find next free entry
                while (chunks[index].size_ != 0)
                    ++index;

                // place the index based chunk at the right spot
                chunks[index] = serialization::create_index_chunk(first, second);
                ++index;
            }
#if defined(HPX_DEBUG)
            // make sure that all spots have been populated
            for (std::size_t i = 0;
                 i != num_zero_copy_chunks + num_non_zero_copy_chunks;
                 ++i)
            {
                HPX_ASSERT(chunks[i].size_ != 0);
            }
#endif
        }

        return chunks;
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Parcelport, typename Buffer>
    void decode_message_with_chunks(
        Parcelport & pp
      , Buffer buffer
      , std::size_t parcel_count
      , std::vector<serialization::serialization_chunk> &chunks
      , std::size_t num_thread = -1
    )
    {
        std::uint64_t inbound_data_size = buffer.data_size_;

        // protect from un-handled exceptions bubbling up
        try {
            try {
                // mark start of serialization
                util::high_resolution_timer timer;
                std::int64_t overall_add_parcel_time = 0;
                performance_counters::parcels::data_point& data =
                    buffer.data_point_;

                {
                    std::vector<parcel> deferred_parcels;
                    // De-serialize the parcel data
                    serialization::input_archive archive(buffer.data_,
                        inbound_data_size, &chunks);

                    if(parcel_count == 0)
                    {
                        archive >> parcel_count; //-V128
                    }
                    if (parcel_count > 1)
                        deferred_parcels.reserve(parcel_count);
                    for(std::size_t i = 0; i != parcel_count; ++i)
                    {
                        bool deferred_schedule = parcel_count > 1;

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                        std::size_t archive_pos = archive.current_pos();
                        std::int64_t serialize_time = timer.elapsed_nanoseconds();
#endif
                        // de-serialize parcel and add it to incoming parcel queue
                        parcel p;
                        // deferred_schedule will be set to false if the action
                        // to be loaded is a non direct action. If we only got
                        // one parcel to decode, deferred_schedule will be
                        // preset to false and the direct action will be called
                        // directly
                        bool migrated = p.load_schedule(archive, num_thread,
                            deferred_schedule);

                        std::int64_t add_parcel_time = timer.elapsed_nanoseconds();

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
                        performance_counters::parcels::data_point action_data;
                        action_data.bytes_ = archive.current_pos() - archive_pos;
                        action_data.serialization_time_ =
                            add_parcel_time - serialize_time;
                        action_data.num_parcels_ = 1;
                        pp.add_received_data(p.get_action()->get_action_name(),
                            action_data);
#endif

                        // make sure this parcel ended up on the right locality
                        naming::gid_type const& here = hpx::get_locality();
                        if (hpx::get_runtime_ptr() && here &&
                            (naming::get_locality_id_from_gid(
                                 p.destination_locality()) !=
                             naming::get_locality_id_from_gid(here)))
                        {
                            std::ostringstream os;
                            os << "parcel destination does not match "
                                  "locality which received the parcel ("
                               << here << "), " << p;
                            HPX_THROW_EXCEPTION(invalid_status,
                                "hpx::parcelset::decode_message",
                                os.str());
                            return;
                        }

                        if (migrated)
                        {
                            naming::resolver_client& client =
                                hpx::naming::get_agas_client();
                            client.route(
                                std::move(p),
                                &detail::parcel_route_handler,
                                threads::thread_priority_normal);
                        }
                        // If we got a direct action,
                        else if (deferred_schedule)
                            deferred_parcels.push_back(std::move(p));

                        // be sure not to measure add_parcel as serialization time
                        overall_add_parcel_time += timer.elapsed_nanoseconds() -
                            add_parcel_time;
                    }

                    // complete received data with parcel count
                    data.num_parcels_ = parcel_count;
                    data.raw_bytes_ = archive.bytes_read();

                    if (!deferred_parcels.empty())
                    {
                        for (std::size_t i = 1; i != deferred_parcels.size(); ++i)
                        {
                            // schedule all but the first parcel on a new thread.
                            hpx::applier::register_thread_nullary(
                                util::bind(
                                    util::one_shot(
                                        [num_thread](parcel&& p)
                                        {
                                            p.schedule_action(num_thread);
                                        }
                                    ), std::move(deferred_parcels[i])),
                                "schedule_parcel",
                                threads::pending, true, threads::thread_priority_boost,
                                num_thread, threads::thread_stacksize_default);
                        }
                        // If we are the first deferred parcel, we don't need to spin
                        // a new thread...
                        deferred_parcels[0].schedule_action(num_thread);
                    }
                }

                // store the time required for serialization
                data.serialization_time_ = timer.elapsed_nanoseconds() -
                    overall_add_parcel_time;

                pp.add_received_data(data);
            }
            catch (hpx::exception const& e) {
                LPT_(error)
                    << "decode_message: caught hpx::exception: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::system::system_error const& e) {
                LPT_(error)
                    << "decode_message: caught boost::system::error: "
                    << e.what();
                hpx::report_error(boost::current_exception());
            }
            catch (boost::exception const&) {
                LPT_(error)
                    << "decode_message: caught boost::exception.";
                hpx::report_error(boost::current_exception());
            }
            catch (std::exception const& e) {
                // We have to repackage all exceptions thrown by the
                // serialization library as otherwise we will loose the
                // e.what() description of the problem, due to slicing.
                boost::throw_exception(boost::enable_error_info(
                    hpx::exception(serialization_error, e.what())));
            }
        }
        catch (...) {
            LPT_(error)
                << "decode_message: caught unknown exception.";
            hpx::report_error(boost::current_exception());
        }
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Parcelport, typename Buffer>
    void decode_message(
        Parcelport & pp
      , Buffer buffer
      , std::size_t parcel_count
      , std::size_t num_thread = -1
    )
    {
        std::vector<serialization::serialization_chunk>
            chunks(decode_chunks(buffer));
        decode_message_with_chunks(pp, std::move(buffer),
            parcel_count, chunks, num_thread);
    }

    template <typename Parcelport, typename Buffer>
    void decode_parcel(Parcelport & parcelport, Buffer buffer, std::size_t num_thread)
    {
//         if(hpx::is_running() && parcelport.async_serialization())
//         {
//             hpx::applier::register_thread_nullary(
//                 util::bind(
//                     util::one_shot(&decode_message<Parcelport, Buffer>),
//                     std::ref(parcelport), std::move(buffer), 1, num_thread),
//                 "decode_parcels",
//                 threads::pending, true, threads::thread_priority_boost,
//                 parcelport.get_next_num_thread());
//         }
//         else
        {
            decode_message(parcelport, std::move(buffer), 1, num_thread);
        }
    }

    template <typename Parcelport, typename Buffer>
    void decode_parcels(Parcelport & parcelport, Buffer buffer, std::size_t num_thread)
    {
//         if(hpx::is_running() && parcelport.async_serialization())
//         {
//             hpx::applier::register_thread_nullary(
//                 util::bind(
//                     util::one_shot(&decode_message<Parcelport, Buffer>),
//                     std::ref(parcelport), std::move(buffer), 0, num_thread),
//                 "decode_parcels",
//                 threads::pending, true, threads::thread_priority_boost,
//                 parcelport.get_next_num_thread());
//         }
//         else
        {
            decode_message(parcelport, std::move(buffer), 0, num_thread);
        }
    }

}}

#endif
