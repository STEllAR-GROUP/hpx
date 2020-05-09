//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCEL_COALESCING)

#include <hpx/functional/function.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/preprocessor/nargs.hpp>
#include <hpx/runtime/parcelset/policies/message_handler.hpp>
#include <hpx/statistics/histogram.hpp>
#include <hpx/util/pool_timer.hpp>

#include <hpx/plugins/parcel/message_buffer.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <hpx/config/warnings_prefix.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    struct HPX_LIBRARY_EXPORT coalescing_message_handler
      : parcelset::policies::message_handler
    {
    private:
        coalescing_message_handler* this_() { return this; }

        typedef lcos::local::spinlock mutex_type;

    public:
        typedef parcelset::policies::message_handler::write_handler_type
            write_handler_type;

        coalescing_message_handler(char const* action_name,
            parcelset::parcelport* pp, std::size_t num = std::size_t(-1),
            std::size_t interval = std::size_t(-1));

        void put_parcel(parcelset::locality const & dest,
            parcelset::parcel p, write_handler_type f);

        bool flush(parcelset::policies::message_handler::flush_mode mode,
            bool stop_buffering = false);

        void flush_terminate();

        // access performance counter data
        std::int64_t get_parcels_count(bool reset);
        std::int64_t get_messages_count(bool reset);
        std::int64_t get_parcels_per_message_count(bool reset);
        std::int64_t get_average_time_between_parcels(bool reset);
        std::vector<std::int64_t>
            get_time_between_parcels_histogram(bool reset);
        void get_time_between_parcels_histogram_creator(
            std::int64_t min_boundary, std::int64_t max_boundary,
            std::int64_t num_buckets,
            util::function_nonser<std::vector<std::int64_t>(bool)>& result);

        // register the given action
        static void register_action(char const* action, error_code& ec);

    protected:
        bool timer_flush();
        bool flush_locked(std::unique_lock<mutex_type>& l,
            parcelset::policies::message_handler::flush_mode mode,
            bool stop_buffering, bool cancel_timer);

        void update_num_messages();
        void update_interval();

    private:
        mutable mutex_type mtx_;
        parcelset::parcelport* pp_;
        std::size_t num_coalesced_parcels_;
        std::size_t interval_;
        detail::message_buffer buffer_;
        util::pool_timer timer_;
        bool stopped_;
        bool allow_background_flush_;
        std::string action_name_;

        // performance counter data
        std::int64_t num_parcels_;
        std::int64_t reset_num_parcels_;
        std::int64_t reset_num_parcels_per_message_parcels_;
        std::int64_t num_messages_;
        std::int64_t reset_num_messages_;
        std::int64_t reset_num_parcels_per_message_messages_;
        std::int64_t started_at_;
        std::int64_t reset_time_num_parcels_;
        std::int64_t last_parcel_time_;

        typedef boost::accumulators::accumulator_set<
                double,     // collects percentiles
                boost::accumulators::features<hpx::util::tag::histogram>
            > histogram_collector_type;

        std::unique_ptr<histogram_collector_type> time_between_parcels_;
        std::int64_t histogram_min_boundary_;
        std::int64_t histogram_max_boundary_;
        std::int64_t histogram_num_buckets_;
    };
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
