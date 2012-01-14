//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/util/full_empty_entry.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/runtime/actions/continuation.hpp>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

namespace hpx { namespace util { namespace detail
{
    // the counter data instance
    full_empty_counter_data full_empty_counter_data_;

    boost::int64_t get_constructed_count()
    {
        lcos::local_spinlock::scoped_lock l(full_empty_counter_data_.mtx_);
        return full_empty_counter_data_.constructed_;
    }

    boost::int64_t get_destructed_count()
    {
        lcos::local_spinlock::scoped_lock l(full_empty_counter_data_.mtx_);
        return full_empty_counter_data_.destructed_;
    }

    boost::int64_t get_read_enqueued_count()
    {
        lcos::local_spinlock::scoped_lock l(full_empty_counter_data_.mtx_);
        return full_empty_counter_data_.read_enqueued_;
    }

    boost::int64_t get_read_dequeued_count()
    {
        lcos::local_spinlock::scoped_lock l(full_empty_counter_data_.mtx_);
        return full_empty_counter_data_.read_dequeued_;
    }

    boost::int64_t get_set_full_count()
    {
        lcos::local_spinlock::scoped_lock l(full_empty_counter_data_.mtx_);
        return full_empty_counter_data_.set_full_;
    }

    // call this to install all counters for full_empty entries
    void install_counters()
    {
        performance_counters::raw_counter_type_data const counter_types[] =
        {
            { "/lcos/full_empty/constructed", performance_counters::counter_raw,
              "returns the number of constructed full_empty entries",
              HPX_PERFORMANCE_COUNTER_V1 },
            { "/lcos/full_empty/destructed", performance_counters::counter_raw,
              "returns the number of destructed full_empty entries",
              HPX_PERFORMANCE_COUNTER_V1 },
            { "/lcos/full_empty/read_enqueued", performance_counters::counter_raw,
              "returns the number of full_empty 'read' enqueue operations",
              HPX_PERFORMANCE_COUNTER_V1 },
            { "/lcos/full_empty/read_dequeued", performance_counters::counter_raw,
              "returns the number of full_empty 'read' dequeue operations",
              HPX_PERFORMANCE_COUNTER_V1 },
            { "/lcos/full_empty/set_full", performance_counters::counter_raw,
              "returns the number of full_empty 'set' operations",
              HPX_PERFORMANCE_COUNTER_V1 }
        };

        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));

        boost::uint32_t const prefix = applier::get_applier().get_prefix_id();
        boost::format full_empty_counter("/lcos(locality#%d/total)/full_empty/%s");

        performance_counters::raw_counter_data const counters[] =
        {
            { boost::str(full_empty_counter % prefix % "constructed"),
              get_constructed_count },
            { boost::str(full_empty_counter % prefix % "destructed"),
              get_destructed_count },
            { boost::str(full_empty_counter % prefix % "read_enqueued"),
              get_read_enqueued_count },
            { boost::str(full_empty_counter % prefix % "read_dequeued"),
              get_read_dequeued_count },
            { boost::str(full_empty_counter % prefix % "set_full"),
              get_set_full_count }
        };

        performance_counters::install_counters(
            counters, sizeof(counters)/sizeof(counters[0]));
    }
}}}
