//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/lcos/detail/full_empty_entry.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/get_and_reset_value.hpp>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

namespace hpx { namespace lcos { namespace detail
{
    // the counter data instance
    full_empty_counter_data full_empty_counter_data_;

    boost::int64_t get_constructed_count(bool reset)
    {
        return util::get_and_reset_value(
            full_empty_counter_data_.constructed_, reset);
    }

    boost::int64_t get_destructed_count(bool reset)
    {
        return util::get_and_reset_value(
            full_empty_counter_data_.destructed_, reset);
    }

    boost::int64_t get_read_enqueued_count(bool reset)
    {
        return util::get_and_reset_value(
            full_empty_counter_data_.read_enqueued_, reset);
    }

    boost::int64_t get_read_dequeued_count(bool reset)
    {
        return util::get_and_reset_value(
            full_empty_counter_data_.read_dequeued_, reset);
    }

    boost::int64_t get_set_full_count(bool reset)
    {
        return util::get_and_reset_value(
            full_empty_counter_data_.set_full_, reset);
    }

    // call this to register all counter types for full_empty entries
    void register_counter_types()
    {
        performance_counters::generic_counter_type_data const counter_types[] =
        {
            { "/full_empty/count/constructed", performance_counters::counter_raw,
              "returns the number of constructed full_empty entries",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, get_constructed_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/full_empty/count/destructed", performance_counters::counter_raw,
              "returns the number of destructed full_empty entries",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, get_destructed_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/full_empty/count/read_enqueued", performance_counters::counter_raw,
              "returns the number of full_empty 'read' enqueue operations",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, get_read_enqueued_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/full_empty/count/read_dequeued", performance_counters::counter_raw,
              "returns the number of full_empty 'read' dequeue operations",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, get_read_dequeued_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/full_empty/count/fired", performance_counters::counter_raw,
              "returns the number of full_empty 'set' operations",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, get_set_full_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }
}}}
