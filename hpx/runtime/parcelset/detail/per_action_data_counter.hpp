//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARCELSET_PER_ACTION_COUNTER_DATA_AUG_04_2016_0540PM)
#define HPX_PARCELSET_PER_ACTION_COUNTER_DATA_AUG_04_2016_0540PM

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)

#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/performance_counters/parcels/data_point.hpp>
#include <hpx/performance_counters/parcels/gatherer.hpp>
#include <hpx/util/jenkins_hash.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace hpx { namespace parcelset { namespace detail
{
    // Per-action based parcel statistics
    struct per_action_data_counter
    {
        typedef hpx::lcos::local::spinlock mutex_type;

        // add collected data
        void add_data(char const* action,
            performance_counters::parcels::data_point const& data);

        // retrieve counter data

        // number of parcels handled
        std::int64_t num_parcels(
            std::string const& action, bool reset);

        // the total time serialization took (nanoseconds)
        std::int64_t total_serialization_time(
            std::string const& action, bool reset);

        // total data managed (bytes)
        std::int64_t total_bytes(
            std::string const& action, bool reset);

    private:
        typedef std::unordered_map<
                std::string, performance_counters::parcels::gatherer_nolock,
                hpx::util::jenkins_hash
            > counter_data_map;

        mutable mutex_type mtx_;
        counter_data_map data_;
    };
}}}

#endif
#endif

