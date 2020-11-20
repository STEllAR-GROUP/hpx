//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS) && defined(HPX_HAVE_NETWORKING)

#include <hpx/hashing/jenkins_hash.hpp>
#include <hpx/runtime/parcelset/detail/data_point.hpp>
#include <hpx/runtime/parcelset/detail/gatherer.hpp>
#include <hpx/synchronization/spinlock.hpp>

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

