//  Copyright (c) 2016-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_COUNTERS) &&                                   \
    defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS) &&                            \
    defined(HPX_HAVE_NETWORKING)
#include <hpx/modules/hashing.hpp>
#include <hpx/modules/synchronization.hpp>

#include <hpx/parcelset_base/detail/data_point.hpp>
#include <hpx/parcelset_base/detail/gatherer.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace hpx::parcelset::detail {

    // Per-action based parcel statistics
    struct per_action_data_counter
    {
        using mutex_type = hpx::spinlock;

        // add collected data
        void add_data(char const* action, parcelset::data_point const& data);

        // retrieve counter data

        // number of parcels handled
        std::int64_t num_parcels(std::string const& action, bool reset);

        // the total time serialization took (nanoseconds)
        std::int64_t total_serialization_time(
            std::string const& action, bool reset);

        // total data managed (bytes)
        std::int64_t total_bytes(std::string const& action, bool reset);

    private:
        using counter_data_map = std::unordered_map<std::string,
            parcelset::gatherer_nolock, hpx::util::jenkins_hash>;

        mutex_type mtx_;
        counter_data_map data_;
    };
}    // namespace hpx::parcelset::detail

#endif
