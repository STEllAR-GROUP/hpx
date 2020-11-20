//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_NETWORKING) && defined(HPX_HAVE_PARCEL_COALESCING)

#include <hpx/functional/function.hpp>
#include <hpx/hashing/jenkins_hash.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/performance_counters/counters_fwd.hpp>
#include <hpx/type_support/static.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace plugins { namespace parcel
{
    ///////////////////////////////////////////////////////////////////////////
    class coalescing_counter_registry
    {
        typedef hpx::lcos::local::spinlock mutex_type;

    public:
        HPX_NON_COPYABLE(coalescing_counter_registry);

    public:
        coalescing_counter_registry() {}

        typedef util::function_nonser<std::int64_t(bool)>
            get_counter_type;
        typedef util::function_nonser<std::vector<std::int64_t>(bool)>
            get_counter_values_type;
        typedef util::function_nonser<
                void(std::int64_t, std::int64_t, std::int64_t,
                    get_counter_values_type&)
            > get_counter_values_creator_type;

        struct counter_functions
        {
            get_counter_type num_parcels;
            get_counter_type num_messages;
            get_counter_type num_parcels_per_message;
            get_counter_type average_time_between_parcels;
            get_counter_values_creator_type time_between_parcels_histogram_creator;
            std::int64_t min_boundary, max_boundary, num_buckets;
        };

        typedef std::unordered_map<
                std::string, counter_functions, hpx::util::jenkins_hash
            > map_type;

        static coalescing_counter_registry& instance();

        void register_action(std::string const& name);

        void register_action(std::string const& name,
            get_counter_type num_parcels, get_counter_type num_messages,
            get_counter_type time_between_parcels,
            get_counter_type average_time_between_parcels,
            get_counter_values_creator_type time_between_parcels_histogram_creator);

        get_counter_type get_parcels_counter(std::string const& name) const;
        get_counter_type get_messages_counter(std::string const& name) const;
        get_counter_type get_parcels_per_message_counter(
            std::string const& name) const;
        get_counter_type get_average_time_between_parcels_counter(
            std::string const& name) const;
        get_counter_values_type get_time_between_parcels_histogram_counter(
            std::string const& name, std::int64_t min_boundary,
            std::int64_t max_boundary, std::int64_t num_buckets);

        bool counter_discoverer(
            performance_counters::counter_info const& info,
            performance_counters::counter_path_elements& p,
            performance_counters::discover_counter_func const& f,
            performance_counters::discover_counters_mode mode, error_code& ec);

        static std::vector<std::int64_t> empty_histogram(bool)
        {
            std::vector<std::int64_t> result = { 0, 0, 1, 0 };
            return result;
        }

    private:
        struct tag {};

        friend struct hpx::util::static_<
                coalescing_counter_registry, tag
            >;

        mutable mutex_type mtx_;
        map_type map_;
    };
}}}

#endif
