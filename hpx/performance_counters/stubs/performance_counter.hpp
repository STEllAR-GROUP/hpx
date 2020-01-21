//  Copyright (c) 2007-2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PERFORMANCE_COUNTERS_STUBS_COUNTER_MAR_03_2009_0745M
#define HPX_PERFORMANCE_COUNTERS_STUBS_COUNTER_MAR_03_2009_0745M

#include <hpx/config.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/runtime/launch_policy.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT performance_counter
      : components::stub_base<
            performance_counters::server::base_performance_counter>
    {
        ///////////////////////////////////////////////////////////////////////
        static counter_info get_info(launch::sync_policy,
            naming::id_type const& targetid, error_code& ec = throws);
        static counter_value get_value(launch::sync_policy,
            naming::id_type const& targetid, bool reset = false,
            error_code& ec = throws);
        static counter_values_array get_values_array(launch::sync_policy,
            naming::id_type const& targetid,
            bool reset = false, error_code& ec = throws);

        static lcos::future<counter_info> get_info(launch::async_policy,
            naming::id_type const& targetid);
        static lcos::future<counter_value> get_value(launch::async_policy,
            naming::id_type const& targetid, bool reset = false);
        static lcos::future<counter_values_array> get_values_array(
            launch::async_policy, naming::id_type const& targetid,
            bool reset = false);

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<bool> start(launch::async_policy,
            naming::id_type const& targetid);
        static lcos::future<bool> stop(launch::async_policy,
            naming::id_type const& targetid);
        static lcos::future<void> reset(launch::async_policy,
            naming::id_type const& targetid);
        static lcos::future<void> reinit(launch::async_policy,
            naming::id_type const& targetid, bool reset);

        static bool start(launch::sync_policy, naming::id_type const& targetid,
            error_code& ec = throws);
        static bool stop(launch::sync_policy, naming::id_type const& targetid,
            error_code& ec = throws);
        static void reset(launch::sync_policy, naming::id_type const& targetid,
            error_code& ec = throws);
        static void reinit(launch::sync_policy, naming::id_type const& targetid,
            bool reset, error_code& ec = throws);

        template <typename T>
        static T
        get_typed_value(naming::id_type const& targetid, bool reset = false,
            error_code& ec = throws)
        {
            counter_value value = get_value(launch::sync, targetid, reset);
            return value.get_value<T>(ec);
        }
    };
}}}

#endif
