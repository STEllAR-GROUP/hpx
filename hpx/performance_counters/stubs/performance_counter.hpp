//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_STUBS_COUNTER_MAR_03_2009_0745M)
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

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        static lcos::future<counter_info> get_info_async(
            naming::id_type const& targetid)
        {
            return get_info(launch::async, targetid);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        static lcos::future<counter_value> get_value_async(
            naming::id_type const& targetid, bool reset = false)
        {
            return get_value(launch::async, targetid, reset);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        static lcos::future<counter_values_array> get_values_array_async(
            naming::id_type const& targetid, bool reset = false)
        {
            return get_values_array(launch::async, targetid, reset);
        }
#endif

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<bool> start(launch::async_policy,
            naming::id_type const& targetid);
        static lcos::future<bool> stop(launch::async_policy,
            naming::id_type const& targetid);
        static lcos::future<void> reset(launch::async_policy,
            naming::id_type const& targetid);

        static bool start(launch::sync_policy, naming::id_type const& targetid,
            error_code& ec = throws);
        static bool stop(launch::sync_policy, naming::id_type const& targetid,
            error_code& ec = throws);
        static void reset(launch::sync_policy, naming::id_type const& targetid,
            error_code& ec = throws);

#if defined(HPX_HAVE_ASYNC_FUNCTION_COMPATIBILITY)
        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        static bool start(naming::id_type const& targetid,
            error_code& ec = throws)
        {
            return start(launch::sync, targetid, ec);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        static bool stop(naming::id_type const& targetid,
            error_code& ec = throws)
        {
            return stop(launch::sync, targetid, ec);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        static void reset(naming::id_type const& targetid,
            error_code& ec = throws)
        {
            reset(launch::sync, targetid, ec);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        static lcos::future<bool> start_async(naming::id_type const& targetid)
        {
            return start(launch::async, targetid);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        static lcos::future<bool> stop_async(naming::id_type const& targetid)
        {
            return stop(launch::async, targetid);
        }

        HPX_DEPRECATED(HPX_DEPRECATED_MSG)
        static lcos::future<void> reset_async(naming::id_type const& targetid)
        {
            return reset(launch::async, targetid);
        }
#endif

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
