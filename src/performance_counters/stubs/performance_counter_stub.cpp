//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/apply.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/runtime/actions/continuation.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace stubs
{
    lcos::future<counter_info> performance_counter::get_info(
        launch::async_policy, naming::id_type const& targetid)
    {
        typedef server::base_performance_counter::
            get_counter_info_action action_type;
        return hpx::async<action_type>(targetid);
    }

    lcos::future<counter_value> performance_counter::get_value(
        launch::async_policy, naming::id_type const& targetid, bool reset)
    {
        typedef server::base_performance_counter::
            get_counter_value_action action_type;
        return hpx::async<action_type>(targetid, reset);
    }

    lcos::future<counter_values_array>
    performance_counter::get_values_array(launch::async_policy,
        naming::id_type const& targetid, bool reset)
    {
        typedef server::base_performance_counter::
            get_counter_values_array_action action_type;
        return hpx::async<action_type>(targetid, reset);
    }

    counter_info performance_counter::get_info(launch::sync_policy,
        naming::id_type const& targetid, error_code& ec)
    {
        return get_info(launch::async, targetid).get(ec);
    }

    counter_value performance_counter::get_value(launch::sync_policy,
        naming::id_type const& targetid, bool reset, error_code& ec)
    {
        return get_value(launch::async, targetid, reset).get(ec);
    }

    counter_values_array
    performance_counter::get_values_array(launch::sync_policy,
        naming::id_type const& targetid, bool reset, error_code& ec)
    {
        return get_values_array(launch::async, targetid, reset).get(ec);
    }

    lcos::future<bool> performance_counter::start(launch::async_policy,
        naming::id_type const& targetid)
    {
        typedef server::base_performance_counter::start_action action_type;
        return hpx::async<action_type>(targetid);
    }

    lcos::future<bool> performance_counter::stop(launch::async_policy,
        naming::id_type const& targetid)
    {
        typedef server::base_performance_counter::stop_action action_type;
        return hpx::async<action_type>(targetid);
    }

    lcos::future<void> performance_counter::reset(launch::async_policy,
        naming::id_type const& targetid)
    {
        typedef server::base_performance_counter::reset_counter_value_action
            action_type;
        return hpx::async<action_type>(targetid);
    }

    bool performance_counter::start(launch::sync_policy,
        naming::id_type const& targetid, error_code& ec)
    {
        return start(launch::async, targetid).get(ec);
    }

    bool performance_counter::stop(launch::sync_policy,
        naming::id_type const& targetid, error_code& ec)
    {
        return stop(launch::async, targetid).get(ec);
    }

    void performance_counter::reset(launch::sync_policy,
        naming::id_type const& targetid, error_code& ec)
    {
        reset(launch::async, targetid).get(ec);
    }
}}}
