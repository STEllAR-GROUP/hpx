//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_STUBS_COUNTER_MAR_03_2009_0745M)
#define HPX_PERFORMANCE_COUNTERS_STUBS_COUNTER_MAR_03_2009_0745M

#include <hpx/config.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT performance_counter
      : components::stub_base<
            performance_counters::server::base_performance_counter>
    {
        ///////////////////////////////////////////////////////////////////////
        static lcos::future<counter_info> get_info_async(
            naming::id_type const& targetid);
        static lcos::future<counter_value> get_value_async(
            naming::id_type const& targetid, bool reset = false);
        static lcos::future<counter_values_array> get_values_array_async(
            naming::id_type const& targetid, bool reset = false);

        static counter_info get_info(naming::id_type const& targetid,
            error_code& ec = throws);
        static counter_value get_value(naming::id_type const& targetid,
            bool reset = false, error_code& ec = throws);
        static counter_values_array get_values_array(
            naming::id_type const& targetid,
            bool reset = false, error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<bool> start_async(naming::id_type const& targetid);
        static lcos::future<bool> stop_async(naming::id_type const& targetid);
        static lcos::future<void> reset_async(naming::id_type const& targetid);

        static bool start(naming::id_type const& targetid,
            error_code& ec = throws);
        static bool stop(naming::id_type const& targetid,
            error_code& ec = throws);
        static void reset(naming::id_type const& targetid,
            error_code& ec = throws);

        template <typename T>
        static T
        get_typed_value(naming::id_type const& targetid, bool reset = false,
            error_code& ec = throws)
        {
            counter_value value = get_value(targetid, reset);
            return value.get_value<T>(ec);
        }
    };
}}}

#endif
