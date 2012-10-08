//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_STUBS_COUNTER_MAR_03_2009_0745M)
#define HPX_PERFORMANCE_COUNTERS_STUBS_COUNTER_MAR_03_2009_0745M

#include <hpx/hpx_fwd.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT performance_counter
    {
        ///////////////////////////////////////////////////////////////////////
        static lcos::future<counter_info> get_info_async(
            naming::id_type const& targetid);
        static lcos::future<counter_value> get_value_async(
            naming::id_type const& targetid);

        static counter_info get_info(naming::id_type const& targetid,
            error_code& ec = throws);
        static counter_value get_value(naming::id_type const& targetid,
            error_code& ec = throws);

        ///////////////////////////////////////////////////////////////////////
        static lcos::future<bool> start_async(naming::id_type const& targetid);
        static lcos::future<bool> stop_async(naming::id_type const& targetid);

        static bool start(naming::id_type const& targetid,
            error_code& ec = throws);
        static bool stop(naming::id_type const& targetid,
            error_code& ec = throws);

        template <typename T>
        static T
        get_typed_value(naming::id_type const& targetid, error_code& ec = throws)
        {
            counter_value value = get_value(targetid);
            return value.get_value<T>(ec);
        }
    };
}}}

#endif
