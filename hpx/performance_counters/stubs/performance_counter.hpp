//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_STUBS_COUNTER_MAR_03_2009_0745M)
#define HPX_PERFORMANCE_COUNTERS_STUBS_COUNTER_MAR_03_2009_0745M

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/stubs/stub_base.hpp>
#include <hpx/lcos/eager_future.hpp>
#include <hpx/performance_counters/server/base_performance_counter.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace stubs
{
    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT performance_counter 
      : components::stubs::stub_base<server::base_performance_counter>
    {
        static lcos::promise<counter_info> get_info_async(
            naming::gid_type const& targetgid);
        static lcos::promise<counter_value> get_value_async(
            naming::gid_type const& targetgid);

        static counter_info get_info(naming::gid_type const& targetgid,
            error_code& ec = throws);
        static counter_value get_value(naming::gid_type const& targetgid,
            error_code& ec = throws);

        static counter_info get_info(naming::id_type const& targetgid,
            error_code& ec = throws);
        static counter_value get_value(naming::id_type const& targetgid,
            error_code& ec = throws);

        template <typename T>
        static T 
        get_typed_value(naming::gid_type const& targetgid, error_code& ec = throws)
        {
            counter_value value = get_value(targetgid);
            return value.get_value<T>(ec);
        }
    };
}}}

#endif
