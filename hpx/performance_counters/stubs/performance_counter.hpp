//  Copyright (c) 2007-2010 Hartmut Kaiser
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
    struct performance_counter 
      : components::stubs::stub_base<server::base_performance_counter>
    {
        static lcos::future_value<counter_info> get_info_async(
            naming::id_type const& targetgid);
        static lcos::future_value<counter_value> get_value_async(
            naming::id_type const& targetgid);

        static counter_info get_info(naming::id_type const& targetgid);
        static counter_value get_value(naming::id_type const& targetgid);
    };

}}}

#endif
