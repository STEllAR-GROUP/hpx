//  Copyright (c) 2007-2009 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PERFORMANCE_COUNTERS_MAR_04_2009_0931AM)
#define HPX_PERFORMANCE_COUNTERS_MAR_04_2009_0931AM

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>

#include <hpx/runtime/naming/name.hpp>
#include <hpx/performance_counters/counters.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx 
{
    ///////////////////////////////////////////////////////////////////////////
    HPX_API_EXPORT performance_counters::counter_status add_counter_type(
        performance_counters::counter_info const& info, 
        error_code& ec = throws);

    HPX_API_EXPORT performance_counters::counter_status remove_counter_type(
        performance_counters::counter_info const& info, 
        error_code& ec = throws);

    HPX_API_EXPORT performance_counters::counter_status add_counter(
        performance_counters::counter_info const& info, 
        boost::int64_t* countervalue, naming::id_type& id, 
        error_code& ec = throws);

    HPX_API_EXPORT performance_counters::counter_status add_counter(
        performance_counters::counter_info const& info, 
        boost::function<boost::int64_t()> f, naming::id_type& id, 
        error_code& ec = throws);

    HPX_API_EXPORT performance_counters::counter_status remove_counter(
        performance_counters::counter_info const& info, 
        naming::id_type const& id, error_code& ec = throws);

///////////////////////////////////////////////////////////////////////////////
}

#endif

