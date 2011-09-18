//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/performance_counters.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    performance_counters::counter_status add_counter_type(
        performance_counters::counter_info const& info, error_code& ec)
    {
        return get_runtime().get_counter_registry().add_counter_type(info, ec);
    }

    performance_counters::counter_status remove_counter_type(
        performance_counters::counter_info const& info, error_code& ec)
    {
        // the runtime might not be available any more
        runtime* rt = get_runtime_ptr();
        return rt ? rt->get_counter_registry().remove_counter_type(info, ec) : 
            performance_counters::status_generic_error;
    }

    performance_counters::counter_status add_counter(
        performance_counters::counter_info const& info, 
        boost::int64_t* countervalue, naming::id_type& id, error_code& ec)
    {
        return get_runtime().get_counter_registry().add_counter(
            info, countervalue, id, ec);
    }

    performance_counters::counter_status add_counter(
        performance_counters::counter_info const& info, 
        boost::function<boost::int64_t()> f, naming::id_type& id, 
        error_code& ec)
    {
        return get_runtime().get_counter_registry().add_counter(
            info, f, id, ec);
    }

    performance_counters::counter_status add_counter(
        performance_counters::counter_info const& info, 
        naming::id_type& id, error_code& ec)
    {
        return get_runtime().get_counter_registry().add_counter(info, id, ec);
    }

    performance_counters::counter_status remove_counter(
        performance_counters::counter_info const& info, 
        naming::id_type const& id, error_code& ec)
    {
        return get_runtime().get_counter_registry().remove_counter(
            info, id, ec);
    }
}}

