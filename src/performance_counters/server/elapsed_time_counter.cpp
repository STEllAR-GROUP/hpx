//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/server/elapsed_time_counter.hpp>

#include <boost/version.hpp>
#include <boost/chrono/chrono.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::performance_counters::server::elapsed_time_counter
> elapsed_time_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY_EX(
    elapsed_time_counter_type, elapsed_time_counter, 
    "base_performance_counter", true);
HPX_DEFINE_GET_COMPONENT_TYPE(
    hpx::performance_counters::server::elapsed_time_counter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    elapsed_time_counter::elapsed_time_counter(counter_info const& info)
      : base_type_holder(info)
    {
        if (info.type_ != counter_elapsed_time) {
            HPX_THROW_EXCEPTION(bad_parameter, 
                "elapsed_time_counter::elapsed_time_counter",
                "unexpected counter type specified for elapsed_time_counter");
        }
    }

    void elapsed_time_counter::get_counter_value(counter_value& value)
    {
        value.value_ = timer_.elapsed() * 10e8;     // gather the current value
        value.scaling_ = 10e8;
        value.scale_inverse_ = true;
        value.status_ = status_valid_data;
        value.time_ = boost::chrono::high_resolution_clock::now().
            time_since_epoch().count();
    }
}}}

