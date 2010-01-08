//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/server/raw_counter.hpp>

#include <boost/chrono/chrono.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::performance_counters::server::raw_counter
> raw_counter_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY(raw_counter_type, raw_counter);

HPX_DEFINE_GET_COMPONENT_TYPE(hpx::performance_counters::server::raw_counter);

namespace hpx { namespace actions
{
    template hpx::performance_counters::counter_info const& 
    continuation::trigger(hpx::performance_counters::counter_info const& arg0);

    template hpx::performance_counters::counter_value const& 
    continuation::trigger(hpx::performance_counters::counter_value const& arg0);
}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    raw_counter::raw_counter(counter_info const& info, 
            boost::function<boost::int64_t()> f)
      : info_(info), f_(f)
    {
    }

    void raw_counter::get_counter_info(counter_info& info)
    {
        info = info_;
    }

    void raw_counter::get_counter_value(counter_value& value)
    {
        value.value_ = f_();                // gather the current value
        value.status_ = status_valid_data;
        value.time_ = boost::chrono::high_resolution_clock::now().
            time_since_epoch().count();
    }

}}}

