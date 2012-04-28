//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/high_resolution_clock.hpp>
#include <hpx/performance_counters/server/raw_counter.hpp>

#include <boost/version.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::performance_counters::server::raw_counter
> raw_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY_EX(
    raw_counter_type, raw_counter, "base_performance_counter",
    hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(
    hpx::performance_counters::server::raw_counter)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    raw_counter::raw_counter(counter_info const& info,
            HPX_STD_FUNCTION<boost::int64_t()> f)
      : base_type_holder(info), f_(f)
    {
        if (info.type_ != counter_raw) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "raw_counter::raw_counter",
                "unexpected counter type specified for raw_counter");
        }
    }

    void raw_counter::get_counter_value(counter_value& value)
    {
        value.value_ = f_();                // gather the current value
        value.scaling_ = 1;
        value.scale_inverse_ = false;
        value.status_ = status_new_data;
        value.time_ = high_resolution_clock::now();
    }
}}}

