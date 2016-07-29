//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/server/raw_values_counter.hpp>
#include <hpx/util/function.hpp>

#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::component<
    hpx::performance_counters::server::raw_values_counter
> raw_values_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    raw_values_counter_type, raw_values_counter, "base_performance_counter",
    hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(
    hpx::performance_counters::server::raw_values_counter)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    raw_values_counter::raw_values_counter(counter_info const& info,
            hpx::util::function_nonser<std::vector<boost::int64_t>(bool)> f)
      : base_type_holder(info), f_(std::move(f)), reset_(false)
    {
        if (info.type_ != counter_histogram) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "raw_values_counter::raw_values_counter",
                "unexpected counter type specified for raw_values_counter");
        }
    }

    hpx::performance_counters::counter_values_array
        raw_values_counter::get_counter_values_array(bool reset)
    {
        hpx::performance_counters::counter_values_array values;
        values.values_ = f_(reset);               // gather the current value
        reset_ = false;
        values.scaling_ = 1;
        values.scale_inverse_ = false;
        values.status_ = status_new_data;
        values.time_ = static_cast<boost::int64_t>(hpx::get_system_uptime());
        values.count_ = ++invocation_count_;
        return values;
    }

    void raw_values_counter::reset_counter_value()
    {
        f_(true);
    }
}}}

