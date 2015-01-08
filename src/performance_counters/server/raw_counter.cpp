//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/server/raw_counter.hpp>

#include <boost/version.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::performance_counters::server::raw_counter
> raw_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    raw_counter_type, raw_counter, "base_performance_counter",
    hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(
    hpx::performance_counters::server::raw_counter)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    raw_counter::raw_counter(counter_info const& info,
            util::function_nonser<boost::int64_t(bool)> f)
      : base_type_holder(info), f_(std::move(f))
    {
        if (info.type_ != counter_raw) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "raw_counter::raw_counter",
                "unexpected counter type specified for raw_counter");
        }
    }

    hpx::performance_counters::counter_value
        raw_counter::get_counter_value(bool reset)
    {
        hpx::performance_counters::counter_value value;
        value.value_ = f_(reset);               // gather the current value
        reset_ = false;
        value.scaling_ = 1;
        value.scale_inverse_ = false;
        value.status_ = status_new_data;
        value.time_ = static_cast<boost::int64_t>(hpx::get_system_uptime());
        value.count_ = ++invocation_count_;
        return value;
    }

    void raw_counter::reset_counter_value()
    {
        f_(true);
    }
}}}

