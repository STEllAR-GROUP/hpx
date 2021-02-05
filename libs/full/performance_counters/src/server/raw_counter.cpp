//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions/continuation.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/server/raw_counter.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/timing/high_resolution_clock.hpp>

#include <cstdint>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::component<
    hpx::performance_counters::server::raw_counter>
    raw_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(raw_counter_type, raw_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(hpx::performance_counters::server::raw_counter)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server {

    raw_counter::raw_counter(counter_info const& info,
        hpx::util::function_nonser<std::int64_t(bool)> f)
      : base_type_holder(info)
      , f_(std::move(f))
      , reset_(false)
    {
        if (info.type_ != counter_raw && info.type_ != counter_elapsed_time &&
            info.type_ != counter_aggregating &&
            info.type_ != counter_monotonically_increasing &&
            info.type_ != counter_average_count &&
            info.type_ != counter_average_timer)
        {
            HPX_THROW_EXCEPTION(bad_parameter, "raw_counter::raw_counter",
                "unexpected counter type specified for raw_counter");
        }
    }

    hpx::performance_counters::counter_value raw_counter::get_counter_value(
        bool reset)
    {
        hpx::performance_counters::counter_value value;
        value.value_ = f_(reset);    // gather the current value
        reset_ = false;
        value.scaling_ = 1;
        value.scale_inverse_ = false;
        value.status_ = status_new_data;
        value.time_ = static_cast<std::int64_t>(hpx::get_system_uptime());
        value.count_ = ++invocation_count_;
        return value;
    }

    void raw_counter::reset_counter_value()
    {
        f_(true);
    }
}}}    // namespace hpx::performance_counters::server
