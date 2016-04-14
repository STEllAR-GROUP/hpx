//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/include/components.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/util/high_resolution_clock.hpp>

#include <mutex>

#include "sine.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::component<
    ::performance_counters::sine::server::sine_counter
> sine_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY_DYNAMIC(
    sine_counter_type, sine_counter, "base_performance_counter");

///////////////////////////////////////////////////////////////////////////////
namespace performance_counters { namespace sine { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    sine_counter::sine_counter(hpx::performance_counters::counter_info const& info)
      : hpx::performance_counters::base_performance_counter<sine_counter>(info),
        current_value_(0),
        timer_(boost::bind(&sine_counter::evaluate, this),
            1000000, "sine example performance counter")
    {
    }

    bool sine_counter::start()
    {
        return timer_.start();
    }

    bool sine_counter::stop()
    {
        return timer_.stop();
    }

    hpx::performance_counters::counter_value
        sine_counter::get_counter_value(bool reset)
    {
        boost::int64_t const scaling = 100000;

        hpx::performance_counters::counter_value value;

        // gather the current value
        {
            std::lock_guard<mutex_type> mtx(mtx_);
            value.value_ = boost::int64_t(current_value_ * scaling);
            if (reset)
                current_value_ = 0;
            value.time_ = evaluated_at_;
        }

        value.scaling_ = scaling;
        value.scale_inverse_ = true;
        value.status_ = hpx::performance_counters::status_new_data;
        value.count_ = ++invocation_count_;

        return value;
    }

    void sine_counter::finalize()
    {
        timer_.stop();
        hpx::performance_counters::base_performance_counter<sine_counter>::finalize();
    }

    bool sine_counter::evaluate()
    {
        std::lock_guard<mutex_type> mtx(mtx_);
        evaluated_at_ = static_cast<boost::int64_t>(hpx::get_system_uptime());
        current_value_ = std::sin(evaluated_at_ / 1e10);
        return true;
    }
}}}

