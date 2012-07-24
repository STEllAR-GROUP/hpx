//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/util/high_resolution_clock.hpp>

#include "sine.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    ::performance_counters::sine::server::sine_counter
> sine_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    sine_counter_type, sine_counter, "base_performance_counter");

///////////////////////////////////////////////////////////////////////////////
namespace performance_counters { namespace sine { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    sine_counter::sine_counter(hpx::performance_counters::counter_info const& info)
      : base_type_holder(info), current_value_(0),
        timer_(boost::bind(&sine_counter::evaluate, this), 1000000,
            "sine example performance counter"),
        started_at_(hpx::util::high_resolution_clock::now())
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

    void sine_counter::get_counter_value(
        hpx::performance_counters::counter_value& value)
    {
        boost::int64_t const scaling = 100000;

        // gather the current value
        {
            mutex_type::scoped_lock mtx(mtx_);
            value.value_ = boost::int64_t(current_value_ * scaling);
            value.time_ = evaluated_at_;
        }

        value.scaling_ = scaling;
        value.scale_inverse_ = true;
        value.status_ = hpx::performance_counters::status_new_data;
    }

    void sine_counter::finalize()
    {
        timer_.stop();
        base_type_holder::finalize();
        base_type::finalize();
    }

    bool sine_counter::evaluate()
    {
        mutex_type::scoped_lock mtx(mtx_);
        evaluated_at_ = hpx::util::high_resolution_clock::now();
        current_value_ = std::sin((evaluated_at_ - started_at_) / 1e10);
        return true;
    }
}}}

