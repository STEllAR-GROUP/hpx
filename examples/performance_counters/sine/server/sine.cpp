//  Copyright (c) 2007-2011 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation_impl.hpp>
#include <hpx/performance_counters/counters.hpp>

#include <boost/version.hpp>
#include <boost/chrono/chrono.hpp>

#include "sine.hpp"

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    ::performance_counters::sine::server::sine_counter
> sine_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(
    sine_counter_type, sine_counter, "base_performance_counter");
HPX_DEFINE_GET_COMPONENT_TYPE(::performance_counters::sine::server::sine_counter);

///////////////////////////////////////////////////////////////////////////////
namespace performance_counters { namespace sine { namespace server
{
    ///////////////////////////////////////////////////////////////////////////
    sine_counter::sine_counter(hpx::performance_counters::counter_info const& info)
      : base_type_holder(info), current_value_(0),
        timer_(boost::bind(&sine_counter::evaluate, this), 1000000, 
            "sine example performance counter"), 
        started_at_(boost::chrono::high_resolution_clock::now())
    {
        timer_.start();
    }

    void sine_counter::get_counter_value(
        hpx::performance_counters::counter_value& value)
    {
        boost::int64_t const scaling = 100000;

        {
            mutex_type::scoped_lock mtx(mtx_);
            value.value_ = current_value_ * scaling;    // gather the current value
        }

        value.scaling_ = scaling;
        value.scale_inverse_ = true;
        value.status_ = hpx::performance_counters::status_valid_data;

        using namespace boost::chrono;
        value.time_ = high_resolution_clock::now().time_since_epoch().count();
    }

    void sine_counter::finalize() 
    {
        timer_.stop();
        base_type_holder::finalize();
        base_type::finalize();
    }

    void sine_counter::evaluate()
    {
        using namespace boost::chrono;

        mutex_type::scoped_lock mtx(mtx_);
        duration<double> up_time = high_resolution_clock::now() - started_at_;
        current_value_ = std::sin(up_time.count() / 10.);
    }
}}}

