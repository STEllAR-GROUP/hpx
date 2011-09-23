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
    
    sine_counter::sine_counter(
            hpx::performance_counters::counter_info const& info)
      : base_type_holder(info), current_value_(0),
        id_(0), started_at_(boost::chrono::high_resolution_clock::now())
    {
        evaluate(hpx::threads::wait_signaled);
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
        mutex_type::scoped_lock mtx(mtx_);
        if (id_) {
            using namespace hpx::threads;

            hpx::error_code ec;       // avoid throwing on error
            hpx::threads::set_thread_state(id_, pending, wait_abort, 
                thread_priority_critical, ec);
        }
        base_type_holder::finalize();
        base_type::finalize();
    }

    hpx::threads::thread_state_enum 
    sine_counter::evaluate(hpx::threads::thread_state_ex_enum statex)
    {
        using namespace hpx::threads;
        using namespace boost::chrono;

        if (statex == wait_abort)
            return terminated;        // object has been finalized, exit

        mutex_type::scoped_lock mtx(mtx_);
        id_ = 0;

        duration<double> up_time = high_resolution_clock::now() - started_at_;
        current_value_ = std::sin(up_time.count() / 10.);

        schedule_thread(1);   // wait one second and repeat

        return terminated;    // do not re-schedule this thread
    }

    // schedule a high priority task after a given time interval
    void sine_counter::schedule_thread(std::size_t secs)
    {
        using namespace hpx::threads;

        // create a new suspended thread
        std::string description("sine example performance counter");
        id_ = hpx::applier::register_thread_plain(
            boost::bind(&sine_counter::evaluate, this, _1), description.c_str(), 
            suspended);

        // schedule this thread to be run after the given amount of seconds
        set_thread_state(id_, boost::posix_time::seconds(secs), 
            pending, wait_signaled, thread_priority_critical);
    }
}}}

