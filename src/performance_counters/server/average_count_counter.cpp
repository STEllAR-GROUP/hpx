//  Copyright (c) 2007-2011 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/stubs/performance_counter.hpp>
#include <hpx/performance_counters/server/average_count_counter.hpp>

#include <boost/version.hpp>
#include <boost/chrono/chrono.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::performance_counters::server::average_count_counter
> average_count_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY_EX(
    average_count_counter_type, average_count_counter,
    "base_performance_counter", true);
HPX_DEFINE_GET_COMPONENT_TYPE(
    hpx::performance_counters::server::average_count_counter);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    average_count_counter::average_count_counter(counter_info const& info,
            std::string const& base_counter_name,
            std::size_t base_time_interval)
      : base_type_holder(info),
        timer_(boost::bind(&average_count_counter::evaluate, this),
            1000 * base_time_interval, info.fullname_, true),
        base_counter_name_(ensure_counter_prefix(base_counter_name))
    {
        if (base_time_interval == 0) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "average_count_counter::average_count_counter",
                "base interval is specified to be zero");
        }

        if (info.type_ != counter_average_count) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "average_count_counter::average_count_counter",
                "unexpected counter type specified for elapsed_time_counter");
        }

        // acquire the current value of the base counter
        counter_value base_value;
        evaluate_base_counter(base_value);

        {
            mutex_type::scoped_lock l(mtx_);
            value_(base_value.value_);
            prev_value_ = base_value;
        }

        timer_.start();       // start interval timer
    }

    void average_count_counter::get_counter_value(counter_value& value)
    {
        mutex_type::scoped_lock l(mtx_);

        value = prev_value_;                // return value
        value.value_ = boost::accumulators::mean(value_);
        value.status_ = status_new_data;
        value.time_ = boost::chrono::high_resolution_clock::now().
            time_since_epoch().count();

        value_ = mean_accumulator_type();   // reset accumulator
        value_(prev_value_.value_);         // start off with last base value
    }

    void average_count_counter::evaluate()
    {
        // gather current base value
        counter_value base_value;
        evaluate_base_counter(base_value);

        // simply average the measured base counter values since it got queried
        // for the last time
        counter_value value;
        if (base_value.scaling_ != prev_value_.scaling_ ||
            base_value.scale_inverse_ != prev_value_.scale_inverse_)
        {
            // not supported right now
            HPX_THROW_EXCEPTION(not_implemented,
                "average_count_counter::get_counter_value",
                "base counter should keep scaling constant over time");
        }
        else {
            mutex_type::scoped_lock l(mtx_);
            value_(base_value.value_);          // accumulate new value
        }
    }

    void average_count_counter::evaluate_base_counter(counter_value& value)
    {
        {
            // lock here to avoid checking out multiple reference counted GIDs
            // from AGAS
            mutex_type::scoped_lock l(mtx_);

            if (!base_counter_id_) {
                // retrieve the base counter gid, if it's not know yet
                error_code ec;
                agas::resolve_name(base_counter_name_, base_counter_id_, ec);
                if (HPX_UNLIKELY(ec || !base_counter_id_))
                {
                    HPX_THROW_EXCEPTION(bad_parameter,
                        "average_count_counter::evaluate_base_counter",
                        boost::str(
                            boost::format("unknown performance counter: '%s'") %
                            base_counter_name_))
                }
            }
        }

        // query the actual value
        value = stubs::performance_counter::get_value(base_counter_id_);
    }
}}}

