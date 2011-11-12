
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/exception.hpp>
#include <hpx/util/spinlock.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/threads/thread.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/traits/get_remote_result.hpp>


#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>

#include <examples/bright_future/dataflow/server/dataflow.hpp>

HPX_REGISTER_COMPONENT_MODULE();

typedef hpx::components::managed_component<hpx::lcos::server::dataflow> dataflow_type;

HPX_REGISTER_MINIMAL_COMPONENT_FACTORY_EX(
    dataflow_type,
    bright_future_dataflow, true);

HPX_DEFINE_GET_COMPONENT_TYPE(dataflow_type::wrapped_type);

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server { namespace detail
{
    // the counter data instance
    dataflow_counter_data dataflow_counter_data_;

    boost::int64_t get_constructed_count()
    {
        return dataflow_counter_data_.constructed_;
    }

    boost::int64_t get_initialized_count()
    {
        return dataflow_counter_data_.initialized_;
    }

    boost::int64_t get_fired_count()
    {
        return dataflow_counter_data_.fired_;
    }

    // call this to install all counters for full_empty entries
    void install_counters()
    {
        performance_counters::raw_counter_type_data const counter_types[] =
        {
            { "/dataflow/constructed", performance_counters::counter_raw,
              "returns the number of constructed dataflow objects",
              HPX_PERFORMANCE_COUNTER_V1 },
            { "/dataflow/initialized", performance_counters::counter_raw,
              "returns the number of initialized dataflow objects",
              HPX_PERFORMANCE_COUNTER_V1 },
            { "/dataflow/fired", performance_counters::counter_raw,
              "returns the number of fired dataflow objects",
              HPX_PERFORMANCE_COUNTER_V1 }
        };

        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));

        boost::uint32_t const prefix = applier::get_applier().get_prefix_id();
        boost::format full_empty_counter("/dataflow(locality#%d/total)/%s");

        performance_counters::raw_counter_data const counters[] =
        {
            { boost::str(full_empty_counter % prefix % "constructed"),
              get_constructed_count },
            { boost::str(full_empty_counter % prefix % "initialized"),
              get_initialized_count },
            { boost::str(full_empty_counter % prefix % "fired"),
              get_fired_count },
        };

        performance_counters::install_counters(
            counters, sizeof(counters)/sizeof(counters[0]));
    }
}}}}

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a px-thread during
// runtime startup. We use this function to register our performance counter
// type and performance counter instances.
//
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_SHUTDOWN_MODULE(::hpx::lcos::server::detail::install_counters, 0);

