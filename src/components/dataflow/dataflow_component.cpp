//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/exception.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/util/unlock_lock.hpp>
#include <hpx/util/stringstream.hpp>
#include <hpx/runtime/threads/thread_data.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/runtime/components/server/managed_component_base.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/include/performance_counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/lcos/base_lco.hpp>
#include <hpx/components/dataflow/server/dataflow.hpp>
#include <hpx/components/dataflow/server/dataflow_trigger.hpp>
#include <hpx/components/dataflow/dataflow_base.hpp>

#include <hpx/util/portable_binary_iarchive.hpp>
#include <hpx/util/portable_binary_oarchive.hpp>

#include <boost/serialization/version.hpp>

HPX_REGISTER_COMPONENT_MODULE()

typedef hpx::components::managed_component<hpx::lcos::server::dataflow> dataflow_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(dataflow_type, dataflow,
    "hpx::lcos::server::dataflow")

typedef hpx::components::managed_component<hpx::lcos::server::dataflow_trigger>
    dataflow_trigger_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(dataflow_trigger_type, dataflow_trigger,
    "hpx::lcos::server::dataflow")

// HPX_REGISTER_ACTION(
//     hpx::lcos::server::dataflow_trigger::init_action
//   , dataflow_trigger_type_init_action
// )

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace lcos { namespace server { namespace detail
{
    // the counter data instance
    dataflow_counter_data dataflow_counter_data_;

    boost::int64_t get_constructed_count()
    {
        lcos::local::spinlock::scoped_lock l(dataflow_counter_data_.mtx_);
        return dataflow_counter_data_.constructed_;
    }

    boost::int64_t get_initialized_count()
    {
        lcos::local::spinlock::scoped_lock l(dataflow_counter_data_.mtx_);
        return dataflow_counter_data_.initialized_;
    }

    boost::int64_t get_fired_count()
    {
        lcos::local::spinlock::scoped_lock l(dataflow_counter_data_.mtx_);
        return dataflow_counter_data_.fired_;
    }

    boost::int64_t get_destructed_count()
    {
        lcos::local::spinlock::scoped_lock l(dataflow_counter_data_.mtx_);
        return dataflow_counter_data_.destructed_;
    }

    void update_constructed_count()
    {
        lcos::local::spinlock::scoped_lock l(dataflow_counter_data_.mtx_);
        ++dataflow_counter_data_.constructed_;
    }

    void update_initialized_count()
    {
        lcos::local::spinlock::scoped_lock l(dataflow_counter_data_.mtx_);
        ++dataflow_counter_data_.initialized_;
    }

    void update_fired_count()
    {
        lcos::local::spinlock::scoped_lock l(dataflow_counter_data_.mtx_);
        ++dataflow_counter_data_.fired_;
    }
    
    void update_destructed_count()
    {
        lcos::local::spinlock::scoped_lock l(dataflow_counter_data_.mtx_);
        ++dataflow_counter_data_.destructed_;
    }

    // call this to register all counter types for dataflow entries
    void register_counter_types()
    {
        performance_counters::generic_counter_type_data const counter_types[] =
        {
            { "/dataflow/count/constructed", performance_counters::counter_raw,
              "returns the number of constructed dataflow objects",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, get_constructed_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/dataflow/count/initialized", performance_counters::counter_raw,
              "returns the number of initialized dataflow objects",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, get_initialized_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/dataflow/count/fired", performance_counters::counter_raw,
              "returns the number of fired dataflow objects",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, get_fired_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            },
            { "/dataflow/count/destructed", performance_counters::counter_raw,
              "returns the number of destructed dataflow objects",
              HPX_PERFORMANCE_COUNTER_V1,
              boost::bind(&performance_counters::locality_raw_counter_creator,
                  _1, get_destructed_count, _2),
              &performance_counters::locality_counter_discoverer,
              ""
            }
        };
        performance_counters::install_counter_types(
            counter_types, sizeof(counter_types)/sizeof(counter_types[0]));
    }
}}}}

namespace dataflow_module
{
    ///////////////////////////////////////////////////////////////////////////
    bool get_startup(HPX_STD_FUNCTION<void()>& startup_func, bool& pre_startup)
    {
        // return our startup-function if performance counters are required
        startup_func = hpx::lcos::server::detail::register_counter_types;
        pre_startup = true;  // run 'register_counter_types' as pre-startup function
        return true;
    }
}

///////////////////////////////////////////////////////////////////////////////
// Register a startup function which will be called as a HPX-thread during
// runtime startup. We use this function to register our performance counter
// type and performance counter instances.
//
// Note that this macro can be used not more than once in one module.
HPX_REGISTER_STARTUP_MODULE(::dataflow_module::get_startup);
