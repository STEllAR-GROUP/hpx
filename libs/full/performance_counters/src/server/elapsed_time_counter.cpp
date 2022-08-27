//  Copyright (c) 2007-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/async_distributed/continuation.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/server/elapsed_time_counter.hpp>
#include <hpx/runtime_components/derived_component_factory.hpp>
#include <hpx/runtime_local/runtime_local_fwd.hpp>
#include <hpx/timing/high_resolution_clock.hpp>

#include <boost/version.hpp>

#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::component<
    hpx::performance_counters::server::elapsed_time_counter>
    elapsed_time_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY(elapsed_time_counter_type,
    elapsed_time_counter, "base_performance_counter",
    hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(
    hpx::performance_counters::server::elapsed_time_counter)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server {

    elapsed_time_counter::elapsed_time_counter() = default;

    elapsed_time_counter::elapsed_time_counter(counter_info const& info)
      : base_type_holder(info)
    {
        if (info.type_ != counter_elapsed_time)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "elapsed_time_counter::elapsed_time_counter",
                "unexpected counter type specified for elapsed_time_counter");
        }
    }

    hpx::performance_counters::counter_value
    elapsed_time_counter::get_counter_value(bool reset)
    {
        if (reset)
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "elapsed_time_counter::get_counter_value",
                "counter /runtime/uptime does no support reset");
        }

        // gather the current value
        std::int64_t now = static_cast<std::int64_t>(hpx::get_system_uptime());
        hpx::performance_counters::counter_value value;
        value.value_ = now;
        value.scaling_ = 1000000000LL;    // coefficient to get seconds
        value.scale_inverse_ = true;
        value.status_ = status_new_data;
        value.time_ = now;
        value.count_ = ++invocation_count_;
        return value;
    }

    void elapsed_time_counter::reset_counter_value()
    {
        HPX_THROW_EXCEPTION(bad_parameter,
            "elapsed_time_counter::reset_counter_value",
            "counter /runtime/uptime does no support reset");
    }

    bool elapsed_time_counter::start()
    {
        return false;
    }
    bool elapsed_time_counter::stop()
    {
        return false;
    }

    void elapsed_time_counter::finalize()
    {
        base_performance_counter::finalize();
        base_type::finalize();
    }

    naming::address elapsed_time_counter::get_current_address() const
    {
        return naming::address(
            naming::get_gid_from_locality_id(agas::get_locality_id()),
            components::get_component_type<elapsed_time_counter>(),
            const_cast<elapsed_time_counter*>(this));
    }
}}}    // namespace hpx::performance_counters::server

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace detail {
    /// Creation function for uptime counters.
    naming::gid_type uptime_counter_creator(
        counter_info const& info, error_code& ec)
    {
        switch (info.type_)
        {
        case counter_elapsed_time:
        {
            // verify the validity of the counter instance name
            counter_path_elements paths;
            get_counter_path_elements(info.fullname_, paths, ec);
            if (ec)
                return naming::invalid_gid;

            // allowed counter names: /runtime(locality#%d/*)/uptime
            if (paths.parentinstance_is_basename_)
            {
                HPX_THROWS_IF(ec, bad_parameter, "uptime_counter_creator",
                    "invalid counter instance parent name: {}",
                    paths.parentinstancename_);
                return naming::invalid_gid;
            }

            // create the counter
            return create_counter(info, ec);
        }

        default:
            HPX_THROWS_IF(ec, bad_parameter, "uptime_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }
}}}    // namespace hpx::performance_counters::detail
