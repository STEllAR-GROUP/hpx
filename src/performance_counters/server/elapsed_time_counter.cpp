//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/derived_component_factory.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/high_resolution_clock.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/server/elapsed_time_counter.hpp>

#include <boost/version.hpp>

///////////////////////////////////////////////////////////////////////////////
typedef hpx::components::managed_component<
    hpx::performance_counters::server::elapsed_time_counter
> elapsed_time_counter_type;

HPX_REGISTER_DERIVED_COMPONENT_FACTORY_EX(
    elapsed_time_counter_type, elapsed_time_counter,
    "base_performance_counter", hpx::components::factory_enabled)
HPX_DEFINE_GET_COMPONENT_TYPE(
    hpx::performance_counters::server::elapsed_time_counter)

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace server
{
    elapsed_time_counter::elapsed_time_counter(counter_info const& info)
      : base_type_holder(info),
        started_at_(static_cast<boost::int64_t>(util::high_resolution_clock::now()))
    {
        if (info.type_ != counter_elapsed_time) {
            HPX_THROW_EXCEPTION(bad_parameter,
                "elapsed_time_counter::elapsed_time_counter",
                "unexpected counter type specified for elapsed_time_counter");
        }
    }

    void elapsed_time_counter::get_counter_value(counter_value& value)
    {
        // gather the current value
        boost::int64_t now = static_cast<boost::int64_t>(util::high_resolution_clock::now());
        value.value_ = now - started_at_;
        value.scaling_ = 1000000000LL;      // coefficient to get seconds
        value.scale_inverse_ = true;
        value.status_ = status_new_data;
        value.time_ = static_cast<boost::uint64_t>(now);
    }
}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace detail
{
    /// Creation function for uptime counters.
    naming::gid_type uptime_counter_creator(counter_info const& info,
        error_code& ec)
    {
        switch (info.type_) {
        case counter_elapsed_time:
            {
                // verify the validity of the counter instance name
                counter_path_elements paths;
                get_counter_path_elements(info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                // allowed counter names: /runtime(locality#%d/*)/uptime
                if (paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter, "uptime_counter_creator",
                        "invalid counter instance parent name: " +
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
}}}

