//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/util/function.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    // Extract the current number of instances for the given component type
    boost::int64_t get_instance_count(components::component_type type)
    {
        using components::stubs::runtime_support;
        return boost::int64_t(runtime_support::get_instance_count(find_here(), type));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Creation function for instance counter
    naming::gid_type component_instance_counter_creator(
        counter_info const& info, error_code& ec)
    {
        switch (info.type_) {
        case counter_raw:
            {
                counter_path_elements paths;
                get_counter_path_elements(info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                if (paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "component_instance_counter_creator",
                        "invalid instance counter name (instance name must not "
                        "be a valid base counter name)");
                    return naming::invalid_gid;
                }

                if (paths.parameters_.empty()) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "component_instance_counter_creator",
                        "invalid instance counter parameter: must specify "
                        "a component type");
                    return naming::invalid_gid;
                }

                // ask AGAS to resolve the component type
                components::component_type type =
                    naming::get_agas_client().get_component_id(paths.parameters_);

                if (type == components::component_invalid) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "component_instance_counter_creator",
                        "invalid component type as counter parameter: " +
                        paths.parameters_);
                    return naming::invalid_gid;
                }

                hpx::util::function_nonser<boost::int64_t()> f =
                    util::bind(&get_instance_count, type);
                return create_raw_counter(info, std::move(f), ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter, "component_instance_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }
}}}

