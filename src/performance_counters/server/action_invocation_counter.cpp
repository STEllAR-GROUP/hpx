//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/components/runtime_support.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/runtime/actions/invocation_count_registry.hpp>
#include <hpx/util/function.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    // Discoverer function for action invocation counters
    bool action_invocation_counter_discoverer(counter_info const& info,
        discover_counter_func const& f,
        discover_counters_mode mode, error_code& ec)
    {
        performance_counters::counter_info i = info;

        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        bool result =
            hpx::actions::detail::invocation_count_registry::instance().
                counter_discoverer(info, p, f, mode, ec);
        if (!result || ec) return false;

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }

    ///////////////////////////////////////////////////////////////////////////
    // Creation function for action invocation counter
    naming::gid_type action_invocation_counter_creator(
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
                        "action_invocation_counter_creator",
                        "invalid action invocation counter name (instance name "
                        "must not be a valid base counter name)");
                    return naming::invalid_gid;
                }

                if (paths.parameters_.empty()) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "action_invocation_counter_creator",
                        "invalid action invocation counter parameter: must "
                        "specify an action type");
                    return naming::invalid_gid;
                }

                // ask registry
                hpx::util::function_nonser<boost::int64_t(bool)> f =
                    hpx::actions::detail::invocation_count_registry::instance().
                        get_invocation_counter(paths.parameters_);

                return detail::create_raw_counter(info, std::move(f), ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter,
                "action_invocation_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }
}}

