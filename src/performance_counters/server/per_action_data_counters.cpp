//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/runtime/parcelset/detail/per_action_data_counter_registry.hpp>
#include <hpx/util/function.hpp>

#include <cstdint>
#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace performance_counters
{
    ///////////////////////////////////////////////////////////////////////////
    // Discoverer function for per-action parcel data counters
    bool per_action_data_counter_discoverer(counter_info const& info,
        discover_counter_func const& f, discover_counters_mode mode,
        hpx::parcelset::detail::per_action_data_counter_registry& registry,
        error_code& ec)
    {
        // compose the counter name templates
        performance_counters::counter_path_elements p;
        performance_counters::counter_status status =
            get_counter_path_elements(info.fullname_, p, ec);
        if (!status_is_valid(status)) return false;

        bool result = registry.counter_discoverer(info, p, f, mode, ec);
        if (!result || ec) return false;

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }

    bool per_action_data_counter_discoverer(counter_info const& info,
        discover_counter_func const& f, discover_counters_mode mode,
        error_code& ec)
    {
        using hpx::parcelset::detail::per_action_data_counter_registry;
        return per_action_data_counter_discoverer(info, f, mode,
            per_action_data_counter_registry::instance(), ec);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Creation function for per-action parcel data counters
    naming::gid_type per_action_data_counter_creator(counter_info const& info,
        hpx::parcelset::detail::per_action_data_counter_registry& registry,
        hpx::util::function_nonser<
            std::int64_t(std::string const&, bool)
        > const& counter_func,
        error_code& ec)
    {
        switch (info.type_) {
        case counter_raw:
            {
                counter_path_elements paths;
                get_counter_path_elements(info.fullname_, paths, ec);
                if (ec) return naming::invalid_gid;

                if (paths.parentinstance_is_basename_) {
                    HPX_THROWS_IF(ec, bad_parameter,
                        "per_action_data_counter_creator",
                        "invalid action invocation counter name (instance name "
                        "must not be a valid base counter name)");
                    return naming::invalid_gid;
                }

                if (paths.parameters_.empty()) {
                    // if no parameters (action name) is given assume that this
                    // counter should report the overall value for all actions
                    using util::placeholders::_1;
                    auto const& f =
                        util::bind(counter_func, paths.parameters_, _1);
                    return performance_counters::locality_raw_counter_creator(
                        info, f,ec);
                }

                // ask registry
                hpx::util::function_nonser<std::int64_t(bool)> f =
                    registry.get_counter(paths.parameters_, counter_func);

                return detail::create_raw_counter(info, std::move(f), ec);
            }
            break;

        default:
            HPX_THROWS_IF(ec, bad_parameter,
                "per_action_data_counter_creator",
                "invalid counter type requested");
            return naming::invalid_gid;
        }
    }

    naming::gid_type per_action_data_counter_creator(
        counter_info const& info,
        hpx::util::function_nonser<
            std::int64_t(std::string const&, bool)
        > const& f,
        error_code& ec)
    {
        using hpx::parcelset::detail::per_action_data_counter_registry;
        return per_action_data_counter_creator(info,
            per_action_data_counter_registry::instance(), f, ec);
    }
}}

#endif
