//  Copyright (c) 2007-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/actions_base/detail/invocation_count_registry.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/performance_counters/action_invocation_counter_discoverer.hpp>
#include <hpx/performance_counters/counters.hpp>
#include <hpx/performance_counters/registry.hpp>
#include <hpx/util/regex_from_pattern.hpp>

#include <regex>
#include <string>
#include <utility>

namespace hpx { namespace performance_counters {

    bool action_invocation_counter_discoverer(
        hpx::actions::detail::invocation_count_registry const& registry,
        performance_counters::counter_info const& info,
        performance_counters::counter_path_elements& p,
        performance_counters::discover_counter_func const& f,
        performance_counters::discover_counters_mode mode, error_code& ec)
    {
        using map_type =
            hpx::actions::detail::invocation_count_registry::map_type;

        map_type const& map = registry.registered_counters();

        if (mode == performance_counters::discover_counters_minimal ||
            p.parentinstancename_.empty() || p.instancename_.empty())
        {
            if (p.parentinstancename_.empty())
            {
                p.parentinstancename_ = "locality#*";
                p.parentinstanceindex_ = -1;
            }

            if (p.instancename_.empty())
            {
                p.instancename_ = "total";
                p.instanceindex_ = -1;
            }
        }

        if (p.parameters_.empty())
        {
            if (mode == performance_counters::discover_counters_minimal)
            {
                std::string fullname;
                performance_counters::get_counter_name(p, fullname, ec);
                if (ec)
                    return false;

                performance_counters::counter_info cinfo = info;
                cinfo.fullname_ = std::move(fullname);
                return f(cinfo, ec) && !ec;
            }

            p.parameters_ = "*";
        }

        if (p.parameters_.find_first_of("*?[]") != std::string::npos)
        {
            std::string str_rx(util::regex_from_pattern(p.parameters_, ec));
            if (ec)
                return false;

            bool found_one = false;
            std::regex rx(str_rx);

            for (auto const& e : map)
            {
                if (!std::regex_match(e.first, rx))
                    continue;
                found_one = true;

                // propagate parameters
                std::string fullname;
                performance_counters::counter_path_elements cp = p;
                cp.parameters_ = e.first;

                performance_counters::get_counter_name(cp, fullname, ec);
                if (ec)
                    return false;

                performance_counters::counter_info cinfo = info;
                cinfo.fullname_ = std::move(fullname);

                if (!f(cinfo, ec) || ec)
                    return false;
            }

            if (!found_one)
            {
                // compose a list of known action types
                std::string types;
                for (auto const& e : map)
                {
                    types += "  " + e.first + "\n";
                }

                HPX_THROWS_IF(ec, bad_parameter,
                    "invocation_count_registry::counter_discoverer",
                    hpx::util::format(
                        "action type {} does not match any known type, "
                        "known action types: \n{}",
                        p.parameters_, types));
                return false;
            }

            if (&ec != &throws)
                ec = make_success_code();

            return true;
        }

        // use given action type directly
        map_type::const_iterator it = map.find(p.parameters_);
        if (it == map.end())
        {
            // compose a list of known action types
            std::string types;
            for (auto const& e : map)
            {
                types += "  " + e.first + "\n";
            }

            HPX_THROWS_IF(ec, bad_parameter,
                "invocation_count_registry::counter_discoverer",
                hpx::util::format(
                    "action type {} does not match any known type, "
                    "known action types: \n{}",
                    p.parameters_, types));
            return false;
        }

        // propagate parameters
        std::string fullname;
        performance_counters::get_counter_name(p, fullname, ec);
        if (ec)
            return false;

        performance_counters::counter_info cinfo = info;
        cinfo.fullname_ = std::move(fullname);

        if (!f(cinfo, ec) || ec)
            return false;

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }
}}    // namespace hpx::performance_counters
