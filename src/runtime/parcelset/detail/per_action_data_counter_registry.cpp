//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_PARCELPORT_ACTION_COUNTERS)
#include <hpx/exception.hpp>
#include <hpx/runtime/parcelset/detail/per_action_data_counter_registry.hpp>
#include <hpx/performance_counters/counter_creators.hpp>
#include <hpx/performance_counters/registry.hpp>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <utility>

#include <boost/format.hpp>
#include <boost/regex.hpp>

namespace hpx { namespace parcelset { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    per_action_data_counter_registry&
    per_action_data_counter_registry::instance()
    {
        hpx::util::static_<per_action_data_counter_registry, tag> registry;
        return registry.get();
    }

    void per_action_data_counter_registry::register_class(std::string name)
    {
        if (name.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "per_action_data_counter_registry::register_class",
                "Cannot register an action with an empty name");
        }

        auto it = map_.find(name);
        if (it == map_.end())
            map_.emplace(std::move(name));
    }

    per_action_data_counter_registry::counter_function_type
        per_action_data_counter_registry::get_counter(
            std::string const& name,
            hpx::util::function_nonser<
                std::int64_t(std::string const&, bool)
            > const& f) const
    {
        map_type::const_iterator it = map_.find(name);
        if (it == map_.end())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "per_action_data_counter_registry::get_receive_counter",
                "unknown action type");
            return nullptr;
        }
        return util::bind(f, name, util::placeholders::_1);
    }

    bool per_action_data_counter_registry::counter_discoverer(
        performance_counters::counter_info const& info,
        performance_counters::counter_path_elements& p,
        performance_counters::discover_counter_func const& f,
        performance_counters::discover_counters_mode mode, error_code& ec)
    {
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
            // if no parameters (action name) is given assume that this counter
            // should report the overall value for all actions
            return performance_counters::locality_counter_discoverer(
                info, f, mode, ec);
        }

        if (p.parameters_.find_first_of("*?[]") != std::string::npos)
        {
            std::string str_rx(
                performance_counters::detail::regex_from_pattern(
                    p.parameters_, ec));
            if (ec) return false;

            bool found_one = false;
            boost::regex rx(str_rx, boost::regex::perl);

            map_type::const_iterator end = map_.end();
            for (map_type::const_iterator it = map_.begin(); it != end; ++it)
            {
                if (!boost::regex_match(*it, rx))
                    continue;
                found_one = true;

                // propagate parameters
                std::string fullname;
                performance_counters::counter_path_elements cp = p;
                cp.parameters_ = *it;

                performance_counters::get_counter_name(cp, fullname, ec);
                if (ec) return false;

                performance_counters::counter_info cinfo = info;
                cinfo.fullname_ = fullname;

                if (!f(cinfo, ec) || ec)
                    return false;
            }

            if (!found_one)
            {
                // compose a list of known action types
                std::string types;
                map_type::const_iterator end = map_.end();
                for (map_type::const_iterator it = map_.begin(); it != end; ++it)
                {
                    types += "  " + *it + "\n";
                }

                HPX_THROWS_IF(ec, bad_parameter,
                    "per_action_data_counter_registry::counter_discoverer",
                    boost::str(boost::format(
                        "action type %s does not match any known type, "
                        "known action types: \n%s") % p.parameters_ % types));
                return false;
            }

            if (&ec != &throws)
                ec = make_success_code();

            return true;
        }

        // use given action type directly
        map_type::const_iterator it = map_.find(p.parameters_);
        if (it == map_.end())
        {
            // compose a list of known action types
            std::string types;
            map_type::const_iterator end = map_.end();
            for (map_type::const_iterator it = map_.begin(); it != end; ++it)
            {
                types += "  " + *it + "\n";
            }

            HPX_THROWS_IF(ec, bad_parameter,
                "per_action_data_counter_registry::counter_discoverer",
                boost::str(boost::format(
                    "action type %s does not match any known type, "
                    "known action types: \n%s") % p.parameters_ % types));
            return false;
        }

        // propagate parameters
        std::string fullname;
        performance_counters::get_counter_name(p, fullname, ec);
        if (ec) return false;

        performance_counters::counter_info cinfo = info;
        cinfo.fullname_ = fullname;

        if (!f(cinfo, ec) || ec)
            return false;

        if (&ec != &throws)
            ec = make_success_code();

        return true;
    }
}}}

#endif
