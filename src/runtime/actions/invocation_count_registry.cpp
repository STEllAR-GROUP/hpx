//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_fwd.hpp>
#include <hpx/exception.hpp>
#include <hpx/runtime/actions/invocation_count_registry.hpp>
#include <hpx/performance_counters/registry.hpp>

#include <string>

#include <boost/format.hpp>
#include <boost/regex.hpp>

namespace hpx { namespace actions { namespace detail
{
    invocation_count_registry& invocation_count_registry::local_instance()
    {
        hpx::util::static_<invocation_count_registry, local_tag> registry;
        return registry.get();
    }

    invocation_count_registry& invocation_count_registry::remote_instance()
    {
        hpx::util::static_<invocation_count_registry, remote_tag> registry;
        return registry.get();
    }

    void invocation_count_registry::register_class(std::string const& name,
        get_invocation_count_type fun)
    {
        if (name.empty())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "invocation_count_registry::register_class",
                "Cannot register an action with an empty name");
        }

        auto it = map_.find(name);
        if (it == map_.end())
        {
#if !defined(HPX_GCC_VERSION) || HPX_GCC_VERSION >= 408000
            map_.emplace(name, fun);
#else
            map_.insert(map_type::value_type(name, fun));
#endif
        }
    }

    invocation_count_registry::get_invocation_count_type
        invocation_count_registry::get_invocation_counter(
            std::string const& name) const
    {
        map_type::const_iterator it = map_.find(name);
        if (it == map_.end())
        {
            HPX_THROW_EXCEPTION(bad_parameter,
                "invocation_count_registry::get_invocation_counter",
                "unknown action type");
            return 0;
        }
        return (*it).second;
    }

    bool invocation_count_registry::counter_discoverer(
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
                if (!boost::regex_match((*it).first, rx))
                    continue;
                found_one = true;

                // propagate parameters
                std::string fullname;
                performance_counters::counter_path_elements cp = p;
                cp.parameters_ = (*it).first;

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
                    types += "  " + (*it).first + "\n";
                }

                HPX_THROWS_IF(ec, bad_parameter,
                    "invocation_count_registry::counter_discoverer",
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
                types += "  " + (*it).first + "\n";
            }

            HPX_THROWS_IF(ec, bad_parameter,
                "invocation_count_registry::counter_discoverer",
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

