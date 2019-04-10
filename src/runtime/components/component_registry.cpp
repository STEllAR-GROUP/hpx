//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2017      Thomas Heller
//  Copyright (c) 2011      Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// make inspect happy: hpxinspect:nodeprecatedname:boost::is_any_of

#include <hpx/runtime/components/component_registry.hpp>

#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/components/component_type.hpp>
#include <hpx/util/find_prefix.hpp>
#include <hpx/util/logging.hpp>
#include <hpx/util/runtime_configuration.hpp>

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/assign/std/vector.hpp>

#include <algorithm>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components { namespace detail
{
    void get_component_info(std::vector<std::string>& fillini,
        std::string const& filepath, bool is_static, char const* name,
        char const* component_string, factory_state_enum state,
        char const* more)
    {
        using namespace boost::assign;
        fillini += std::string("[hpx.components.") + name + "]";
        fillini += std::string("name = ") + component_string;

        if (!is_static)
        {
            if (filepath.empty()) {
                fillini += std::string("path = ") +
                    util::find_prefixes("/hpx", component_string);
            }
            else {
                fillini += std::string("path = ") + filepath;
            }
        }

        switch (state) {
        case factory_enabled:
            fillini += "enabled = 1";
            break;
        case factory_disabled:
            fillini += "enabled = 0";
            break;
        case factory_check:
            fillini += "enabled = $[hpx.components.load_external]";
            break;
        }

        if (is_static) {
            fillini += "static = 1";
        }

        if (more) {
            std::vector<std::string> data;
            boost::split(data, more, boost::is_any_of("\n"));
            std::copy(data.begin(), data.end(), std::back_inserter(fillini));
        }
    }

    bool is_component_enabled(char const* name)
    {
        hpx::util::runtime_configuration const& config = hpx::get_config();
        std::string enabled_entry = config.get_entry(
            std::string("hpx.components.") + name + ".enabled", "0");

        boost::algorithm::to_lower(enabled_entry);
        if (enabled_entry == "no" || enabled_entry == "false" ||
            enabled_entry == "0")
        {
            LRT_(info) << "plugin factory disabled: " << name;
            return false;     // this component has been disabled
        }
        return true;
    }
}}}
