//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2017      Thomas Heller
//  Copyright (c) 2011      Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/runtime/components/component_registry.hpp>

#include <hpx/components_base/component_type.hpp>
#include <hpx/modules/logging.hpp>
#include <hpx/prefix/find_prefix.hpp>
#include <hpx/runtime_configuration/runtime_configuration.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/string_util/classification.hpp>
#include <hpx/string_util/split.hpp>

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
        fillini.emplace_back(std::string("[hpx.components.") + name + "]");
        fillini.emplace_back(std::string("name = ") + component_string);

        if (!is_static)
        {
            if (filepath.empty()) {
                fillini.emplace_back(std::string("path = ") +
                    util::find_prefixes("/hpx", component_string));
            }
            else {
                fillini.emplace_back(std::string("path = ") + filepath);
            }
        }

        switch (state) {
        case factory_enabled:
            fillini.emplace_back("enabled = 1");
            break;
        case factory_disabled:
            fillini.emplace_back("enabled = 0");
            break;
        case factory_check:
            fillini.emplace_back("enabled = $[hpx.components.load_external]");
            break;
        }

        if (is_static) {
            fillini.emplace_back("static = 1");
        }

        if (more) {
            std::vector<std::string> data;
            hpx::string_util::split(data, more, hpx::string_util::is_any_of("\n"));
            std::copy(data.begin(), data.end(), std::back_inserter(fillini));
        }
    }

    bool is_component_enabled(char const* name)
    {
        hpx::util::runtime_configuration const& config = hpx::get_config();
        std::string enabled_entry = config.get_entry(
            std::string("hpx.components.") + name + ".enabled", "0");

        std::transform(enabled_entry.begin(), enabled_entry.end(),
            enabled_entry.begin(), [](char c) { return std::tolower(c); });

        if (enabled_entry == "no" || enabled_entry == "false" ||
            enabled_entry == "0")
        {
            LRT_(info) << "plugin factory disabled: " << name;
            return false;     // this component has been disabled
        }
        return true;
    }
}}}
