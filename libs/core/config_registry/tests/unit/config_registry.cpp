//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/modules/config_registry.hpp>

#include <cstddef>
#include <iostream>
#include <string>

// Every other module should be able to use this module, so we use none here.
// That means no testing facilities. We use a bare-bones testing macro here.
static std::size_t tests_failed = 0;

void check_test(std::string filename, std::size_t linenumber, bool success)
{
    if (!success)
    {
        ++tests_failed;
        std::cerr << "Test failed in " << filename << " on line " << linenumber
                  << std::endl;
    }
}

#define TEST(expr) check_test(__FILE__, __LINE__, expr)

int report_errors()
{
    if (tests_failed != 0)
    {
        std::cerr << tests_failed << " test" << (tests_failed > 1 ? "s" : "")
                  << " failed" << std::endl;
        return 1;
    }

    return 0;
}

static hpx::config_registry::add_module_config_helper add_config{
    {"static", {"1", "2", "3"}}};

int main()
{
    {
        auto const& configs = hpx::config_registry::get_module_configs();

        TEST(configs.size() == 1);
        TEST(configs[0].module_name == "static");
        TEST(configs[0].config_entries.size() == 3);
        TEST(configs[0].config_entries[0] == "1");
        TEST(configs[0].config_entries[1] == "2");
        TEST(configs[0].config_entries[2] == "3");

        hpx::config_registry::add_module_config({"dynamic", {"4", "5"}});
    }

    {
        auto const& configs = hpx::config_registry::get_module_configs();

        TEST(configs[0].module_name == "static");
        TEST(configs[0].config_entries.size() == 3);
        TEST(configs[0].config_entries[0] == "1");
        TEST(configs[0].config_entries[1] == "2");
        TEST(configs[0].config_entries[2] == "3");

        TEST(configs[1].module_name == "dynamic");
        TEST(configs[1].config_entries.size() == 2);
        TEST(configs[1].config_entries[0] == "4");
        TEST(configs[1].config_entries[1] == "5");
    }

    return report_errors();
}
