//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/runtime_local/config_entry.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/util/from_string.hpp>

#include <atomic>
#include <string>

void test_get_entry()
{
    std::string val = hpx::get_config_entry("hpx.localities", "42");
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(hpx::util::from_string<int>(val), 1);

    val = hpx::get_config_entry("hpx.localities", 42);
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(hpx::util::from_string<int>(val), 1);
}

std::atomic<bool> invoked_callback(false);

void config_entry_callback(std::string const& key, std::string const& val)
{
    HPX_TEST_EQ(key, std::string("hpx.config.entry.test"));
    HPX_TEST_EQ(val, std::string("test1"));

    HPX_TEST(!invoked_callback.load());
    invoked_callback = true;
}

void test_set_entry()
{
    std::string val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST(val.empty());

    hpx::set_config_entry("hpx.config.entry.test", "test");
    val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(val, std::string("test"));

    hpx::set_config_entry_callback(
        "hpx.config.entry.test", &config_entry_callback);

    hpx::set_config_entry("hpx.config.entry.test", "test1");
    val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(val, std::string("test1"));

    HPX_TEST(invoked_callback.load());
}

int main()
{
    test_get_entry();
    test_set_entry();
    return 0;
}
