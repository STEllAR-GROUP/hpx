//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <string>

void test_get_entry()
{
    std::string val = hpx::get_config_entry("hpx.localities", "42");
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(boost::lexical_cast<int>(val), 1);

    val = hpx::get_config_entry("hpx.localities", 42);
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(boost::lexical_cast<int>(val), 1);
}

void test_set_entry()
{
    std::string val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST(val.empty());

    hpx::set_config_entry("hpx.config.entry.test", "test");
    val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(val, std::string("test"));

    hpx::set_config_entry("hpx.config.entry.test", "test1");
    val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(val, std::string("test1"));
}

int main(int argc, char* argv[])
{
    test_get_entry();
    test_set_entry();
    return 0;
}
