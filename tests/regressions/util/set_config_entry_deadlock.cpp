//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/runtime/config_entry.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

boost::atomic<bool> invoked_callback(false);

void config_entry_callback()
{
    // this used to cause a deadlock in the config registry
    std::string val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST_EQ(val, std::string("test1"));

    HPX_TEST(!invoked_callback.load());
    invoked_callback = true;
}

int hpx_main(int argc, char* argv[])
{
    std::string val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST(val.empty());

    hpx::set_config_entry("hpx.config.entry.test", "test");
    val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(val, std::string("test"));

    hpx::set_config_entry_callback("hpx.config.entry.test",
        hpx::util::bind(&config_entry_callback));

    hpx::set_config_entry("hpx.config.entry.test", "test1");
    HPX_TEST(invoked_callback.load());

    val = hpx::get_config_entry("hpx.config.entry.test", "");
    HPX_TEST(!val.empty());
    HPX_TEST_EQ(val, std::string("test1"));

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ(hpx::init(argc, argv), 0);
    return hpx::util::report_errors();
}

