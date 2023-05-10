//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/runtime.hpp>
#include <hpx/util/from_string.hpp>

#include <cstddef>
#include <string>
#include <thread>
#include <vector>

void enumerate_threads(std::size_t num_custom_threads)
{
    std::size_t counts[std::size_t(hpx::os_thread_type::custom_thread) + 1] = {
        0};

    bool result =
        hpx::enumerate_os_threads([&counts](hpx::os_thread_data const& data) {
            if (data.type_ != hpx::os_thread_type::unknown)
            {
                HPX_TEST(std::size_t(data.type_) <=
                    std::size_t(hpx::os_thread_type::custom_thread));

                ++counts[std::size_t(data.type_)];
                HPX_TEST(data.label_.find(hpx::get_os_thread_type_name(
                             data.type_)) != std::string::npos);
            }
            return true;
        });
    HPX_TEST(result);

    HPX_TEST_EQ(
        counts[std::size_t(hpx::os_thread_type::main_thread)], std::size_t(1));

    std::size_t num_workers = hpx::get_num_worker_threads();
    HPX_TEST_EQ(
        counts[std::size_t(hpx::os_thread_type::worker_thread)], num_workers);

#ifdef HPX_HAVE_IO_POOL
    std::size_t num_io_threads = hpx::util::from_string<std::size_t>(
        hpx::get_config_entry("hpx.threadpools.io_pool_size", "0"));
    HPX_TEST_EQ(
        counts[std::size_t(hpx::os_thread_type::io_thread)], num_io_threads);
#endif

#ifdef HPX_HAVE_TIMER_POOL
    std::size_t num_timer_threads = hpx::util::from_string<std::size_t>(
        hpx::get_config_entry("hpx.threadpools.timer_pool_size", "0"));
    HPX_TEST_EQ(counts[std::size_t(hpx::os_thread_type::timer_thread)],
        num_timer_threads);
#endif

    HPX_TEST_EQ(counts[std::size_t(hpx::os_thread_type::custom_thread)],
        num_custom_threads);
}

int hpx_main()
{
    enumerate_threads(0);

    auto* rt = hpx::get_runtime_ptr();

    std::thread t([rt]() {
        hpx::register_thread(rt, "custom");
        enumerate_threads(1);
        hpx::unregister_thread(rt);
    });
    t.join();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    hpx::local::init_params init_args;

    HPX_TEST_EQ(hpx::local::init(hpx_main, argc, argv, init_args), 0);

    return hpx::util::report_errors();
}
