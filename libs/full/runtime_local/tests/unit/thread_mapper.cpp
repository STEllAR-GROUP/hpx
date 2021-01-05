//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>
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

#ifdef HPX_HAVE_NETWORKING
    std::size_t num_parcel_threads = 0;
    std::vector<std::string> const parcelport_names = {
        "tcp", "mpi", "libfabric"};
    for (auto parcelport_name : parcelport_names)
    {
        if (hpx::get_config_entry(
                "hpx.parcel." + parcelport_name + ".enable", "0") != "0")
        {
            num_parcel_threads +=
                hpx::util::from_string<std::size_t>(hpx::get_config_entry(
                    "hpx.parcel." + parcelport_name + ".parcel_pool_size",
                    "0"));
        }
    }
    HPX_TEST_EQ(counts[std::size_t(hpx::os_thread_type::parcel_thread)],
        num_parcel_threads);
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

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // make sure networking is enabled
    std::vector<std::string> cfg = {"hpx.expect_connecting_localities=1"};

    hpx::init_params init_args;
    init_args.cfg = cfg;

    HPX_TEST_EQ(hpx::init(argc, argv, init_args), 0);

    return hpx::util::report_errors();
}
