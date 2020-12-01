//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/util/from_string.hpp>

#include <algorithm>
#include <cstddef>
#include <map>
#include <mutex>
#include <string>

///////////////////////////////////////////////////////////////////////////////
std::mutex mtx;
std::multimap<std::string, std::size_t> threads;

std::size_t count_registrations = 0;
std::size_t count_deregistrations = 0;

///////////////////////////////////////////////////////////////////////////////
void on_thread_start(std::size_t global_thread_num,
    std::size_t /* local_thread_num */, char const* /* pool_name */,
    char const* name)
{
    std::lock_guard<std::mutex> l(mtx);

    // threads shouldn't be registered twice
    auto it = threads.find(name);
    HPX_TEST(it == threads.end() || it->second != global_thread_num);

    threads.emplace(name, global_thread_num);

    ++count_registrations;
}

void on_thread_stop(std::size_t global_thread_num,
    std::size_t /* local_thread_num */, char const* /* pool_name */,
    char const* name)
{
    std::lock_guard<std::mutex> l(mtx);

    // make sure this thread was registered
    auto p = threads.equal_range(name);
    HPX_TEST(p.first != p.second);

    bool found_thread = false;
    for (auto it = p.first; it != p.second; ++it)
    {
        if (it->second == global_thread_num)
        {
            found_thread = true;
        }
    }
    HPX_TEST(found_thread);

    ++count_deregistrations;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // verify that all kernel threads were registered
    std::lock_guard<std::mutex> l(mtx);

    auto p = threads.equal_range("main-thread");
    HPX_TEST_EQ(std::distance(p.first, p.second), 1);

    p = threads.equal_range("worker-thread");
    HPX_TEST_EQ(std::size_t(std::distance(p.first, p.second)),
        hpx::get_num_worker_threads());

    p = threads.equal_range("timer-thread");
    auto cfg = hpx::get_config_entry("hpx.threadpools.timer_pool_size", "0");
    HPX_TEST_EQ(
        std::distance(p.first, p.second), hpx::util::from_string<int>(cfg));

#if defined(HPX_HAVE_NETWORKING)
    if (hpx::is_networking_enabled())
    {
        p = threads.equal_range("parcel-thread");
        cfg = hpx::get_config_entry("hpx.threadpools.parcel_pool_size", "0");
        HPX_TEST_EQ(
            std::distance(p.first, p.second), hpx::util::from_string<int>(cfg));
    }
#endif

    p = threads.equal_range("io-thread");
    cfg = hpx::get_config_entry("hpx.threadpools.io_pool_size", "0");
    HPX_TEST_EQ(
        std::distance(p.first, p.second), hpx::util::from_string<int>(cfg));

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    auto on_start = hpx::register_thread_on_start_func(&on_thread_start);
    HPX_TEST(on_start.empty());

    auto on_stop = hpx::register_thread_on_stop_func(&on_thread_stop);
    HPX_TEST(on_stop.empty());

    HPX_TEST_EQ(0, hpx::init(argc, argv));

    HPX_TEST_EQ(count_registrations, count_deregistrations);

    return hpx::util::report_errors();
}
