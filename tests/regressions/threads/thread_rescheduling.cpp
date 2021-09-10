////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <boost/dynamic_bitset.hpp>

#include <chrono>
#include <cstdint>
#include <vector>

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using std::chrono::milliseconds;

using hpx::naming::id_type;

using hpx::threads::register_thread;

using hpx::async;
using hpx::lcos::future;

using hpx::this_thread::suspend;
using hpx::threads::set_thread_state;
using hpx::threads::thread_id_type;
using hpx::threads::thread_id_ref_type;

using hpx::find_here;
using hpx::init;

///////////////////////////////////////////////////////////////////////////////
namespace detail {
    template <typename T1>
    std::uint64_t wait(std::vector<future<T1>> const& lazy_values,
        std::int32_t suspend_for = 10)
    {
        boost::dynamic_bitset<> handled(lazy_values.size());
        std::uint64_t handled_count = 0;

        while (handled_count < lazy_values.size())
        {
            bool suspended = false;

            for (std::uint64_t i = 0; i < lazy_values.size(); ++i)
            {
                // loop over all lazy_values, executing the next as soon as its
                // value gets available
                if (!handled[i] && lazy_values[i].is_ready())
                {
                    handled[i] = true;
                    ++handled_count;

                    // give thread-manager a chance to look for more work while
                    // waiting
                    suspend();
                    suspended = true;
                }
            }

            // suspend after one full loop over all values, 10ms should be fine
            // (default parameter)
            if (!suspended)
                suspend(milliseconds(suspend_for));
        }
        return handled.count();
    }
}    // namespace detail

///////////////////////////////////////////////////////////////////////////////
void change_thread_state(thread_id_type thread)
{
    set_thread_state(thread, hpx::threads::thread_schedule_state::suspended);
}

///////////////////////////////////////////////////////////////////////////////
void tree_boot(
    std::uint64_t count, std::uint64_t grain_size, thread_id_type thread)
{
    HPX_TEST(grain_size);
    HPX_TEST(count);

    std::vector<future<void>> promises;

    std::uint64_t const actors = (count > grain_size) ? grain_size : count;

    std::uint64_t child_count = 0;
    std::uint64_t children = 0;

    if (count > grain_size)
    {
        for (children = grain_size; children != 0; --children)
        {
            child_count = ((count - grain_size) / children);

            if (child_count >= grain_size)
                break;
        }

        promises.reserve(children + grain_size);
    }
    else
        promises.reserve(count);

    for (std::uint64_t i = 0; i < children; ++i)
        promises.push_back(async(&tree_boot, child_count, grain_size, thread));

    for (std::uint64_t i = 0; i < actors; ++i)
        promises.push_back(async(&change_thread_state, thread));

    detail::wait(promises);
}

bool woken = false;

///////////////////////////////////////////////////////////////////////////////
void test_dummy_thread(std::uint64_t)
{
    while (true)
    {
        hpx::threads::thread_restart_state statex =
            suspend(hpx::threads::thread_schedule_state::suspended);

        if (statex == hpx::threads::thread_restart_state::terminate)
        {
            woken = true;
            return;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    std::uint64_t const futures = vm["futures"].as<std::uint64_t>();
    std::uint64_t const grain_size = vm["grain-size"].as<std::uint64_t>();

    {
        hpx::threads::thread_init_data data(
            hpx::threads::make_thread_function_nullary(
                hpx::util::deferred_call(&test_dummy_thread, futures)),
            "test_dummy_thread");
        thread_id_ref_type thread_id = register_thread(data);
        HPX_TEST_NEQ(thread_id, hpx::threads::invalid_thread_id);

        // Flood the queues with suspension operations before the rescheduling
        // attempt.
        future<void> before =
            async(&tree_boot, futures, grain_size, thread_id.noref());

        set_thread_state(thread_id.noref(),
            hpx::threads::thread_schedule_state::pending,
            hpx::threads::thread_restart_state::signaled);

        // Flood the queues with suspension operations after the rescheduling
        // attempt.
        future<void> after =
            async(&tree_boot, futures, grain_size, thread_id.noref());

        before.get();
        after.get();

        set_thread_state(thread_id.noref(),
            hpx::threads::thread_schedule_state::pending,
            hpx::threads::thread_restart_state::terminate);
    }

    hpx::finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()("futures", value<std::uint64_t>()->default_value(64),
        "number of futures to invoke before and after the rescheduling")

        ("grain-size", value<std::uint64_t>()->default_value(4),
            "grain size of the future tree");

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    HPX_TEST_EQ(0, hpx::init(argc, argv, init_args));

    HPX_TEST(woken);

    return hpx::util::report_errors();
}

#endif
