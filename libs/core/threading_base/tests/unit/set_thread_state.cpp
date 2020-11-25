////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/modules/testing.hpp>

#include <boost/dynamic_bitset.hpp>

#include <chrono>
#include <cstdint>
#include <iostream>
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
using hpx::threads::thread_data;
using hpx::threads::thread_id_type;

using hpx::find_here;

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
void change_thread_state(std::uint64_t thread)
{
    //    std::cout << "waking up thread (wait_signaled)\n";
    thread_id_type id(reinterpret_cast<thread_data*>(thread));
    set_thread_state(id, hpx::threads::thread_schedule_state::pending,
        hpx::threads::thread_restart_state::signaled);

    //    std::cout << "suspending thread (wait_timeout)\n";
    set_thread_state(id, hpx::threads::thread_schedule_state::suspended,
        hpx::threads::thread_restart_state::timeout);
}

HPX_PLAIN_ACTION(change_thread_state, change_thread_state_action)

///////////////////////////////////////////////////////////////////////////////
void tree_boot(std::uint64_t count, std::uint64_t grain_size,
    id_type const& prefix, std::uint64_t thread);

HPX_PLAIN_ACTION(tree_boot, tree_boot_action)

///////////////////////////////////////////////////////////////////////////////
void tree_boot(std::uint64_t count, std::uint64_t grain_size,
    id_type const& prefix, std::uint64_t thread)
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
    {
        promises.push_back(async<tree_boot_action>(
            prefix, child_count, grain_size, prefix, thread));
    }

    for (std::uint64_t i = 0; i < actors; ++i)
        promises.push_back(async<change_thread_state_action>(prefix, thread));

    detail::wait(promises);
}

///////////////////////////////////////////////////////////////////////////////
void test_dummy_thread(std::uint64_t futures)
{
    std::uint64_t woken = 0, signaled = 0, timed_out = 0;

    while (true)
    {
        hpx::threads::thread_restart_state statex =
            suspend(hpx::threads::thread_schedule_state::suspended);

        if (statex == hpx::threads::thread_restart_state::signaled)
        {
            ++signaled;
            ++woken;
        }

        else if (statex == hpx::threads::thread_restart_state::timeout)
        {
            ++timed_out;
            ++woken;
        }

        else if (statex == hpx::threads::thread_restart_state::terminate)
        {
            std::cout << "woken:     " << woken << "/" << (futures * 2) << "\n"
                      << "signaled:  " << signaled << "/" << futures << "\n"
                      << "timed out: " << timed_out << "/0\n";
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
        id_type const prefix = find_here();

        hpx::threads::thread_init_data data(
            hpx::threads::make_thread_function_nullary(
                hpx::util::deferred_call(&test_dummy_thread, futures)),
            "test_dummy_thread");
        thread_id_type thread = register_thread(data);

        tree_boot(futures, grain_size, prefix,
            reinterpret_cast<std::uint64_t>(thread.get()));

        set_thread_state(thread, hpx::threads::thread_schedule_state::pending,
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

    // clang-format off
    cmdline.add_options()
        ( "futures"
        , value<std::uint64_t>()->default_value(64)
        , "number of futures to invoke")

        ( "grain-size"
        , value<std::uint64_t>()->default_value(4)
        , "grain size of the future tree")
    ;
    // clang-format on

    // Initialize and run HPX
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
