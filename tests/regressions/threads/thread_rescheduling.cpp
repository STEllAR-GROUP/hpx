////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/plain_actions.hpp>
#include <hpx/include/async.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/dynamic_bitset.hpp>

#include <cstdint>
#include <chrono>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using std::chrono::milliseconds;

using hpx::naming::id_type;

using hpx::applier::register_thread_nullary;

using hpx::lcos::future;
using hpx::async;

using hpx::threads::thread_id_type;
using hpx::this_thread::suspend;
using hpx::threads::set_thread_state;
using hpx::threads::thread_state_ex_enum;
using hpx::threads::pending;
using hpx::threads::suspended;
using hpx::threads::wait_signaled;
using hpx::threads::wait_terminate;

using hpx::init;
using hpx::finalize;
using hpx::find_here;

///////////////////////////////////////////////////////////////////////////////
namespace detail
{
    template <typename T1>
    std::uint64_t wait(
        std::vector<future<T1> > const& lazy_values
      , std::int32_t suspend_for = 10
        )
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
}

///////////////////////////////////////////////////////////////////////////////
void change_thread_state(
    thread_id_type thread
    )
{
    set_thread_state(thread, suspended);
}

///////////////////////////////////////////////////////////////////////////////
void tree_boot(
    std::uint64_t count
  , std::uint64_t grain_size
  , thread_id_type thread
    )
{
    HPX_ASSERT(grain_size);
    HPX_ASSERT(count);

    std::vector<future<void> > promises;

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

///////////////////////////////////////////////////////////////////////////////
void test_dummy_thread(
    std::uint64_t
    )
{
    bool woken = false;

    while (true)
    {
        thread_state_ex_enum statex = suspend(suspended);

        woken = true;

        if (statex == wait_terminate)
        {
            HPX_TEST(woken);
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
        thread_id_type thread_id = register_thread_nullary(
            hpx::util::bind(&test_dummy_thread, futures));
        HPX_TEST(thread_id != hpx::threads::invalid_thread_id);

        // Flood the queues with suspension operations before the rescheduling
        // attempt.
        future<void> before = async(&tree_boot, futures, grain_size, thread_id);

        set_thread_state(thread_id, pending, wait_signaled);

        // Flood the queues with suspension operations after the rescheduling
        // attempt.
        future<void> after = async(&tree_boot, futures, grain_size, thread_id);

        before.get();
        after.get();

        set_thread_state(thread_id, pending, wait_terminate);
    }

    finalize();

    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "futures"
        , value<std::uint64_t>()->default_value(64)
        , "number of futures to invoke before and after the rescheduling")

        ( "grain-size"
        , value<std::uint64_t>()->default_value(4)
        , "grain size of the future tree")
    ;

    // Initialize and run HPX
    HPX_TEST_EQ(0, init(cmdline, argc, argv));

    return hpx::util::report_errors();
}

