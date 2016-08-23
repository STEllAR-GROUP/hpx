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

#include <boost/dynamic_bitset.hpp>

#include <chrono>
#include <cstdint>
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
using hpx::threads::thread_data;
using hpx::this_thread::suspend;
using hpx::threads::set_thread_state;
using hpx::threads::thread_state_ex_enum;
using hpx::threads::pending;
using hpx::threads::suspended;
using hpx::threads::wait_signaled;
using hpx::threads::wait_timeout;
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
    std::uint64_t thread
    )
{
//    std::cout << "waking up thread (wait_signaled)\n";
    thread_id_type id(reinterpret_cast<thread_data*>(thread));
    set_thread_state(id, pending, wait_signaled);

//    std::cout << "suspending thread (wait_timeout)\n";
    set_thread_state(id, suspended, wait_timeout);
}

HPX_PLAIN_ACTION(change_thread_state, change_thread_state_action)

///////////////////////////////////////////////////////////////////////////////
void tree_boot(
    std::uint64_t count
  , std::uint64_t grain_size
  , id_type const& prefix
  , std::uint64_t thread
    );

HPX_PLAIN_ACTION(tree_boot, tree_boot_action)

///////////////////////////////////////////////////////////////////////////////
void tree_boot(
    std::uint64_t count
  , std::uint64_t grain_size
  , id_type const& prefix
  , std::uint64_t thread
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
    {
        promises.push_back(async<tree_boot_action>
            (prefix, child_count, grain_size, prefix, thread));
    }

    for (std::uint64_t i = 0; i < actors; ++i)
        promises.push_back(async<change_thread_state_action>(prefix, thread));

    detail::wait(promises);
}

///////////////////////////////////////////////////////////////////////////////
void test_dummy_thread(
    std::uint64_t futures
    )
{
    std::uint64_t woken = 0
                  , signaled = 0
                  , timed_out = 0;

    while (true)
    {
        thread_state_ex_enum statex = suspend(suspended);

        if (statex == wait_signaled)
        {
            ++signaled;
            ++woken;
        }

        else if (statex == wait_timeout)
        {
            ++timed_out;
            ++woken;
        }

        else if (statex == wait_terminate)
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

        thread_id_type thread = register_thread_nullary
            (hpx::util::bind(&test_dummy_thread, futures));

        tree_boot(futures, grain_size, prefix,
            reinterpret_cast<std::uint64_t>(thread.get()));

        set_thread_state(thread, pending, wait_terminate);
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
        , "number of futures to invoke")

        ( "grain-size"
        , value<std::uint64_t>()->default_value(4)
        , "grain size of the future tree")
    ;

    // Initialize and run HPX
    return init(cmdline, argc, argv);
}

