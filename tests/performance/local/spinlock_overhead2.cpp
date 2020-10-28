//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)

#include <hpx/actions/continuation.hpp>
#include <hpx/actions_base/plain_action.hpp>
#include <hpx/async_combinators/wait_each.hpp>
#include <hpx/execution_base/register_locks.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/async.hpp>
#include <hpx/iostream.hpp>
#include <hpx/modules/format.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/timing.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

using hpx::program_options::variables_map;
using hpx::program_options::options_description;
using hpx::program_options::value;

using hpx::find_here;

using hpx::naming::id_type;

using hpx::lcos::future;
using hpx::async;
using hpx::lcos::wait_each;

using hpx::chrono::high_resolution_timer;

using hpx::cout;
using hpx::flush;

#define N 100

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
double global_init[N] = {0};
std::uint64_t num_iterations = 0;

std::size_t k1 = 0;
std::size_t k2 = 0;
std::size_t k3 = 0;

namespace test
{
    struct local_spinlock
    {
    private:
        std::uint64_t v_;

        ///////////////////////////////////////////////////////////////////////////
        static void yield(std::size_t k)
        {
            if (k < k1)
            {
            }
            if(k < k2)
            {
#if defined(BOOST_SMT_PAUSE)
                BOOST_SMT_PAUSE
#endif
            }
            else if(k < k3 || k & 1)
            {
                /*
                if(hpx::threads::get_self_ptr())
                {
                    hpx::this_thread::suspend();
                }
                else
                */
                {
#if defined(HPX_WINDOWS)
                    Sleep(0);
#elif defined(BOOST_HAS_PTHREADS)
                    sched_yield();
#else
#endif
                }
            }
            else
            {
                /*
                if (hpx::threads::get_self_ptr())
                {
                    hpx::this_thread::suspend(std::chrono::microseconds(1));
                }
                else
                */
                {
#if defined(HPX_WINDOWS)
                    Sleep(1);
#elif defined(BOOST_HAS_PTHREADS)
                    // g++ -Wextra warns on {} or {0}
                    struct timespec rqtp = { 0, 0 };

                    // POSIX says that timespec has tv_sec and tv_nsec
                    // But it doesn't guarantee order or placement

                    rqtp.tv_sec = 0;
                    rqtp.tv_nsec = 1000;

                    nanosleep( &rqtp, nullptr );
#else
#endif
                }
            }
        }

    public:
        local_spinlock() : v_(0)
        {
            HPX_ITT_SYNC_CREATE(this, "test::local_spinlock", "");
        }

        local_spinlock(local_spinlock const&) = delete;
        local_spinlock& operator=(local_spinlock const&) = delete;

        ~local_spinlock()
        {
            HPX_ITT_SYNC_DESTROY(this);
        }

        void lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

            for (std::size_t k = 0; !try_lock(); ++k)
            {
                local_spinlock::yield(k);
            }

            HPX_ITT_SYNC_ACQUIRED(this);
            hpx::util::register_lock(this);
        }

        bool try_lock()
        {
            HPX_ITT_SYNC_PREPARE(this);

#if defined(BOOST_SP_HAS_SYNC_INTRINSICS) || defined(BOOST_SP_HAS_SYNC)
            std::uint64_t r = __sync_lock_test_and_set(&v_, 1);
#else
            std::uint64_t r = BOOST_INTERLOCKED_EXCHANGE(&v_, 1);
            HPX_COMPILER_FENCE;
#endif

            if (r == 0) {
                HPX_ITT_SYNC_ACQUIRED(this);
                hpx::util::register_lock(this);
                return true;
            }

            HPX_ITT_SYNC_CANCEL(this);
            return false;
        }

        void unlock()
        {
            HPX_ITT_SYNC_RELEASING(this);

#if defined(BOOST_SP_HAS_SYNC_INTRINSICS) || defined(BOOST_SP_HAS_SYNC)
            __sync_lock_release(&v_);
#else
            HPX_COMPILER_FENCE;
            *const_cast<std::uint64_t volatile*>(&v_) = 0;
#endif

            HPX_ITT_SYNC_RELEASED(this);
            hpx::util::unregister_lock(this);
        }
    };
}

test::local_spinlock mtx[N];

///////////////////////////////////////////////////////////////////////////////
double null_function(std::size_t i)
{
    double d = 0.;
    std::size_t idx = i % N;
    {
        std::lock_guard<test::local_spinlock> l(mtx[idx]);
        d = global_init[idx];
    }
    for (double j = 0; j < num_iterations; ++j)
    {
        d += 1 / (2. * j + 1);
    }
    {
        std::lock_guard<test::local_spinlock> l(mtx[idx]);
        global_init[idx] = d;
    }
    return d;
}

HPX_PLAIN_ACTION(null_function, null_action)

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        num_iterations = vm["delay-iterations"].as<std::uint64_t>();

        const std::uint64_t count = vm["futures"].as<std::uint64_t>();

        k1 = vm["k1"].as<std::size_t>();
        k2 = vm["k2"].as<std::size_t>();
        k3 = vm["k3"].as<std::size_t>();

        const id_type here = find_here();

        if (HPX_UNLIKELY(0 == count))
            throw std::logic_error("error: count of 0 futures specified\n");

        std::vector<future<double> > futures;

        futures.reserve(count);

        // start the clock

        /*
        k1 = 1;
        k2 = 2;
        for(; k1 < 256; k1 = k1 * 2)*/
        {
            //for(; k2 < 2056; k2 = k2 * 2)
            {
                high_resolution_timer walltime;
                for (std::uint64_t i = 0; i < count; ++i)
                    futures.push_back(async<null_action>(here, i));

                wait_each(hpx::util::unwrapping(
                    [] (double r) { global_scratch += r; }),
                    futures);

                // stop the clock
                const double duration = walltime.elapsed();

                if (vm.count("csv"))
                    hpx::util::format_to(cout,
                        "{3},{4},{5},{2}\n",
                        count,
                        duration,
                        k1,
                        k2,
                        k3
                    ) << flush;
                else
                    hpx::util::format_to(cout,
                        "invoked {1} futures in {2} seconds "
                        "(k1 = {3}, k2 = {4}, k3 = {5})\n",
                        count,
                        duration,
                        k1,
                        k2,
                        k3
                    ) << flush;
                hpx::util::print_cdash_timing("Spinlock2", duration);
            }
        }
    }

    hpx::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(
    int argc
  , char* argv[]
    )
{
    // Configure application-specific options.
    options_description cmdline("usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "futures"
        , value<std::uint64_t>()->default_value(500000)
        , "number of futures to invoke")

        ( "delay-iterations"
        , value<std::uint64_t>()->default_value(0)
        , "number of iterations in the delay loop")

        ( "k1"
        , value<std::size_t>()->default_value(4)
        , "")

        ( "k2"
        , value<std::size_t>()->default_value(16)
        , "")

        ( "k3"
        , value<std::size_t>()->default_value(32)
        , "")

        ( "csv"
        , "output results as csv (format: count,duration)")
        ;

    // Initialize and run HPX.
    hpx::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::init(argc, argv, init_args);
}
#endif
