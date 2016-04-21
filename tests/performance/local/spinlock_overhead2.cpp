//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2011 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/runtime.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/lcos/wait_each.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/actions/continuation.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/util/register_locks.hpp>
#include <hpx/include/async.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/format.hpp>
#include <boost/cstdint.hpp>
#include <boost/chrono/duration.hpp>

#include <stdexcept>
#include <vector>

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;

using hpx::find_here;

using hpx::naming::id_type;

using hpx::lcos::future;
using hpx::async;
using hpx::lcos::wait_each;

using hpx::util::high_resolution_timer;

using hpx::cout;
using hpx::flush;

#define N 100

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
double global_init[N] = {0};
boost::uint64_t num_iterations = 0;

std::size_t k1 = 0;
std::size_t k2 = 0;
std::size_t k3 = 0;

namespace test
{
    struct local_spinlock
    {
        HPX_NON_COPYABLE(local_spinlock);

    private:
        boost::uint64_t v_;

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
                    hpx::this_thread::suspend(boost::chrono::microseconds(1));
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

                    nanosleep( &rqtp, 0 );
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

#if !defined( BOOST_SP_HAS_SYNC )
            boost::uint64_t r = BOOST_INTERLOCKED_EXCHANGE(&v_, 1);
            BOOST_COMPILER_FENCE
#else
            boost::uint64_t r = __sync_lock_test_and_set(&v_, 1);
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

#if !defined( BOOST_SP_HAS_SYNC )
            BOOST_COMPILER_FENCE
            *const_cast<boost::uint64_t volatile*>(&v_) = 0;
#else
            __sync_lock_release(&v_);
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
        num_iterations = vm["delay-iterations"].as<boost::uint64_t>();

        const boost::uint64_t count = vm["futures"].as<boost::uint64_t>();

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
                for (boost::uint64_t i = 0; i < count; ++i)
                    futures.push_back(async<null_action>(here, i));

                wait_each(hpx::util::unwrapped(
                    [] (double r) { global_scratch += r; }),
                    futures);

                // stop the clock
                const double duration = walltime.elapsed();

                if (vm.count("csv"))
                    cout << ( boost::format("%3%,%4%,%5%,%2%\n")
                            % count
                            % duration
                            % k1
                            % k2
                            % k3
                            )
                         << flush;
                else
                    cout << ( boost::format("invoked %1% futures in %2% seconds \
                                (k1 = %3%, k2 = %4%, k3 = %5%)\n")
                            % count
                            % duration
                            % k1
                            % k2
                            % k3
                            )
                         << flush;
            }
        }
    }

    finalize();
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
        , value<boost::uint64_t>()->default_value(500000)
        , "number of futures to invoke")

        ( "delay-iterations"
        , value<boost::uint64_t>()->default_value(0)
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
    return init(cmdline, argc, argv);
}
