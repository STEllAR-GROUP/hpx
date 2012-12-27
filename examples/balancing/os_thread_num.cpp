//  Copyright (c) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/lcos/local/barrier.hpp>
#include <hpx/util/lockfree/fifo.hpp>

#include <map>

#include <boost/foreach.hpp>

using boost::lockfree::fifo;

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::lcos::local::barrier;

using hpx::threads::threadmanager_base;
using hpx::threads::pending;
using hpx::threads::thread_priority_normal;

using hpx::applier::register_work;

using hpx::init;
using hpx::finalize;

using hpx::cout;
using hpx::flush;

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
boost::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double delay()
{
    double d = 0.;
    for (boost::uint64_t i = 0; i < num_iterations; ++i)
        d += 1 / (2. * i + 1);
    return d;
}

///////////////////////////////////////////////////////////////////////////////
void get_os_thread_num(barrier& barr, fifo<std::size_t>& os_threads)
{
    global_scratch = delay();
    os_threads.enqueue(hpx::get_worker_thread_num());
    barr.wait();
}


///////////////////////////////////////////////////////////////////////////////
typedef std::map<std::size_t, std::size_t>
    result_map;

typedef std::multimap<std::size_t, std::size_t, std::greater<std::size_t> >
    sorter;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        num_iterations = vm["delay-iterations"].as<boost::uint64_t>();

        const bool csv = vm.count("csv");

        const std::size_t pxthreads = vm["pxthreads"].as<std::size_t>();

        result_map results;

        {
            // Have the fifo preallocate the nodes.
            fifo<std::size_t> os_threads(pxthreads);

            barrier barr(pxthreads + 1);

            for (std::size_t j = 0; j < pxthreads; ++j)
            {
                register_work(boost::bind(&get_os_thread_num
                                        , boost::ref(barr)
                                        , boost::ref(os_threads))
                  , "get_os_thread_num"
                  , pending
                  , thread_priority_normal
                  , 0);
            }

            barr.wait(); // wait for all PX threads to enter the barrier

            std::size_t shepherd = 0;

            while (os_threads.dequeue(shepherd))
                ++results[shepherd];
        }

        sorter sort;

        BOOST_FOREACH(result_map::value_type const& result, results)
        {
            sort.insert(sorter::value_type(result.second, result.first));
        }

        BOOST_FOREACH(sorter::value_type const& result, sort)
        {
            if (csv)
                cout << ( boost::format("%1%,%2%\n")
                        % result.second
                        % result.first)
                     << flush;
            else
                cout << ( boost::format("OS-thread %1% ran %2% PX-threads\n")
                        % result.second
                        % result.first)
                     << flush;
        }
    }

    // initiate shutdown of the runtime system
    finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description
       cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        ( "pxthreads"
        , value<std::size_t>()->default_value(128)
        , "number of PX-threads to invoke")

        ( "delay-iterations"
        , value<boost::uint64_t>()->default_value(65536)
        , "number of iterations in the delay loop")

        ( "csv"
        , "output results as csv (format: OS-thread,PX-threads)")
        ;

    // Initialize and run HPX
    return init(cmdline, argc, argv);
}

