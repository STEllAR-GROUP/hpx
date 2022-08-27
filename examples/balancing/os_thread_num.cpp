//  Copyright (c) 2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/barrier.hpp>
#include <hpx/local/init.hpp>
#include <hpx/local/runtime.hpp>
#include <hpx/local/thread.hpp>
#include <hpx/modules/allocator_support.hpp>
#include <hpx/modules/format.hpp>

#include <boost/lockfree/queue.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>

template <typename T>
using queue =
    boost::lockfree::queue<T, hpx::util::aligned_allocator<std::size_t>>;

using hpx::program_options::options_description;
using hpx::program_options::value;
using hpx::program_options::variables_map;

using hpx::barrier;

using hpx::threads::register_work;
using hpx::threads::thread_init_data;

///////////////////////////////////////////////////////////////////////////////
// we use globals here to prevent the delay from being optimized away
double global_scratch = 0;
std::uint64_t num_iterations = 0;

///////////////////////////////////////////////////////////////////////////////
double delay()
{
    double d = 0.;
    for (std::uint64_t i = 0; i < num_iterations; ++i)
        d += 1 / (2. * i + 1);
    return d;
}

///////////////////////////////////////////////////////////////////////////////
void get_os_thread_num(
    std::shared_ptr<barrier<>> barr, queue<std::size_t>& os_threads)
{
    global_scratch = delay();
    os_threads.push(hpx::get_worker_thread_num());
    barr->arrive_and_wait();
}

///////////////////////////////////////////////////////////////////////////////
typedef std::map<std::size_t, std::size_t> result_map;

typedef std::multimap<std::size_t, std::size_t, std::greater<std::size_t>>
    sorter;

///////////////////////////////////////////////////////////////////////////////
int hpx_main(variables_map& vm)
{
    {
        num_iterations = vm["delay-iterations"].as<std::uint64_t>();

        const bool csv = vm.count("csv");

        const std::size_t pxthreads = vm["pxthreads"].as<std::size_t>();

        result_map results;

        {
            // Have the queue preallocate the nodes.
            queue<std::size_t> os_threads(pxthreads);

            std::shared_ptr<barrier<>> barr =
                std::make_shared<barrier<>>(pxthreads + 1);

            for (std::size_t j = 0; j < pxthreads; ++j)
            {
                thread_init_data data(
                    hpx::threads::make_thread_function_nullary(hpx::bind(
                        &get_os_thread_num, barr, std::ref(os_threads))),
                    "get_os_thread_num", hpx::threads::thread_priority::normal,
                    hpx::threads::thread_schedule_hint(0));
                register_work(data);
            }

            // wait for all HPX threads to enter the barrier
            barr->arrive_and_wait();

            std::size_t shepherd = 0;

            while (os_threads.pop(shepherd))
                ++results[shepherd];
        }

        sorter sort;

        for (result_map::value_type const& result : results)
        {
            sort.insert(sorter::value_type(result.second, result.first));
        }

        for (sorter::value_type const& result : sort)
        {
            if (csv)
                hpx::util::format_to(
                    std::cout, "{1},{2}\n", result.second, result.first)
                    << std::flush;
            else
                hpx::util::format_to(std::cout,
                    "OS-thread {1} ran {2} PX-threads\n", result.second,
                    result.first)
                    << std::flush;
        }
    }

    // initiate shutdown of the runtime system
    hpx::local::finalize();
    return 0;
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options
    options_description cmdline("Usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()("pxthreads", value<std::size_t>()->default_value(128),
        "number of PX-threads to invoke")

        ("delay-iterations", value<std::uint64_t>()->default_value(65536),
            "number of iterations in the delay loop")

            ("csv", "output results as csv (format: OS-thread,PX-threads)");

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = cmdline;

    return hpx::local::init(hpx_main, argc, argv, init_args);
}
