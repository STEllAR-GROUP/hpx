//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2013 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/util/high_resolution_timer.hpp>
#include <hpx/lcos/local/barrier.hpp>
#include <hpx/runtime/agas/interface.hpp>
#include <hpx/runtime/agas/addressing_service.hpp>
#include <hpx/util/activate_counters.hpp>
//#include <hpx/include/iostreams.hpp>

#include <stdexcept>

#include <boost/format.hpp>
#include <boost/cstdint.hpp>

#include "worker_timed.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::get_os_thread_count;

using hpx::applier::register_work;

using hpx::this_thread::suspend;
using hpx::threads::get_thread_count;

using hpx::util::high_resolution_timer;

using hpx::reset_active_counters;
using hpx::stop_active_counters;

//using hpx::cout;
//using hpx::flush;

using std::cout;
using std::flush;

///////////////////////////////////////////////////////////////////////////////
// Command-line variables.
boost::uint64_t tasks = 500000;
boost::uint64_t delay = 0;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
void print_results(
    boost::uint64_t cores
  , double walltime
  , std::vector<std::string> const& counters
  , boost::shared_ptr<hpx::util::activate_counters> ac 
    )
{
    if (header)
    {
        cout << "# HPX " << hpx::build_string() << " "
                         << hpx::build_type() << "\n"
             << "# Boost " << hpx::boost_version() << "\n"
             << "#\n";

        // Note that if we change the number of fields above, we have to
        // change the constant that we add when printing out the field # for
        // performance counters below (e.g. the last_index part).
        cout << "# 0: OS-threads - Independent Variable\n"
                "# 1: Tasks - Independent Variable\n"
                "# 2: Delay [micro-seconds] - Independent Variable\n"
                "# 3: Total Walltime [seconds]\n"
                ;

        boost::uint64_t const last_index = 3;

        for (boost::uint64_t i = 0; i < counters.size(); ++i)
        {
            cout << "# " << (i + 1 + last_index) << ": " << ac->name(i);

            if (!ac->unit_of_measure(i).empty())
                cout << " [" << ac->unit_of_measure(i) << "]";

            cout << "\n";            
        }
    }

    std::string const cores_str = boost::str(boost::format("%lu") % cores);
    std::string const tasks_str = boost::str(boost::format("%lu") % tasks);
    std::string const delay_str = boost::str(boost::format("%lu") % delay);

    cout << ( boost::format("%lu %lu %lu %.14g")
            % cores
            % tasks 
            % delay 
            % walltime);

    if (ac)
    {   
        hpx::util::activate_counters::counter_values_type values
            = ac->evaluate_counters();

        for (boost::uint64_t i = 0; i < counters.size(); ++i)
            cout << ( boost::format(" %.14g")
                    % values[i].get().get_value<double>());
    }

    cout << "\n";
}

///////////////////////////////////////////////////////////////////////////////
// avoid having one single volatile variable to become a contention point
int invoke_worker_timed(double delay_sec)
{
    volatile int i = 0;
    worker_timed(delay_sec, &i);
    return i;
}

///////////////////////////////////////////////////////////////////////////////
void wait_for_tasks(hpx::lcos::local::barrier& bar)
{
    BOOST_ASSERT(get_thread_count(hpx::threads::thread_priority_normal) == 1);

    bar.wait();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        if (vm.count("no-header"))
            header = false;

        std::vector<std::string> counters;
        if (vm.count("counter"))
            counters = vm["counter"].as<std::vector<std::string> >();

        if (0 == tasks)
            throw std::invalid_argument("count of 0 tasks specified\n");

        boost::shared_ptr<hpx::util::activate_counters> ac;

        if (!counters.empty())
            ac.reset(new hpx::util::activate_counters(counters));

        // Start the clock.
        high_resolution_timer t;

        for (boost::uint64_t i = 0; i < tasks; ++i)
            register_work(boost::bind(&invoke_worker_timed
                                    , double(delay) * 1e-6)
              , "invoke_worker_timed");

        // Schedule a low-priority thread; when it is executed, it checks to
        // make sure all the tasks (which are normal priority) have been 
        // executed, and then it
        hpx::lcos::local::barrier bar(2);

        register_work(boost::bind(&wait_for_tasks, boost::ref(bar)),
            "wait_for_tasks", hpx::threads::pending,
            hpx::threads::thread_priority_low);

        bar.wait();

        // Stop the clock
        double time_elapsed = t.elapsed();

        print_results(get_os_thread_count(), time_elapsed, counters, ac);
    }

    return finalize();
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
        ( "tasks"
        , value<boost::uint64_t>(&tasks)->default_value(500000)
        , "number of tasks to invoke")

        ( "delay"
        , value<boost::uint64_t>(&delay)->default_value(0)
        , "time in micro-seconds for the delay loop")

        ( "counter"
        , value<std::vector<std::string> >()->composing()
        , "activate and report the specified performance counter")

        ( "no-header"
        , "do not print out the csv header row")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}

