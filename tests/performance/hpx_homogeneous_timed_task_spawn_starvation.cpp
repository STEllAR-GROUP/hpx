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

#include <stdexcept>

#include <boost/format.hpp>
#include <boost/cstdint.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include "worker_timed.hpp"

using boost::program_options::variables_map;
using boost::program_options::options_description;
using boost::program_options::value;

using hpx::init;
using hpx::finalize;
using hpx::get_os_thread_count;

using hpx::applier::register_work;
using hpx::applier::register_thread_nullary;

using hpx::this_thread::suspend;
using hpx::threads::get_thread_count;

using hpx::util::high_resolution_timer;

using hpx::reset_active_counters;
using hpx::stop_active_counters;

using std::cout;
using std::flush;

///////////////////////////////////////////////////////////////////////////////
// Command-line variables.
boost::uint64_t tasks = 500000;
boost::uint64_t delay = 0;
bool header = true;

///////////////////////////////////////////////////////////////////////////////
std::string format_build_date(std::string timestamp)
{
    boost::gregorian::date d = boost::gregorian::from_us_string(timestamp);

    char const* fmt = "%02i-%02i-%04i";

    return boost::str(boost::format(fmt)
                     % d.month().as_number() % d.day() % d.year());
}

void print_results(
    boost::uint64_t cores
  , double walltime
  , std::vector<std::string> const& counter_shortnames
  , boost::shared_ptr<hpx::util::activate_counters> ac 
    )
{
    if (header)
    {
        cout << "# VERSION: " << HPX_GIT_COMMIT << " "
             << format_build_date(__DATE__) << "\n"
             << "#\n";

        // Note that if we change the number of fields above, we have to
        // change the constant that we add when printing out the field # for
        // performance counters below (e.g. the last_index part).
        cout <<
                "## 0:DELAY:Delay [micro-seconds] - Independent Variable\n"
                "## 1:TASKS:Tasks - Independent Variable\n"
                "## 2:OSTHRDS:OS-threads - Independent Variable\n"
                "## 3:WTIME:Total Walltime [seconds]\n"
                ;

        boost::uint64_t const last_index = 3;

        for (boost::uint64_t i = 0; i < counter_shortnames.size(); ++i)
        {
            cout << "## "
                 << (i + 1 + last_index) << ":"
                 << counter_shortnames[i] << ":"
                 << ac->name(i);

            if (!ac->unit_of_measure(i).empty())
                cout << " [" << ac->unit_of_measure(i) << "]";

            cout << "\n";            
        }
    }

    std::string const cores_str = boost::str(boost::format("%lu") % cores);
    std::string const tasks_str = boost::str(boost::format("%lu") % tasks);
    std::string const delay_str = boost::str(boost::format("%lu") % delay);

    cout << ( boost::format("%lu %lu %lu %.14g")
            % delay 
            % tasks 
            % cores
            % walltime);

    if (ac)
    {   
        hpx::util::activate_counters::counter_values_type values
            = ac->evaluate_counters();

        for (boost::uint64_t i = 0; i < counter_shortnames.size(); ++i)
            cout << ( boost::format(" %.14g")
                    % values[i].get().get_value<double>());
    }

    cout << "\n";
}

///////////////////////////////////////////////////////////////////////////////
int invoke_worker_timed(double delay_sec)
{
    volatile int i = 0;
    worker_timed(delay_sec, &i);
    return i;
}

///////////////////////////////////////////////////////////////////////////////
void blocker(
    boost::barrier& entered
  , boost::barrier& started
    )
{
    entered.wait();
    started.wait();
}

///////////////////////////////////////////////////////////////////////////////
void wait_for_tasks(hpx::lcos::local::barrier& finished)
{
    if (get_thread_count(hpx::threads::thread_priority_normal) != 1)
    {
        register_work(boost::bind(&wait_for_tasks, boost::ref(finished)),
            "wait_for_tasks", hpx::threads::pending,
            hpx::threads::thread_priority_low);
    }
    else
        finished.wait();
}

void spawn_workers(boost::uint64_t local_tasks, boost::uint64_t num_thread)
{
    for (boost::uint64_t i = 0; i < local_tasks; ++i)
        register_work(boost::bind(&invoke_worker_timed
                                , double(delay) * 1e-6)
          , "invoke_worker_timed"
          , hpx::threads::pending
          , hpx::threads::thread_priority_normal
          , num_thread
            );
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        boost::uint64_t const os_thread_count = get_os_thread_count();

        if (vm.count("no-header"))
            header = false;

        boost::uint64_t tasks_per_feeder = 0;
        boost::uint64_t total_tasks = 0;
        std::string const scaling = vm["scaling"].as<std::string>();

        if ("strong" == scaling)
        {
            tasks_per_feeder = tasks / os_thread_count;
            total_tasks      = tasks;
        }
        else if ("weak" == scaling)
        {
            tasks_per_feeder = tasks; 
            total_tasks      = tasks * os_thread_count;
        }
        else
            throw std::invalid_argument(
                "invalid scaling type specified (valid options are \"strong\" "
                "or \"weak\")");

        std::vector<std::string> counter_shortnames;
        std::vector<std::string> counters;
        if (vm.count("counter"))
        {
            std::vector<std::string> raw_counters =
                vm["counter"].as<std::vector<std::string> >();

            for (boost::uint64_t i = 0; i < raw_counters.size(); ++i)
            {
                std::vector<std::string> entry;
                boost::algorithm::split(entry, raw_counters[i],
                    boost::algorithm::is_any_of(","),
                    boost::algorithm::token_compress_on);

                HPX_ASSERT(entry.size() == 2);

                counter_shortnames.push_back(entry[0]);
                counters.push_back(entry[1]);
            }
        }

        if (0 == tasks)
            throw std::invalid_argument("count of 0 tasks specified\n");

        boost::shared_ptr<hpx::util::activate_counters> ac;
        if (!counters.empty())
            ac.reset(new hpx::util::activate_counters(counters));

        ///////////////////////////////////////////////////////////////////////
        // Block all other OS threads.
/*
        boost::barrier entered(os_thread_count);
        boost::barrier started(os_thread_count);

        for (boost::uint64_t i = 0; i < (os_thread_count - 1); ++i)
        {
            register_work(boost::bind(
                &blocker, boost::ref(entered), boost::ref(started)));
        }

        entered.wait();
*/

        ///////////////////////////////////////////////////////////////////////
        // Start the clock.
        high_resolution_timer t;

        if (ac)
            ac->reset_counters();

        // This needs to stay here; we may have suspended as recently as the
        // performance counter reset above.
        boost::uint64_t const num_thread = hpx::get_worker_thread_num();

        spawn_workers(total_tasks, num_thread);

//        started.wait();

        // Schedule a low-priority thread; when it is executed, it checks to
        // make sure all the tasks (which are normal priority) have been 
        // executed, and then it
        hpx::lcos::local::barrier finished(2);

        register_work(boost::bind(&wait_for_tasks, boost::ref(finished)),
            "wait_for_tasks", hpx::threads::pending,
            hpx::threads::thread_priority_low);

        finished.wait();

        // Stop the clock
        double time_elapsed = t.elapsed();

        print_results(os_thread_count, time_elapsed, counter_shortnames, ac);
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
        ( "scaling"
        , value<std::string>()->default_value("weak")
        , "type of scaling to benchmark (valid options are \"strong\" or "
          "\"weak\")")

        ( "tasks"
        , value<boost::uint64_t>(&tasks)->default_value(500000)
        , "number of tasks to invoke (when strong-scaling, this is the total "
          "number of tasks invoked; when weak-scaling, it is the number of "
          "tasks per core)")

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

