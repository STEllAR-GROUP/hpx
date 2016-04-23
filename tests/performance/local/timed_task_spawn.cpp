//  Copyright (c) 2011-2012 Bryce Adelstein-Lelbach
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c)      2013 Thomas Heller
//  Copyright (c)      2013 Patricia Grubel
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// FIXME: Calling the tasks "workers" overloads the term worker-thread (which
// refers to OS-threads).

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/bind.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/cstdint.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/format.hpp>
#include <boost/math/common_factor.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/mutex.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "worker_timed.hpp"

char const* benchmark_name = "Homogeneous Timed Task Spawn - HPX";

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
boost::uint64_t suspended_tasks = 0;
boost::uint64_t delay = 0;
bool header = true;
bool csv_header = false;
std::string scaling("weak");
std::string distribution("static-balanced");

boost::uint64_t suspend_step = 0;
boost::uint64_t no_suspend_step = 1;

///////////////////////////////////////////////////////////////////////////////
std::string format_build_date(std::string timestamp)
{
    boost::gregorian::date d = boost::gregorian::from_us_string(timestamp);

    char const* fmt = "%02i-%02i-%04i";

    return boost::str(boost::format(fmt)
                     % d.month().as_number() % d.day() % d.year());
}

///////////////////////////////////////////////////////////////////////////////
void print_results(
    boost::uint64_t cores
  , double walltime
  , double warmup_estimate
  , std::vector<std::string> const& counter_shortnames
  , std::shared_ptr<hpx::util::activate_counters> ac
    )
{
    std::vector<hpx::performance_counters::counter_value> counter_values;

    if (ac)
        counter_values = ac->evaluate_counters_sync();

    if (csv_header)
    {
        header = false;
        cout << "Delay,Tasks,STasks,OS_Threads,Execution_Time_sec,Warmup_sec";

        for (boost::uint64_t i = 0; i < counter_shortnames.size(); ++i)
        {
            cout << "," << counter_shortnames[i];
        }
        cout << "\n";


    }

    if (header)
    {
        cout << "# BENCHMARK: " << benchmark_name
                 << " (" << scaling << " scaling, "
                 << distribution << " distribution)\n";

        cout << "# VERSION: " << HPX_HAVE_GIT_COMMIT << " "
                 << format_build_date(__DATE__) << "\n"
             << "#\n";

        // Note that if we change the number of fields above, we have to
        // change the constant that we add when printing out the field # for
        // performance counters below (e.g. the last_index part).
        cout <<
                "## 0:DELAY:Delay [micro-seconds] - Independent Variable\n"
                "## 1:TASKS:# of Tasks - Independent Variable\n"
                "## 2:STASKS:# of Tasks to Suspend - Independent Variable\n"
                "## 3:OSTHRDS:OS-threads - Independent Variable\n"
                "## 4:WTIME:Total Walltime [seconds]\n"
                "## 5:WARMUP:Total Walltime [seconds]\n"
                ;

        boost::uint64_t const last_index = 5;

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

    cout << ( boost::format("%lu, %lu, %lu, %lu, %.14g, %.14g")
            % delay
            % tasks
            % suspended_tasks
            % cores
            % walltime
            % warmup_estimate
            );

    if (ac)
    {
        for (boost::uint64_t i = 0; i < counter_shortnames.size(); ++i)
            cout << ( boost::format(", %.14g")
                    % counter_values[i].get_value<double>());
    }

    cout << "\n";
}

///////////////////////////////////////////////////////////////////////////////
void wait_for_tasks(
    hpx::lcos::local::barrier& finished
  , boost::uint64_t suspended_tasks
    )
{
    boost::uint64_t const pending_count =
        get_thread_count(hpx::threads::thread_priority_normal
                       , hpx::threads::pending);

    if (pending_count == 0)
    {
        boost::uint64_t const all_count =
            get_thread_count(hpx::threads::thread_priority_normal);

        if (all_count != suspended_tasks + 1)
        {
            register_work(hpx::util::bind(&wait_for_tasks
                                    , boost::ref(finished)
                                    , suspended_tasks)
                , "wait_for_tasks", hpx::threads::pending
                , hpx::threads::thread_priority_low);
            return;
        }
    }

    finished.wait();
}

///////////////////////////////////////////////////////////////////////////////
hpx::threads::thread_state_enum invoke_worker_timed_no_suspension(
    hpx::threads::thread_state_ex_enum ex = hpx::threads::wait_signaled
    )
{
    worker_timed(delay * 1000);
    return hpx::threads::terminated;
}

hpx::threads::thread_state_enum invoke_worker_timed_suspension(
    hpx::threads::thread_state_ex_enum ex = hpx::threads::wait_signaled
    )
{
    worker_timed(delay * 1000);

    hpx::error_code ec(hpx::lightweight);
    hpx::this_thread::suspend(hpx::threads::suspended, "suspend", ec);

    return hpx::threads::terminated;
}

///////////////////////////////////////////////////////////////////////////////
typedef void (*stage_worker_function)(boost::uint64_t, bool);

void stage_worker_static_balanced_stackbased(
    boost::uint64_t target_thread
  , bool suspend
    )
{
    if (suspend)
        hpx::threads::register_thread_plain(
            &invoke_worker_timed_suspension
          , "invoke_worker_timed_suspension"
          , hpx::threads::pending
          , false
          , hpx::threads::thread_priority_normal
          , target_thread
            );
    else
        hpx::threads::register_thread_plain(
            &invoke_worker_timed_no_suspension
          , "invoke_worker_timed_no_suspension"
          , hpx::threads::pending
          , false
          , hpx::threads::thread_priority_normal
          , target_thread
            );
}

void stage_worker_static_imbalanced(
    boost::uint64_t target_thread
  , bool suspend
    )
{
    if (suspend)
        hpx::threads::register_thread_plain(
            &invoke_worker_timed_suspension
          , "invoke_worker_timed_suspension"
          , hpx::threads::pending
          , false
          , hpx::threads::thread_priority_normal
          , 0
            );
    else
        hpx::threads::register_thread_plain(
            &invoke_worker_timed_no_suspension
          , "invoke_worker_timed_no_suspension"
          , hpx::threads::pending
          , false
          , hpx::threads::thread_priority_normal
          , 0
            );
}

void stage_worker_round_robin(
    boost::uint64_t target_thread
  , bool suspend
    )
{
    if (suspend)
        hpx::threads::register_thread_plain(
            &invoke_worker_timed_suspension
          , "invoke_worker_timed_suspension"
          , hpx::threads::pending
          , false
            );
    else
        hpx::threads::register_thread_plain(
            &invoke_worker_timed_no_suspension
          , "invoke_worker_timed_no_suspension"
          , hpx::threads::pending
          , false
            );
}

void stage_workers(
    boost::uint64_t target_thread
  , boost::uint64_t local_tasks
  , stage_worker_function stage_worker
    )
{
    boost::uint64_t num_thread = hpx::get_worker_thread_num();

    if (num_thread != target_thread)
    {
        register_work(hpx::util::bind(&stage_workers
                                , target_thread
                                , local_tasks
                                , stage_worker)
            , "stage_workers"
            , hpx::threads::pending
            , hpx::threads::thread_priority_normal
            , target_thread
              );
        return;
    }

    for (boost::uint64_t i = 0; i < local_tasks;)
    {
        for (boost::uint64_t j = 0; j < suspend_step; ++j)
        {
            stage_worker(target_thread, true);
            ++i;
        }
        for (boost::uint64_t j = 0; j < no_suspend_step; ++j)
        {
            stage_worker(target_thread, false);
            ++i;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(
    variables_map& vm
    )
{
    {
        if (vm.count("no-header"))
            header = false;

        if (vm.count("csv-header"))
            csv_header = true;

        if (0 == tasks)
            throw std::invalid_argument("count of 0 tasks specified\n");

        if (suspended_tasks > tasks)
            throw std::invalid_argument(
                "suspended tasks must be smaller than tasks\n");

        boost::uint64_t const os_thread_count = get_os_thread_count();

        ///////////////////////////////////////////////////////////////////////
        stage_worker_function stage_worker;

        if ("static-balanced-stackbased" == distribution)
            stage_worker = &stage_worker_static_balanced_stackbased;
        else if ("static-imbalanced" == distribution)
            stage_worker = &stage_worker_static_imbalanced;
        else if ("round-robin" == distribution)
            stage_worker = &stage_worker_round_robin;
        else
            throw std::invalid_argument(
                "invalid distribution type specified (valid options are "
                "\"static-balanced\", \"static-imbalanced\" or \"round-robin\")"
                );

        ///////////////////////////////////////////////////////////////////////
        boost::uint64_t tasks_per_feeder = 0;
        //boost::uint64_t total_tasks = 0;
        boost::uint64_t suspended_tasks_per_feeder = 0;
        boost::uint64_t total_suspended_tasks = 0;

        if ("strong" == scaling)
        {
            if (tasks % os_thread_count)
                throw std::invalid_argument(
                    "tasks must be cleanly divisable by OS-thread count\n");

            if (suspended_tasks % os_thread_count)
                throw std::invalid_argument(
                    "suspended tasks must be cleanly divisable by OS-thread "
                    "count\n");

            tasks_per_feeder = tasks / os_thread_count;
            //total_tasks      = tasks;
            suspended_tasks_per_feeder = suspended_tasks / os_thread_count;
            total_suspended_tasks      = suspended_tasks;
        }
        else if ("weak" == scaling)
        {
            tasks_per_feeder = tasks;
            //total_tasks      = tasks * os_thread_count;
            suspended_tasks_per_feeder = suspended_tasks;
            total_suspended_tasks      = suspended_tasks * os_thread_count;
        }
        else
            throw std::invalid_argument(
                "invalid scaling type specified (valid options are \"strong\" "
                "or \"weak\")");

        ///////////////////////////////////////////////////////////////////////
        if (suspended_tasks != 0)
        {
            boost::uint64_t gcd = boost::math::gcd(tasks_per_feeder
                                                 , suspended_tasks_per_feeder);

            suspend_step = suspended_tasks_per_feeder / gcd;
            // We check earlier to make sure that there are never more
            // suspended tasks than tasks requested.
            no_suspend_step = (tasks_per_feeder / gcd) - suspend_step;
        }

        ///////////////////////////////////////////////////////////////////////
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

        std::shared_ptr<hpx::util::activate_counters> ac;
        if (!counters.empty())
            ac.reset(new hpx::util::activate_counters(counters));

        ///////////////////////////////////////////////////////////////////////
        // Start the clock.
        high_resolution_timer t;

        if (ac)
            ac->reset_counters();

        // This needs to stay here; we may have suspended as recently as the
        // performance counter reset (which is called just before the staging
        // function).
        boost::uint64_t const num_thread = hpx::get_worker_thread_num();

        for (boost::uint64_t i = 0; i < os_thread_count; ++i)
        {
            if (num_thread == i) continue;

            register_work(hpx::util::bind(&stage_workers
                                    , i
                                    , tasks_per_feeder
                                    , stage_worker
                                      )
                , "stage_workers"
                , hpx::threads::pending
                , hpx::threads::thread_priority_normal
                , i
                  );
        }

        stage_workers(num_thread, tasks_per_feeder, stage_worker);

        double warmup_estimate = t.elapsed();

        // Schedule a low-priority thread; when it is executed, it checks to
        // make sure all the tasks (which are normal priority) have been
        // executed, and then it
        hpx::lcos::local::barrier finished(2);

        register_work(hpx::util::bind(&wait_for_tasks
                                , boost::ref(finished)
                                , total_suspended_tasks
                                 )
            , "wait_for_tasks", hpx::threads::pending
            , hpx::threads::thread_priority_low);

        finished.wait();

        // Stop the clock
        double time_elapsed = t.elapsed();

        print_results(os_thread_count, time_elapsed, warmup_estimate
                    , counter_shortnames, ac);
    }

    if (suspended_tasks != 0)
        // Force termination of all suspended tasks.
        hpx::get_runtime().get_thread_manager().abort_all_suspended_threads();

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
        , value<std::string>(&scaling)->default_value("weak")
        , "type of scaling to benchmark (valid options are \"strong\" or "
          "\"weak\")")

        ( "distribution"
        , value<std::string>(&distribution)->default_value("static-balanced")
        , "type of distribution to perform (valid options are "
          "\"static-balanced\", \"static-imbalanced\" or \"round-robin\")")

        ( "tasks"
        , value<boost::uint64_t>(&tasks)->default_value(500000)
        , "number of tasks to invoke (when strong-scaling, this is the total "
          "number of tasks invoked; when weak-scaling, it is the number of "
          "tasks per core)")

        ( "suspended-tasks"
        , value<boost::uint64_t>(&suspended_tasks)->default_value(0)
        , "number of tasks to suspend (when strong-scaling, this is the total "
          "number of tasks suspended; when weak-scaling, it is the number of "
          "suspended per core)")

        ( "delay"
        , value<boost::uint64_t>(&delay)->default_value(5)
        , "duration of delay in microseconds")

        ( "counter"
        , value<std::vector<std::string> >()->composing()
        , "activate and report the specified performance counter")

        ( "no-header"
        , "do not print out the header")

        ( "csv-header"
        , "print out csv header")
        ;

    // Initialize and run HPX.
    return init(cmdline, argc, argv);
}
