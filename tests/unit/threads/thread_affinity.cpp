///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2007-2012 Hartmut Kaiser
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/actions.hpp>
#include <hpx/include/components.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>
#include <list>
#include <set>

#include <boost/ref.hpp>
#include <boost/foreach.hpp>

#include <hwloc.h>

std::size_t thread_affinity_worker(std::size_t desired)
{
    // Returns the OS-thread number of the worker that is running this
    // PX-thread.
    bool numa_sensitive = false;
    std::size_t current = hpx::get_worker_thread_num(&numa_sensitive);

    if (current == desired)
    {
#if HPX_HAVE_HWLOC
        hwloc_topology_t topo;
        hwloc_topology_init(&topo);
        hwloc_topology_load(topo);

        // retrieve the current affinity mask
        hwloc_cpuset_t cpuset = hwloc_bitmap_alloc();
        if (0 == hwloc_get_cpubind(topo, cpuset, HWLOC_CPUBIND_THREAD)) {
            // sadly get_cpubind is not implemented for Windows based systems
            char buf[128];
            hwloc_bitmap_snprintf(buf, sizeof(buf), cpuset);
            std::size_t current_mask = strtoul(buf, 0, 16);     // parse the value back

            // extract the desired affinity mask
            hpx::threads::topology const& t = hpx::get_runtime().get_topology();
            std::size_t desired_mask = t.get_thread_affinity_mask(current, numa_sensitive);
            HPX_TEST_EQ(desired_mask, current_mask);
        }

        hwloc_bitmap_free(cpuset);
        hwloc_topology_destroy(topo);
#endif
        return desired;
    }

    // This PX-thread has been run by the wrong OS-thread, make the foreman
    // try again by rescheduling it.
    return std::size_t(-1);
}

HPX_PLAIN_ACTION(thread_affinity_worker, thread_affinity_worker_action)

void thread_affinity_foreman()
{
    // Get the number of worker OS-threads in use by this locality.
    std::size_t const os_threads = hpx::get_os_thread_count();

    // Find the global name of the current locality.
    hpx::naming::id_type const here = hpx::find_here();

    // Populate a set with the OS-thread numbers of all OS-threads on this
    // locality. When the hello world message has been printed on a particular
    // OS-thread, we will remove it from the set.
    std::set<std::size_t> attendance;
    for (std::size_t os_thread = 0; os_thread < os_threads; ++os_thread)
        attendance.insert(os_thread);

    // As long as there are still elements in the set, we must keep scheduling
    // PX-threads. Because HPX features work-stealing task schedulers, we have
    // no way of enforcing which worker OS-thread will actually execute
    // each PX-thread.
    while (!attendance.empty())
    {
        // Each iteration, we create a task for each element in the set of
        // OS-threads that have not said "Hello world". Each of these tasks
        // is encapsulated in a future.
        std::vector<hpx::lcos::future<std::size_t> > futures;
        futures.reserve(attendance.size());

        BOOST_FOREACH(std::size_t worker, attendance)
        {
            // Asynchronously start a new task. The task is encapsulated in a
            // future, which we can query to determine if the task has
            // completed.
            typedef thread_affinity_worker_action action_type;
            futures.push_back(hpx::async<action_type>(here, worker));
        }

        // Wait for all of the futures to finish. The callback version of the
        // hpx::lcos::wait function takes two arguments: a vector of futures,
        // and a binary callback.  The callback takes two arguments; the first
        // is the index of the future in the vector, and the second is the
        // return value of the future. hpx::lcos::wait doesn't return until
        // all the futures in the vector have returned.
        hpx::lcos::wait(futures,
            [&](std::size_t, std::size_t t) {
                if (std::size_t(-1) != t)
                    attendance.erase(t);
            });
    }
}

HPX_PLAIN_ACTION(thread_affinity_foreman, thread_affinity_foreman_action)

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& /*vm*/)
{
    {
        // Get a list of all available localities.
        std::vector<hpx::naming::id_type> localities =
            hpx::find_all_localities();

        // Reserve storage space for futures, one for each locality.
        std::vector<hpx::lcos::future<void> > futures;
        futures.reserve(localities.size());

        BOOST_FOREACH(hpx::naming::id_type const& node, localities)
        {
            // Asynchronously start a new task. The task is encapsulated in a
            // future, which we can query to determine if the task has
            // completed.
            typedef thread_affinity_foreman_action action_type;
            futures.push_back(hpx::async<action_type>(node));
        }

        // The non-callback version of hpx::lcos::wait takes a single parameter,
        // a future of vectors to wait on. hpx::lcos::wait only returns when
        // all of the futures have finished.
        hpx::lcos::wait(futures);
    }

    // Initiate shutdown of the runtime system.
    hpx::finalize();
    return hpx::util::report_errors();
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    // Configure application-specific options.
    boost::program_options::options_description
       desc_commandline("usage: " HPX_APPLICATION_STRING " [options]");

    // we force this test to use several (4) threads
    using namespace boost::assign;
    std::vector<std::string> cfg;
    cfg += "hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::thread::hardware_concurrency());

    // Initialize and run HPX.
    return hpx::init(desc_commandline, argc, argv, cfg);
}

