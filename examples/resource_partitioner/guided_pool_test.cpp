//  Copyright (c) 2017 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
//
#include <hpx/parallel/algorithms/for_loop.hpp>
#include <hpx/parallel/executors.hpp>
//
#include <hpx/runtime/resource/partitioner.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>
#include <hpx/runtime/threads/detail/scheduled_thread_pool_impl.hpp>
#include <hpx/runtime/threads/executors/pool_executor.hpp>
#include <hpx/runtime/threads/executors/guided_pool_executor.hpp>
//
#include <hpx/include/iostreams.hpp>
#include <hpx/include/runtime.hpp>
//
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
//
#include "shared_priority_scheduler.hpp"
#include "system_characteristics.hpp"

static bool use_pools = true;
static bool use_scheduler = false;
static int pool_threads = 1;

#define CUSTOM_POOL_NAME "Custom"

// this is our custom scheduler type
using high_priority_sched = hpx::threads::policies::shared_priority_scheduler<>;
using namespace hpx::threads::policies;

// Force an instantiation of the pool type templated on our custom scheduler
// we need this to ensure that the pool has the generated member functions needed
// by the linker for this pool type
// template class hpx::threads::detail::scheduled_thread_pool<high_priority_sched>;

// dummy function we will call using async
void async_guided(std::size_t n, bool printout, const std::string &message)
{
    if (printout) {
        std::cout << "[async_guided] " << message << " " << n << "\n";
    }
    for (std::size_t i(0); i < n; ++i)
    {
        double f = std::sin(2 * M_PI * i / n);
        if (printout) {
            std::cout << "sin(" << i << ") = " << f << ", ";
        }
    }
    if (printout) {
        std::cout << "\n";
    }
}

// ------------------------------------------------------------------------
// specialize the hint template for our function type
namespace hpx { namespace threads { namespace executors
{
    template <>
    template <typename R, typename...Args>
    struct HPX_EXPORT pool_numa_hint<R(*)(Args...)>
    {
        int operator ()(Args ...args) const {
            return 56;
        }
    };
}}}

// ------------------------------------------------------------------------
// specialize the hint template for lambda args
namespace hpx { namespace threads { namespace executors
{
    template <>
    struct HPX_EXPORT pool_numa_hint<int, double, const std::string &>
    {
        int operator ()(int, double, const std::string &) const {
            return 42;
        }
    };
}}}

using namespace hpx::threads::executors;

// this is called on an hpx thread after the runtime starts up
int hpx_main(boost::program_options::variables_map& vm)
{
    if (vm.count("use-pools"))
        use_pools = true;
    if (vm.count("use-scheduler"))
        use_scheduler = true;
    //
    std::cout << "[hpx_main] starting ..."
              << "use_pools " << use_pools << " "
              << "use_scheduler " << use_scheduler << "\n";

    std::size_t num_threads = hpx::get_num_worker_threads();
    std::cout << "HPX using threads = " << num_threads << std::endl;

    using hint_type1 = pool_numa_hint<decltype(&async_guided)>;

    hpx::threads::executors::guided_pool_executor<hint_type1> guided_exec(CUSTOM_POOL_NAME);
    hpx::future<void> gf1 = hpx::async(guided_exec, &async_guided, 5, true, "Guided function");

    using hint_type2 = pool_numa_hint<int, double, const std::string &>;

    hpx::threads::executors::guided_pool_executor<hint_type2> guided_exec2(CUSTOM_POOL_NAME);
    hpx::future<void> gf2 = hpx::async(guided_exec2,
        [](int a, double x, const std::string &msg) {
            std::cout << "inside async lambda " << msg << std::endl;
        },
        5, 3.14, "Guided function 2");

//    hpx::future<void> gf2 = gf1.then(
//        guided_exec, [](hpx::future<void>&& f) { async_guided(5, true, "guided continuation"); });

    gf1.get();

    return hpx::finalize();
}

// the normal int main function that is called at startup and runs on an OS thread
// the user must call hpx::init to start the hpx runtime which will execute hpx_main
// on an hpx thread
int main(int argc, char* argv[])
{
    boost::program_options::options_description desc_cmdline("Test options");
    desc_cmdline.add_options()
        ( "use-pools,u", "Enable advanced HPX thread pools and executors")
        ( "use-scheduler,s", "Enable custom priority scheduler")
        ( "pool-threads,m",
          boost::program_options::value<int>()->default_value(1),
          "Number of threads to assign to custom pool")
    ;

    // HPX uses a boost program options variable map, but we need it before
    // hpx-main, so we will create another one here and throw it away after use
    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::command_line_parser(argc, argv)
            .allow_unregistered()
            .options(desc_cmdline)
            .run(),
        vm);

    if (vm.count("use-pools"))
    {
        use_pools = true;
    }
    if (vm.count("use-scheduler"))
    {
        use_scheduler = true;
    }

    pool_threads = vm["pool-threads"].as<int>();

    // Create the resource partitioner
    hpx::resource::partitioner rp(desc_cmdline, argc, argv);

    //    auto &topo = rp.get_topology();
    std::cout << "[main] obtained reference to the resource_partitioner\n";

    // create a thread pool and supply a lambda that returns a new pool with
    // a user supplied scheduler attached
    rp.create_thread_pool(CUSTOM_POOL_NAME,
        [](hpx::threads::policies::callback_notifier& notifier,
            std::size_t num_threads, std::size_t thread_offset,
            std::size_t pool_index, std::string const& pool_name)
        -> std::unique_ptr<hpx::threads::detail::thread_pool_base>
        {
            std::cout << "User defined scheduler creation callback "
                      << std::endl;
            std::unique_ptr<high_priority_sched> scheduler(
                new high_priority_sched(num_threads, 1, false, false,
                    "shared-priority-scheduler"));

            auto mode = scheduler_mode(scheduler_mode::delay_exit);

            std::unique_ptr<hpx::threads::detail::thread_pool_base> pool(
                new hpx::threads::detail::scheduled_thread_pool<
                        high_priority_sched
                    >(std::move(scheduler), notifier,
                        pool_index, pool_name, mode, thread_offset));
            return pool;
        });

    // rp.add_resource(rp.numa_domains()[0].cores()[0].pus(), CUSTOM_POOL_NAME);
    // add N cores to Custom pool
    int count = 0;
    for (const hpx::resource::numa_domain& d : rp.numa_domains())
    {
        for (const hpx::resource::core& c : d.cores())
        {
            for (const hpx::resource::pu& p : c.pus())
            {
                if (count < pool_threads)
                {
                    std::cout << "Added pu " << count++ << " to " CUSTOM_POOL_NAME " pool\n";
                    rp.add_resource(p, CUSTOM_POOL_NAME);
                }
            }
        }
    }

    std::cout << "[main] resources added to thread_pools \n";

    return hpx::init();
}
