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

static int pool_threads  = 0;

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
        std::cout << "[async_guided] <std::size_t, bool, const std::string> "
                  << message << " n=" << n << "\n";
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
/*
template <typename ... Args>
int a_function(Args...args) {
    std::cout << "A_function double is " << std::endl;
    return 2;
}
*/
std::string a_function(hpx::future<double> &&df)
{
    std::cout << "A_function double is " << df.get() << std::endl;
    return "The number 2";
}

namespace hpx { namespace threads { namespace executors
{

    struct guided_test_tag {};

    // ------------------------------------------------------------------------
    template <>
    struct HPX_EXPORT pool_numa_hint<guided_test_tag>
    {
        // ------------------------------------------------------------------------
        // specialize the hint operator for params
        int operator()(std::size_t i, bool b, const std::string& msg) const
        {
            std::cout << "<std::size_t, bool, const std::string> hint "
                      << "invoked with : "
                      << i << " " << b << " " << msg << std::endl;
            return 1;
        }

        // ------------------------------------------------------------------------
        // specialize the hint operator for params
        int operator()(int i, double d, const std::string& msg) const
        {
            std::cout << "<int, double, const std::string> hint "
                      << "invoked with : "
                      << i << " " << d << " " << msg << std::endl;
            return 42;
        }

        // ------------------------------------------------------------------------
        // specialize the hint operator for params
        int operator()(double x) const
        {
            std::cout << "double hint invoked with " << x << std::endl;
            return 27;
        }

        // ------------------------------------------------------------------------
        // specialize the hint operator for an arbitrary function/args type
        template <typename...Args>
        int operator ()(Args ...args) const {
            std::cout << "Variadic hint invoked " << std::endl;
            return 56;
        }

    };
}}}

using namespace hpx::threads::executors;

// this is called on an hpx thread after the runtime starts up
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t num_threads = hpx::get_num_worker_threads();
    std::cout << "HPX using threads = " << num_threads << std::endl;

    // ------------------------------------------------------------------------
    // test 1
    // ------------------------------------------------------------------------
    std::cout << std::endl << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Testing async guided exec " << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    // we must specialize the numa callback hint for the function type we are invoking
    using hint_type1 = pool_numa_hint<guided_test_tag>;
    // create an executor using that hint type
    hpx::threads::executors::guided_pool_executor<hint_type1>
            guided_exec(CUSTOM_POOL_NAME);
    // invoke an async function using our numa hint executor
    hpx::future<void> gf1 = hpx::async(guided_exec, &async_guided,
        std::size_t(5), true, std::string("Guided function"));
    gf1.get();

    // ------------------------------------------------------------------------
    // test 2
    // ------------------------------------------------------------------------
    std::cout << std::endl << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Testing async guided exec lambda" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    // specialize the numa hint callback for a lambda type invocation
    // the args of the async lambda must match the args of the hint type
    using hint_type2 = pool_numa_hint<guided_test_tag>;
    // create an executor using the numa hint type
    hpx::threads::executors::guided_pool_executor<hint_type2>
            guided_lambda_exec(CUSTOM_POOL_NAME);
    // invoke a lambda asynchronously and use the numa executor
    hpx::future<double> gf2 = hpx::async(guided_lambda_exec,
        [](int a, double x, const std::string &msg) mutable -> double {
            std::cout << "inside <int, double, string> async lambda "
                      << msg << std::endl;
            // return a double as an example
            return 3.1415;
        },
        5, 2.718, "Guided function 2");
    gf2.get();

    // ------------------------------------------------------------------------
    // static checks for laughs
    // ------------------------------------------------------------------------
    using namespace hpx::traits;
    static_assert(
        has_sync_execute_member<hpx::threads::executors::guided_pool_executor<hint_type2>>::value == std::false_type(),
        "check has_sync_execute_member<Executor>::value");
    static_assert(
        has_async_execute_member<hpx::threads::executors::guided_pool_executor<hint_type2>>::value == std::true_type(),
        "check has_async_execute_member<Executor>::value");
    static_assert(
        has_then_execute_member<hpx::threads::executors::guided_pool_executor<hint_type2>>::value == std::true_type(),
        "has_then_execute_member<executor>::value");
    static_assert(
        has_post_member<hpx::threads::executors::guided_pool_executor<hint_type2>>::value == std::false_type(),
        "has_post_member<executor>::value");

    // ------------------------------------------------------------------------
    // test 3
    // ------------------------------------------------------------------------
    std::cout << std::endl << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Testing async guided exec continuation"         << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    // specialize the numa hint callback for another lambda type invocation
    // the args of the async lambda must match the args of the hint type
    using hint_type3 = pool_numa_hint<guided_test_tag>;
    // create an executor using the numa hint type
    hpx::threads::executors::guided_pool_executor<hint_type3>
            guided_cont_exec(CUSTOM_POOL_NAME);
    // invoke the lambda asynchronously and use the numa executor
    auto new_future = hpx::async([]() -> double { return 2 * 3.1415; })
                          .then(guided_cont_exec, a_function);

    /*
    [](double df)
    {
        double d = df; // .get();
        std::cout << "received a double of value " << d << std::endl;
        return d*2;
    }));
*/
    new_future.get();

    return hpx::finalize();
}

// the normal int main function that is called at startup and runs on an OS thread
// the user must call hpx::init to start the hpx runtime which will execute hpx_main
// on an hpx thread
int main(int argc, char* argv[])
{
    boost::program_options::options_description desc_cmdline("Test options");
    desc_cmdline.add_options()
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
        -> std::unique_ptr<hpx::threads::thread_pool_base>
        {
            std::cout << "User defined scheduler creation callback "
                      << std::endl;
            std::unique_ptr<high_priority_sched> scheduler(
                new high_priority_sched(num_threads,
                    {6, 6, 64}, "shared-priority-scheduler"));

            auto mode = scheduler_mode(scheduler_mode::delay_exit);

            std::unique_ptr<hpx::threads::thread_pool_base> pool(
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
