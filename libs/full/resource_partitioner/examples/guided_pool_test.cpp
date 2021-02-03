//  Copyright (c) 2017-2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
//
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/resource_partitioner.hpp>
#include <hpx/include/runtime.hpp>
#include <hpx/include/threads.hpp>
#include <hpx/iostream.hpp>
//
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
//
#include "system_characteristics.hpp"

static int pool_threads = 0;

#define CUSTOM_POOL_NAME "Custom"

// this is our custom scheduler type
using high_priority_sched =
    hpx::threads::policies::shared_priority_queue_scheduler<>;
using hpx::threads::policies::scheduler_mode;

// Force an instantiation of the pool type templated on our custom scheduler
// we need this to ensure that the pool has the generated member functions needed
// by the linker for this pool type
// template class hpx::threads::detail::scheduled_thread_pool<high_priority_sched>;

// dummy function we will call using async
void async_guided(std::size_t n, bool printout, std::string const& message)
{
    if (printout)
    {
        std::cout << "[async_guided] <std::size_t, bool, const std::string> "
                  << message << " n=" << n << "\n";
    }
    for (std::size_t i(0); i < n; ++i)
    {
        double f = std::sin(2 * M_PI * i / n);
        if (printout)
        {
            std::cout << "sin(" << i << ") = " << f << ", ";
        }
    }
    if (printout)
    {
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
std::string a_function(hpx::future<double>&& df)
{
    std::cout << "A_function double is " << df.get() << std::endl;
    return "The number 2";
}

namespace hpx { namespace parallel { namespace execution {

    struct guided_test_tag
    {
    };

    // ------------------------------------------------------------------------
    template <>
    struct pool_numa_hint<guided_test_tag>
    {
        // ------------------------------------------------------------------------
        // specialize the hint operator for params
        int operator()(std::size_t i, bool b, std::string const& msg) const
        {
            std::cout << "<std::size_t, bool, const std::string> hint "
                      << "invoked with : " << i << " " << b << " " << msg
                      << std::endl;
            return 1;
        }

        // ------------------------------------------------------------------------
        // specialize the hint operator for params
        int operator()(int i, double d, std::string const& msg) const
        {
            std::cout << "<int, double, const std::string> hint "
                      << "invoked with : " << i << " " << d << " " << msg
                      << std::endl;
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
        template <typename... Args>
        int operator()(Args...) const
        {
            std::cout << "Variadic hint invoked " << std::endl;
            return 56;
        }
    };
}}}    // namespace hpx::parallel::execution

using namespace hpx::parallel::execution;

// this is called on an hpx thread after the runtime starts up
int hpx_main()
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
    hpx::parallel::execution::guided_pool_executor<hint_type1> guided_exec(
        &hpx::resource::get_thread_pool(CUSTOM_POOL_NAME));
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
    hpx::parallel::execution::guided_pool_executor<hint_type2>
        guided_lambda_exec(&hpx::resource::get_thread_pool(CUSTOM_POOL_NAME));
    // invoke a lambda asynchronously and use the numa executor
    hpx::future<double> gf2 = hpx::async(
        guided_lambda_exec,
        [](int, double, std::string const& msg) mutable -> double {
            std::cout << "inside <int, double, string> async lambda " << msg
                      << std::endl;
            // return a double as an example
            return 3.1415;
        },
        5, 2.718, "Guided function 2");
    gf2.get();

    // ------------------------------------------------------------------------
    // static checks for laughs
    // ------------------------------------------------------------------------
    static_assert(
        hpx::traits::has_sync_execute_member<hpx::parallel::execution::
                guided_pool_executor<hint_type2>>::value == std::false_type(),
        "check has_sync_execute_member<Executor>::value");
    static_assert(
        hpx::traits::has_async_execute_member<hpx::parallel::execution::
                guided_pool_executor<hint_type2>>::value == std::true_type(),
        "check has_async_execute_member<Executor>::value");
    static_assert(
        hpx::traits::has_then_execute_member<hpx::parallel::execution::
                guided_pool_executor<hint_type2>>::value == std::true_type(),
        "has_then_execute_member<executor>::value");
    static_assert(
        hpx::traits::has_post_member<hpx::parallel::execution::
                guided_pool_executor<hint_type2>>::value == std::false_type(),
        "has_post_member<executor>::value");

    // ------------------------------------------------------------------------
    // test 3
    // ------------------------------------------------------------------------
    std::cout << std::endl << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Testing async guided exec continuation" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    // specialize the numa hint callback for another lambda type invocation
    // the args of the async lambda must match the args of the hint type
    using hint_type3 = pool_numa_hint<guided_test_tag>;
    // create an executor using the numa hint type
    hpx::parallel::execution::guided_pool_executor<hint_type3> guided_cont_exec(
        &hpx::resource::get_thread_pool(CUSTOM_POOL_NAME));
    // invoke the lambda asynchronously and use the numa executor
    auto new_future = hpx::async([]() -> double {
        return 2 * 3.1415;
    }).then(guided_cont_exec, a_function);

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

void init_resource_partitioner_handler(
    hpx::resource::partitioner& rp, hpx::program_options::variables_map const&)
{
    // create a thread pool and supply a lambda that returns a new pool with
    // a user supplied scheduler attached
    rp.create_thread_pool(CUSTOM_POOL_NAME,
        [](hpx::threads::thread_pool_init_parameters init,
            hpx::threads::policies::thread_queue_init_parameters
                thread_queue_init)
            -> std::unique_ptr<hpx::threads::thread_pool_base> {
            std::cout << "User defined scheduler creation callback "
                      << std::endl;
            high_priority_sched::init_parameter_type scheduler_init(
                init.num_threads_, {1, 1, 64}, init.affinity_data_,
                thread_queue_init, "shared-priority-scheduler");
            std::unique_ptr<high_priority_sched> scheduler(
                new high_priority_sched(scheduler_init));

            init.mode_ = scheduler_mode(scheduler_mode::delay_exit);

            std::unique_ptr<hpx::threads::thread_pool_base> pool(
                new hpx::threads::detail::scheduled_thread_pool<
                    high_priority_sched>(std::move(scheduler), init));
            return pool;
        });

    // rp.add_resource(rp.numa_domains()[0].cores()[0].pus(), CUSTOM_POOL_NAME);
    // add N cores to Custom pool
    int count = 0;
    for (hpx::resource::numa_domain const& d : rp.numa_domains())
    {
        for (hpx::resource::core const& c : d.cores())
        {
            for (hpx::resource::pu const& p : c.pus())
            {
                if (count < pool_threads)
                {
                    std::cout << "Added pu " << count++
                              << " to " CUSTOM_POOL_NAME " pool\n";
                    rp.add_resource(p, CUSTOM_POOL_NAME);
                }
            }
        }
    }

    std::cout << "[rp_callback] resources added to thread_pools \n";
}

// the normal int main function that is called at startup and runs on an OS thread
// the user must call hpx::init to start the hpx runtime which will execute hpx_main
// on an hpx thread
int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_cmdline("Test options");
    desc_cmdline.add_options()("pool-threads,m",
        hpx::program_options::value<int>()->default_value(1),
        "Number of threads to assign to custom pool");

    // HPX uses a boost program options variable map, but we need it before
    // hpx-main, so we will create another one here and throw it away after use
    hpx::program_options::variables_map vm;
    hpx::program_options::store(
        hpx::program_options::command_line_parser(argc, argv)
            .allow_unregistered()
            .options(desc_cmdline)
            .run(),
        vm);

    pool_threads = vm["pool-threads"].as<int>();

    // Setup the init parameters
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_cmdline;

    // Set the callback to init the thread_pools
    init_args.rp_callback = &init_resource_partitioner_handler;

    return hpx::init(argc, argv, init_args);
}
