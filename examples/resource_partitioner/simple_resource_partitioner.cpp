//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
//
#include <hpx/runtime/resource_partitioner.hpp>
#include <hpx/runtime/threads/cpu_mask.hpp>
#include <hpx/runtime/threads/executors/customized_pool_executors.hpp>
//
// we should not need this
#include <hpx/runtime/threads/detail/thread_pool.hpp>
#include <hpx/runtime/threads/detail/thread_pool_impl.hpp>
#include <hpx/runtime/threads/policies/scheduler_mode.hpp>
//
#include <hpx/include/iostreams.hpp>
#include <hpx/include/runtime.hpp>
//
#include <cmath>
//
#include "shared_priority_scheduler.hpp"
#include "system_characteristics.h"

namespace resource {
namespace pools
{
    enum ids {
        DEFAULT = 0,
        MPI = 1,
        GPU = 2,
        MATRIX = 3,
};
}}

// this is our custom scheduler type
using high_priority_sched = hpx::threads::policies::shared_priority_scheduler<>;

// Force an instantiation of the pool type templated on our custom scheduler
// we need this to ensure that the pool has the generated member functions needed
// by the linker for this pool type
template class hpx::threads::detail::thread_pool_impl<high_priority_sched>;

// dummy function we will call using async
void do_stuff(std::size_t n){
    std::cout << "[do stuff] " << n << "\n";
    for (std::size_t  i(0); i<n; ++i){
        std::cout << "sin(" << i << ") = " << sin(2*M_PI*i/n) << ", ";
    }
    std::cout << "\n";
}

// this is called on an hpx thread after the runtime starts up
int hpx_main(int argc, char* argv[])
{
    std::cout << "[hpx_main] starting ..." << "\n";

    // get a pointer to the resource_partitioner instance
    hpx::resource::resource_partitioner& rpart = hpx::get_resource_partitioner();

    // print partition characteristics
    std::cout << "\n\n[hpx_main] print resource_partitioner characteristics : " << "\n";
    rpart.print_init_pool_data();

    // print partition characteristics
    std::cout << "\n\n[hpx_main] print thread-manager characteristics : " << "\n";
    hpx::threads::get_thread_manager().print_pools();

    // print system characteristics
    print_system_characteristics();

    // get executors
    hpx::threads::executors::customized_pool_executor test_exec("mpi");
    std::cout << "\n\n[hpx_main] got customized executor " << "\n";
    // use these executors to schedule work
    hpx::future<void> future_1 = hpx::async(test_exec, &do_stuff, 32);

    auto future_2 = future_1.then(test_exec, [](hpx::future<void> &&f) {
        do_stuff(64);
    });

    future_2.get();
    return hpx::finalize();
}

using namespace boost::program_options;

// the normal int main function that is called at startup and runs on an OS thread
// the user must call hpx::init to start the hpx runtime which will execute hpx_main
// on an hpx thread
int main(int argc, char* argv[])
{
    options_description test_options("Test options");
    test_options.add_options()
        ("use-pools,u", "Enable advanced HPX thread pools and executors")
    ;

    options_description desc_cmdline;
    desc_cmdline.add(test_options);

    auto &rp = hpx::get_resource_partitioner(argc, argv);
    auto &topo = rp.get_topology();
    std::cout << "[main] " << "obtained reference to the resource_partitioner\n";

    // create a thread pool and supply a lambda that returns a new pool with
    // the a user supplied scheduler attached
    rp.create_thread_pool("default", [](
        hpx::threads::policies::callback_notifier &notifier,
        std::size_t index, char const* name,
        hpx::threads::policies::scheduler_mode m)
    {
        std::cout << "User defined scheduler creation callback " << std::endl;
        high_priority_sched::init_parameter_type init(
                    hpx::get_resource_partitioner().get_num_threads(name),
                    "shared-priority-scheduler");
        high_priority_sched* scheduler = new high_priority_sched(init);
        return new hpx::threads::detail::thread_pool_impl<high_priority_sched>(
            scheduler, notifier, index, name, m);
    });

    // Create a thread pool using the default scheduler provided by HPX
    rp.create_thread_pool("mpi", hpx::resource::scheduling_policy::local_priority_fifo);
    std::cout << "[main] " << "thread_pools created \n";

    rp.add_resource(rp.numa_domains()[0].cores_[0].pus_, "mpi");
    std::cout << "[main] " << "resources added to thread_pools \n";

/*
    for (const hpx::resource::numa_domain &d : rp.get_numa_domains()) {
        for (const hpx::resource::core &c : d.cores_) {
            for (const hpx::resource::pu &p : c.pus_) {
                if (p.id_ == rp.get_topology().get_number_of_pus()/2) {
                    rp.add_resource(p, "single_thread");
                }

                std::cout << "[PU] number : " << p << " is on ... \n"
                          << "socket    : " << topo.get_socket_number(p) << "\n"
                          << "numa-node : " << topo.get_numa_node_number(p) << "\n"
                          << "core      : " << topo.get_core_number(p) << hpx::flush << "\n";

            }
        }
    }
*/

    std::cout << "[main] " << "Calling hpx::init... \n";
    return hpx::init(desc_cmdline, argc, argv);
}
