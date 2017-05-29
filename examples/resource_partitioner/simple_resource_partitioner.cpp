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
#include <hpx/include/iostreams.hpp>
#include <hpx/include/runtime.hpp>
//
#include <cmath>
//
#include "system_characteristics.h"


void do_stuff(std::size_t n){
    std::cout << "[do stuff] " << n << "\n";
    for (std::size_t  i(0); i<n; ++i){
        std::cout << "sin(" << i << ") = " << sin(2*M_PI*i/n) << ", ";
    }
    std::cout << "\n";
}


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
    hpx::threads::executors::customized_pool_executor my_exec_single("single_thread");
    std::cout << "\n\n[hpx_main] got customized executor " << "\n";

/*    // get executors
    hpx::threads::executors::customized_pool_executor my_exec1("first_core");
    std::cout << "\n\n[hpx_main] got customized executor " << "\n";

    // get executors
    hpx::threads::executors::customized_pool_executor my_exec2("last_core");
    std::cout << "\n\n[hpx_main] got customized executor " << "\n"; */

    // use these executors to schedule work
    hpx::future<void> future_1 = hpx::async(my_exec_single, &do_stuff, 32);

    auto future_2 = future_1.then(my_exec_single, [](hpx::future<void> &&f) {
        do_stuff(64);
    });

    future_2.get();

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::cout << "[main] " << "Starting program... \n";

    auto &rp = hpx::get_resource_partitioner();
    auto &topo = rp.get_topology();
    std::cout << "[main] " << "obtained reference to the resource_partitioner\n";

//    rp.create_thread_pool("first_core");
//    rp.create_thread_pool("last_core", hpx::resource::abp_priority);
    rp.create_thread_pool("single_thread");
    std::cout << "[main] " << "thread_pools created \n";

//    rp.add_resource(rp.get_numa_domains().front().cores_.front().pus_, "first_core");
//    rp.add_resource(rp.get_numa_domains().back().cores_.back().pus_,  "last_core");
    for (const hpx::resource::numa_domain &d : rp.get_numa_domains()) {
        for (const hpx::resource::core &c : d.cores_) {
            for (const hpx::resource::pu &p : c.pus_) {
                if (p.id_ == rp.get_topology().get_number_of_pus()/2) {
                    rp.add_resource(p, "single_thread");
                }
/*
                std::cout << "[PU] number : " << p << " is on ... \n"
                          << "socket    : " << topo.get_socket_number(p) << "\n"
                          << "numa-node : " << topo.get_numa_node_number(p) << "\n"
                          << "core      : " << topo.get_core_number(p) << hpx::flush << "\n";
*/
            }
        }
    }
    std::cout << "[main] " << "resources added to thread_pools \n";

    std::cout << "[main] " << "Calling hpx::init... \n";
    return hpx::init(argc, argv);
}
