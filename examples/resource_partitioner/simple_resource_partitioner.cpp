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


void do_stuff(){
    std::cout << "[do stuff] \n";
    for(size_t i(18); i < 42; i++){
        std::cout << "sin(" << i << ") = " << sin(i) << ", ";
    }
    std::cout << "\n";
}


int hpx_main(int argc, char* argv[])
{
    hpx::cout << "[hpx_main] starting ..." << "\n";

    // get a pointer to the resource_partitioner instance
    hpx::resource::resource_partitioner& rpart = hpx::get_resource_partitioner();

    // print partition characteristics
    hpx::cout << "\n\n [hpx_main] print resource_partitioner characteristics : " << "\n";
    rpart.print_init_pool_data();

    // print partition characteristics
    hpx::cout << "\n\n [hpx_main] print thread-manager characteristics : " << "\n";
    hpx::threads::get_thread_manager().print_pools();

    // print system characteristics
    print_system_characteristics();

    // get executors
    hpx::threads::executors::customized_pool_executor my_exec("first_pool");
    hpx::cout << "\n\n [hpx_main] got customized executor " << "\n";

    // use these executors to schedule work
    my_exec.add(do_stuff);


    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    std::cout << "[main] " << "Starting program... \n";

    auto &rp = hpx::get_resource_partitioner();

    std::cout << "[main] " << "obtained reference to the resource_partitioner\n";

    rp.create_thread_pool("first_pool");
    rp.create_thread_pool("second_pool", hpx::resource::abp_priority);

    std::cout << "[main] " << "thread_pools created \n";

    rp.add_resource(1, "first_pool");
    rp.add_resource(0, "second_pool");
    rp.add_resource_to_default(1);
    rp.add_resource_to_default(3);

    std::cout << "[main] " << "resources added to thread_pools \n";
    std::cout << "[main] " << "Calling hpx::init... \n";

    return hpx::init(argc, argv);
}
