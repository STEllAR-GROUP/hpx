//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/include/runtime.hpp>
#include <hpx/modules/threadmanager.hpp>

#include <iostream>

void print_system_characteristics()
{
    std::cout << "[hpx-main] System queries: \n\n";

    // -------------------------------------- //
    //      get pointer to instances          //
    //      I can query                       //
    // -------------------------------------- //

    hpx::runtime* rt = hpx::get_runtime_ptr();
    hpx::util::runtime_configuration cfg = rt->get_config();
    const hpx::threads::topology& topo = rt->get_topology();

    // -------------------------------------- //
    //      print runtime characteristics     //
    //                                        //
    // -------------------------------------- //

    //! -------------------------------------- runtime
    std::cout << "[Runtime], instance number " << rt->get_instance_number()
              << "\n"
              << "called by thread named     " << hpx::get_thread_name()
              << "\n\n";

    //! -------------------------------------- thread_manager
    std::cout << "[Thread manager]\n"
              << "worker thread number  : " << std::dec
              << hpx::get_worker_thread_num() << "\n\n";

    //! -------------------------------------- runtime_configuration
    std::cout << "[Runtime configuration]\n"
              << "number of localities  : " << cfg.get_num_localities() << "\n"
              << "os thread count       : " << cfg.get_os_thread_count() << "\n"
              << "                        " << hpx::get_os_thread_count()
              << "\n"
              << "command line          : " << cfg.get_cmd_line() << "\n\n";

    //! -------------------------------------- topology
    topo.print_hwloc(std::cout);
}
