//  Copyright (c) 2017 Shoshana Jakobovits
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/threadmanager.hpp>
#include <hpx/runtime.hpp>

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

    //! -------------------------------------- cache sizes
    hpx::threads::mask_type core0 = topo.get_core_affinity_mask(0);
    std::cout << "[System Cache sizes (core 0)]\n"
              << "L1 Cache: " << topo.get_cache_size(core0, 1) << "\n"
              << "L2 Cache: " << topo.get_cache_size(core0, 2) << "\n"
              << "L3 Cache: " << topo.get_cache_size(core0, 3) << "\n\n";

    hpx::threads::mask_type machine = topo.get_machine_affinity_mask();
    std::cout << "[System Cache sizes (all available cores)]\n"
              << "L1 Cache: " << topo.get_cache_size(machine, 1) << "\n"
              << "L2 Cache: " << topo.get_cache_size(machine, 2) << "\n"
              << "L3 Cache: " << topo.get_cache_size(machine, 3) << "\n\n";
}
