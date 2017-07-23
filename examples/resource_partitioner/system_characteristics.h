//  Copyright (c) 2017 Shoshana Jakobovits
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef RM_EXPERIMENT_SYSTEM_CHARACTERISTICS_H_H
#define RM_EXPERIMENT_SYSTEM_CHARACTERISTICS_H_H

#include <hpx/include/runtime.hpp>
#include <hpx/runtime/threads/threadmanager.hpp>
#include <hpx/runtime_impl.hpp>

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
    hpx::threads::threadmanager_base& thrd_manager = rt->get_thread_manager();

    // -------------------------------------- //
    //      print runtime characteristics     //
    //                                        //
    // -------------------------------------- //

    //! -------------------------------------- runtime
    std::cout << "[Runtime], instance number " << rt->get_instance_number()
              << "\n"
              << "called by thread named     " << rt->get_thread_name()
              << "\n\n";

    //! -------------------------------------- thread_manager
    std::cout << "[Thread manager]\n"
              << "worker thread number  : "
              << thrd_manager.get_worker_thread_num() << "\n\n";

    //! -------------------------------------- runtime_configuration
    std::cout << "[Runtime configuration]\n"
              << "number of localities  : " << cfg.get_num_localities() << "\n"
              << "os thread count       : " << cfg.get_os_thread_count() << "\n"
              << "                        " << hpx::get_os_thread_count()
              << "\n"
              << "command line          : " << cfg.get_cmd_line() << "\n\n";

    //! -------------------------------------- affinity data
    /*
    std::size_t num_of_affinity_masks(affdat_ptr->affinity_masks_.size());
    unsigned long num_of_pu_nums(affdat_ptr->pu_nums_.size());
    std::cout << "[Affinity Data]\n"
              << "number of threads     : " << affdat_ptr->num_threads_ << "\n"
              << "affinity domain       : " << affdat_ptr->affinity_domain_ << "\n"
              << "number of pu_nums_    : " << num_of_pu_nums << "\n"
              << "number of aff. masks  : " << num_of_affinity_masks << "\n"
              << "affinity masks        : " << "\n";

    for(std::size_t i(0); i<num_of_affinity_masks; i++){
        std::cout << "                        " << std::bitset<8>(affdat_ptr->affinity_masks_[i]) << "\n";
    }
*/
    /*    std::cout << "pu_nums               : " << "\n";
    for(unsigned long i(0); i<num_of_pu_nums; i++){
        std::cout << "                        " << std::bitset<8>(affdat_ptr->pu_nums_[i]) << ",  " << affdat_ptr->pu_nums_[i] << "\n";
    }*/

    //! -------------------------------------- topology
    topo.print_hwloc();
    /*    std::cout << "[Topology]\n"
              << "number of sockets     : " << topo.get_number_of_sockets() << "\n"
              << "number of numa nodes  : " << topo.get_number_of_numa_nodes() << "\n"
              << "number of cores       : " << topo.get_number_of_cores() << "\n"
              << "number of PUs         : " << topo.get_number_of_pus() << "\n"
              << "hardware concurrency  : " << hpx::threads::hardware_concurrency() << hpx::flush << "\n\n";
*/
    //! -------------------------------------- topology (masks)
    /*    std::cout << "[Topology] masks :\n"
              << "machine               : " << bitset_type(topo.get_machine_affinity_mask()) << "\n";*/

    /*    std::cout << "[Topology]\n"
              << "vector of socket numbers : \n";
    print_vector(topo.get_socket_numbers_());
    std::cout << "vector of numa-node numbers : \n";
    print_vector(topo.get_numa_node_numbers_());
    std::cout << "vector of core numbers : \n";
    print_vector(topo.get_core_numbers_());
    std::cout << "vector of PU numbers : \n";
    print_vector(topo.get_pu_numbers_());*/
}

#endif    //RM_EXPERIMENT_SYSTEM_CHARACTERISTICS_H_H
