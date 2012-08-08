//  Copyright (c) 2012 Vinay C Amatya                                       
//                                                                               
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/performance_counters.hpp>
#include <boost/atomic.hpp>
#include "mem_counter.hpp"

typedef hpx::performance_counters::server::proc_statm proc_statm_type;

namespace read_mem = hpx::performance_counters::server;

void register_counter_type()
{
    namespace pc = hpx::performance_counters;
    pc::install_counter_type(
        "/memory/vm",
        &read_mem::read_psm_vm,
        "returns the virtual memory for the pid value of process which calls this counter"
    );
    pc::install_counter_type(
        "/memory/resident",
        &read_mem::read_psm_resident,
        "returns the resident memory for the pid value of process which calls this counter"
    );
}

int hpx_main()
{
    {
    }
    hpx::this_thread::suspend(10000);
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    hpx::register_startup_function(&register_counter_type);
    return hpx::init(argc, argv);
}
