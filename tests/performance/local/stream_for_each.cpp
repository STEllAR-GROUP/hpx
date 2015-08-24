//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_algorithm.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/iostreams.hpp>

#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>
#include <boost/range/functions.hpp>

#include <vector>
#include <string>

///////////////////////////////////////////////////////////////////////////////
hpx::threads::topology const& retrieve_topology()
{
    static hpx::threads::topology const& topo = hpx::threads::create_topology();
    return topo;
}

///////////////////////////////////////////////////////////////////////////////
boost::uint64_t numa_domain_worker(std::size_t domain,
    hpx::lcos::local::latch& l, std::size_t vector_size)
{
    hpx::threads::topology const& topo = retrieve_topology();
    std::size_t pus = topo.get_number_of_numa_node_pus(domain);

    std::string bind_desc = boost::str(
            boost::format("thread:0-%d=numanode:%d.pu:0-%d") %
                (pus-1) % domain % (pus-1)
        );

    // allocate data
    std::vector<char> src(vector_size), dest(vector_size);

    // create executor for this NUMA domain
    hpx::threads::executors::local_priority_queue_os_executor exec(
        pus, bind_desc);

    // synchronize across NUMA domains
    l.count_down_and_wait();

    boost::uint64_t start = hpx::util::high_resolution_clock::now();

    // perform actual benchmark
    hpx::parallel::copy(
        hpx::parallel::par.on(exec),
        boost::begin(src), boost::end(src),
        boost::begin(dest));

    return hpx::util::high_resolution_clock::now() - start;
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    std::size_t vector_size = vm["vector_size"].as<std::size_t>();

    hpx::threads::topology const& topo = retrieve_topology();
    std::size_t numa_domains = topo.get_number_of_numa_nodes();

    hpx::lcos::local::latch l(numa_domains);

    std::vector<hpx::future<boost::uint64_t> > workers;
    workers.reserve(numa_domains);

    for (std::size_t i = 0; i != numa_domains; ++i)
    {
        // create one worker per NUMA domain with part of the data to work on
        hpx::threads::executors::default_executor exec(i);
        workers.push_back(
            hpx::async(exec, &numa_domain_worker, i, boost::ref(l),
                vector_size/numa_domains)
        );
    }

    hpx::wait_all(workers);

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // extract hardware topology
    hpx::threads::topology const& topo = retrieve_topology();
    std::size_t numa_nodes = topo.get_number_of_numa_nodes();
    std::size_t pus_per_numa_node = topo.get_number_of_numa_node_pus(0);

    // The idea of this benchmark is to create as many base-threads as we have
    // NUMA domains. Each of those kernel threads are bound to one of the
    // domains such that they can wander between the cores of this domain.
    //
    // The benchmark uses the static_priority scheduler for this which prevents
    // HPX threads from being stolen across the NUMA domain boundaries.
    //
    // The benchmark itself spawns one HPX-thread for each of those kernel
    // threads. Each HPX thread creates a new local_priority os_executor which
    // is then used to run the actual measurements.

    // create one kernel thread per available NUMA domain
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(numa_nodes));

    // use full machine
    cfg.push_back("hpx.cores=all");

    // run the static_priority scheduler
    cfg.push_back("hpx.scheduler=static-priority");

    // set affinity domain for the base scheduler threads to 'numa'
    cfg.push_back("hpx.affinity=numa");

    // make sure each of the base kernel-threads run on separate NUMA domain
    cfg.push_back("hpx.pu_step=" +
        boost::lexical_cast<std::string>(pus_per_numa_node));

    boost::program_options::options_description cmdline(
        "usage: " HPX_APPLICATION_STRING " [options]");

    cmdline.add_options()
        (   "vector_size",
            boost::program_options::value<std::size_t>()->default_value(1000),
            "size of vector (default: 1000)")
        ;

    return hpx::init(cmdline, argc, argv, cfg);
}

