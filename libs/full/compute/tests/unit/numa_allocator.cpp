//  Copyright (c) 2017-2018 John Biddiscombe
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define SHARED_PRIORITY_SCHEDULER_DEBUG true
#define THREAD_QUEUE_MC_DEBUG true
#define GUIDED_POOL_EXECUTOR_DEBUG true
#define QUEUE_HOLDER_NUMA_DEBUG true
#define QUEUE_HOLDER_THREAD_DEBUG true
#define NUMA_BINDING_ALLOCATOR_DEBUG true
#define NUMA_BINDING_ALLOCATOR_INIT_MEMORY true

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
//
#include <hpx/modules/executors.hpp>
#include <hpx/parallel/algorithms/for_loop.hpp>
//
#include <hpx/resource_partitioner/partitioner.hpp>
#include <hpx/thread_executors/guided_pool_executor.hpp>
#include <hpx/thread_pools/scheduled_thread_pool_impl.hpp>
#include <hpx/topology/cpu_mask.hpp>
//
#include <hpx/include/runtime.hpp>
#include <hpx/iostream.hpp>
//
#include <hpx/modules/testing.hpp>
//
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
// The allocator that binds pages to numa domains
#include <hpx/compute/host/numa_binding_allocator.hpp>
// Example binder functions for different page binding mappings
#include "allocator_binder_linear.hpp"
#include "allocator_binder_matrix.hpp"
// Scheduler that honours numa placement hints for tasks
#include <hpx/schedulers/shared_priority_queue_scheduler.hpp>

// ------------------------------------------------------------------------
// allocator maker for this test
template <template <typename> class Binder, typename T>
auto get_allocator(std::shared_ptr<Binder<T>> numa_binder, int allocator_mode)
{
    using namespace hpx::compute::host;
    if (allocator_mode == 0)
    {
        return numa_binding_allocator<T>(numa_binder,
            hpx::threads::hpx_hwloc_membind_policy::membind_firsttouch, 0);
    }
    else if (allocator_mode == 1)
    {
        return numa_binding_allocator<T>(numa_binder,
            hpx::threads::hpx_hwloc_membind_policy::membind_interleave, 0);
    }
    else /*if (allocator_mode==2)*/
    {
        return numa_binding_allocator<T>(numa_binder,
            hpx::threads::hpx_hwloc_membind_policy::membind_user, 0);
    }
}

// ------------------------------------------------------------------------
template <template <typename> class Binder, typename T, typename Allocator>
void test_binding(std::shared_ptr<Binder<T>> numa_binder, Allocator& allocator)
{
    // num_numa_domains is only correct when using the default pool
    int num_numa_domains =
        hpx::resource::get_partitioner().numa_domains().size();

    // create a container
    std::vector<T, Allocator> data(allocator);

    // resize it and trigger a page touch function for each page
    std::size_t num_bytes = numa_binder->memory_bytes();
    data.reserve(num_bytes / sizeof(T));
    T* M = data.data();
    // this is a debugging function that returns a string of numa bindings
    std::string domain_string = allocator.get_page_numa_domains(M, num_bytes);

    // now generate the 'correct' string of numa domains per page
    std::size_t const pagesize = hpx::threads::get_memory_page_size();
    std::size_t const pageN = pagesize / sizeof(T);
    std::size_t const num_pages = (num_bytes + pagesize - 1) / pagesize;
    T* page_ptr = M;

    std::stringstream temp;
    temp << "Numa page binding for page count " << num_pages << "\n";
    for (std::size_t i = 0; i < num_pages; ++i)
    {
        // we pass the base pointer and current page pointer
        std::size_t dom =
            numa_binder->operator()(M, page_ptr, pagesize, num_numa_domains);
        temp << dom;
        page_ptr += pageN;
    }
    HPX_TEST_EQ(domain_string, temp.str());

    std::size_t xsize = numa_binder->array_size(0);
    std::size_t ysize = numa_binder->array_size(1);
    std::size_t xstep = numa_binder->display_step(0);
    std::size_t ystep = numa_binder->display_step(1);

    std::cout << "============================\n";
    std::cout << "get_numa_domain() " << num_numa_domains
              << " Domain Numa pattern\n";
    for (unsigned int j = 0; j < ysize; j += ystep)
    {
        for (unsigned int i = 0; i < xsize; i += xstep)
        {
            T* page_ptr = &M[i * numa_binder->memory_step(0) +
                j * numa_binder->memory_step(1)];
            int dom = allocator.get_numa_domain(page_ptr);
            if (dom == -1)
            {
                std::cout << '-';
            }
            else
                std::cout << std::hex << dom;
        }
        std::cout << "\n";
    }
    std::cout << "============================\n\n";

#ifdef NUMA_BINDING_ALLOCATOR_INIT_MEMORY
    std::cout << "============================\n";
    std::cout << "Contents of memory locations\n";
    for (unsigned int j = 0; j < ysize; j += ystep)
    {
        for (unsigned int i = 0; i < xsize; i += xstep)
        {
            T* page_ptr = &M[i * numa_binder->memory_step(0) +
                j * numa_binder->memory_step(1)];
            std::cout << *page_ptr << " ";
        }
        std::cout << "\n";
    }
    std::cout << "============================\n\n";
#endif

    std::cout << "============================\n";
    std::cout << "Expected " << num_numa_domains << " Domain Numa pattern\n";
    for (unsigned int j = 0; j < ysize; j += ystep)
    {
        for (unsigned int i = 0; i < xsize; i += xstep)
        {
            T* page_ptr = &M[i * numa_binder->memory_step(0) +
                j * numa_binder->memory_step(1)];
            int d = numa_binder->operator()(
                M, page_ptr, pagesize, num_numa_domains);
            std::cout << std::hex << d;
        }
        std::cout << "\n";
    }
    std::cout << "============================\n\n";

#ifdef NUMA_BINDING_ALLOCATOR_DEBUG_PAGE_BINDING
    std::cout << allocator.display_binding(M, numa_binder) << std::endl;
#endif
}

// ------------------------------------------------------------------------
// this is called on an hpx thread after the runtime starts up
// ------------------------------------------------------------------------
int hpx_main(hpx::program_options::variables_map& vm)
{
    int Nc = vm["size"].as<int>();
    int Nr = vm["size"].as<int>();
    int Nt = vm["nb"].as<int>();
    int Nd = vm["tiles-per-domain"].as<int>();
    int p = vm["col-proc"].as<int>();
    int q = vm["row-proc"].as<int>();

    std::size_t num_threads = hpx::get_num_worker_threads();
    std::cout << "HPX using threads = " << num_threads << std::endl;

    // ---------------------------------
    using namespace hpx::compute::host;
    using matrix_elem = double;
    int allocator_mode = 2;

    // ---------------------------------
    // test linear 1D array
    std::cout << "Test 1D" << std::endl << std::endl;
    using binder_type_1D = linear_numa_binder<matrix_elem>;
    std::shared_ptr<binder_type_1D> numa_binder_1D =
        std::make_shared<binder_type_1D>(Nc);

    auto allocator_1D = get_allocator(numa_binder_1D, allocator_mode);
    test_binding(numa_binder_1D, allocator_1D);

    // ---------------------------------
    // test 2D matrix : todo add tiles per page etc
    std::cout << "Test 2D" << std::endl << std::endl;
    using binder_type_2D = matrix_numa_binder<matrix_elem>;
    std::shared_ptr<binder_type_2D> numa_binder_2D =
        std::make_shared<binder_type_2D>(10 * Nc, 10 * Nr, Nt, Nd, p, q);

    auto allocator_2D = numa_binding_allocator<matrix_elem>(numa_binder_2D,
        hpx::threads::hpx_hwloc_membind_policy::membind_user, 0);
    test_binding(numa_binder_2D, allocator_2D);

    return hpx::finalize();
}

// ------------------------------------------------------------------------
// scheduler type needed for numa bound tasks
// ------------------------------------------------------------------------
using high_priority_sched =
    hpx::threads::policies::shared_priority_queue_scheduler<>;
using hpx::threads::policies::scheduler_mode;

void init_resource_partitioner_handler(
    hpx::resource::partitioner& rp, hpx::program_options::variables_map const&)
{
    using numa_scheduler =
        hpx::threads::policies::shared_priority_queue_scheduler<>;
    using hpx::threads::policies::scheduler_mode;
    // setup the default pool with a numa aware scheduler
    rp.create_thread_pool("default",
        [](hpx::threads::thread_pool_init_parameters init,
            hpx::threads::policies::thread_queue_init_parameters
                thread_queue_init)
            -> std::unique_ptr<hpx::threads::thread_pool_base> {
            numa_scheduler::init_parameter_type scheduler_init(
                init.num_threads_, {1, 1, 64}, init.affinity_data_,
                thread_queue_init, "shared-priority-scheduler");
            std::unique_ptr<numa_scheduler> scheduler(
                new numa_scheduler(scheduler_init));

            scheduler_mode mode =
                scheduler_mode(scheduler_mode::do_background_work |
                    scheduler_mode::delay_exit);
            init.mode_ = mode;

            std::unique_ptr<hpx::threads::thread_pool_base> pool(
                new hpx::threads::detail::scheduled_thread_pool<
                    high_priority_sched>(std::move(scheduler), init));
            return pool;
        });
}

// the normal int main function that is called at startup and runs on an OS thread
// the user must call hpx::init to start the hpx runtime which will execute hpx_main
// on an hpx thread
int main(int argc, char* argv[])
{
    hpx::program_options::options_description desc_cmdline("Test options");
    // clang-format off
    desc_cmdline.add_options()
        ("size,n",
         hpx::program_options::value<int>()->default_value(1024),
         "Matrix size.")
        ("tiles-per-domain,t",
         hpx::program_options::value<int>()->default_value(1),
        "Number of Tiles per numa domain.")
        ("nb",
         hpx::program_options::value<int>()->default_value(128),
        "Block cyclic distribution size.")
        ("row-proc,p",
         hpx::program_options::value<int>()->default_value(1),
        "Number of row processes in the 2D communicator.")
        ("col-proc,q",
         hpx::program_options::value<int>()->default_value(1),
        "Number of column processes in the 2D communicator.")
        ("no-check",
         "Disable result checking")
        // clang-format on
        ;

    // HPX uses a boost program options variable map, but we need it before
    // hpx-main, so we will create another one here and throw it away after use
    hpx::program_options::variables_map vm;
    hpx::program_options::store(
        hpx::program_options::command_line_parser(argc, argv)
            .allow_unregistered()
            .options(desc_cmdline)
            .run(),
        vm);

    // Setup the init parameters
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_cmdline;
    // Set the callback to init the thread_pools
    init_args.rp_callback = &init_resource_partitioner_handler;

    hpx::init(argc, argv, init_args);
    return hpx::util::report_errors();
}
