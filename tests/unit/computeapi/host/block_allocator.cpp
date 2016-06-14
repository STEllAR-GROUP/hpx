//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/compute.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename T>
T* test_block_allocation(hpx::compute::host::block_allocator<T>& alloc,
    std::size_t count)
{
    return alloc.allocate(count);
}

template <typename T>
void test_block_construction(hpx::compute::host::block_allocator<T>& alloc,
    T* p, std::size_t count)
{
    alloc.bulk_construct(p, count);
}

template <typename T>
void test_block_destruction(hpx::compute::host::block_allocator<T>& alloc,
    T* p, std::size_t count)
{
    return alloc.bulk_destroy(p, count);
}

template <typename T>
void test_block_deallocation(hpx::compute::host::block_allocator<T>& alloc,
    T* p, std::size_t count)
{
    return alloc.deallocate(p, count);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void test_bulk_allocator(std::size_t count)
{
    hpx::compute::host::block_allocator<T> alloc;
    T* p = test_block_allocation(alloc, count);
    test_block_construction(alloc, p, count);
    test_block_destruction(alloc, p, count);
    test_block_deallocation(alloc, p, count);
}

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> construction_count(0);
boost::atomic<std::size_t> destruction_count(0);

struct test
{
    test()
    {
        ++construction_count;
    }
    ~test()
    {
        ++destruction_count;
    }
};

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    {
        std::size_t count = std::rand();
        test_bulk_allocator<int>(count);
    }

    {
        std::size_t count = std::rand();
        test_bulk_allocator<test>(count);
        HPX_TEST_EQ(construction_count.load(), count);
        HPX_TEST_EQ(destruction_count.load(), count);
    }

    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
