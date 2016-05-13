///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/include/compute.hpp>
#include <hpx/include/parallel_for_loop.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_copy.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <hpx/hpx_init.hpp>

#include <numeric>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

typedef hpx::compute::cuda::default_executor executor_type;
typedef hpx::compute::cuda::allocator<int> target_allocator;
typedef hpx::compute::vector<int, target_allocator> target_vector;

void test_for_loop(executor_type& exec,
    target_vector& d_A, target_vector& d_B, target_vector& d_C, std::vector<int> const& ref)
{
    hpx::parallel::for_loop_n(
        hpx::parallel::par.on(exec),
        d_A.data(), d_A.size(),
        hpx::parallel::induction(d_B.data()),
        hpx::parallel::induction(d_C.data()),
        [] HPX_DEVICE (int* A, int* B, int* C)
        {
            *C = *A + *B;
        });

    std::vector<int> h_C(d_C.size());
    hpx::parallel::copy(
        hpx::parallel::par,
        d_C.begin(), d_C.end(), h_C.begin());

    HPX_TEST_EQ(h_C.size(), ref.size());
    HPX_TEST_EQ(d_C.size(), ref.size());
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        HPX_TEST_EQ(h_C[i], ref[i]);
        HPX_TEST_EQ(d_C[i], ref[i]);
    }
}

void test_for_each(executor_type& exec,
    target_vector& d_A, target_vector& d_B, target_vector& d_C, std::vector<int> const& ref)
{
    std::vector<int> h_C(d_C.size());
    hpx::parallel::copy(
        hpx::parallel::par,
        d_A.begin(), d_A.end(), h_C.begin());

    hpx::parallel::for_each(
        hpx::parallel::par.on(exec),
        d_A.data(), d_A.data() + d_A.size(),
        [] HPX_DEVICE (int & i) mutable
        {
             i += 5;
        });

    HPX_TEST_EQ(h_C.size(), ref.size());
    HPX_TEST_EQ(d_C.size(), ref.size());
    for(std::size_t i = 0; i < ref.size(); ++i)
    {
        HPX_TEST_EQ(h_C[i] + 5, d_A[i]);
    }
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    int const N = 100;
    std::vector<int> h_A(N);
    std::vector<int> h_B(N);
    std::vector<int> h_C_ref(N);
    std::vector<int> h_C(N);

    std::iota(h_A.begin(), h_A.end(), (std::rand() % 100) + 2);
    std::iota(h_B.begin(), h_B.end(), (std::rand() % 100) + 2);

    std::transform(
        h_A.begin(), h_A.end(), h_B.begin(), h_C_ref.begin(),
        [](int a, int b) { return a + b; });

    // define execution target (here device 0)
    hpx::compute::cuda::target target;

    // allocate data on the device
    target_allocator alloc(target);

    target_vector d_A(N, alloc);
    target_vector d_B(N, alloc);
    target_vector d_C(N, alloc);

    // copy data to device
    hpx::parallel::copy(
        hpx::parallel::par,
        h_A.begin(), h_A.end(), d_A.begin());
    hpx::parallel::copy(
        hpx::parallel::par,
        h_B.begin(), h_B.end(), d_B.begin());

    // create executor
    executor_type exec(target);

    // Run tests:
    test_for_loop(exec, d_A, d_B, d_C, h_C_ref);
    test_for_each(exec, d_A, d_B, d_C, h_C_ref);

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

    // Initialize and run HPX
    hpx::init(desc_commandline, argc, argv);

    return hpx::util::report_errors();
}
