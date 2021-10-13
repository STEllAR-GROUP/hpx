//----------------------------------------------------------------------------
/// \file test_partial_sort_copy.cpp
/// \brief Test program of the partial_sort_copy function
///
//  Copyright (c) 2020 Francisco Jose Tapia (fjtapia@gmail.com )
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//-----------------------------------------------------------------------------
#include <hpx/local/execution.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/partial_sort_copy.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

template <typename IteratorTag>
void test_partial_sort(IteratorTag)
{
    std::list<uint64_t> l = {9, 7, 6, 8, 5, 4, 1, 2, 3};
    uint64_t v1[20], v2[20];

    //------------------------------------------------------------------------
    // Output size is smaller than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    hpx::partial_sort_copy(l.begin(), l.end(), &v1[0], &v1[4]);
    std::partial_sort_copy(l.begin(), l.end(), &v2[0], &v2[4]);

    for (int i = 0; i < 4; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 4; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };

    //------------------------------------------------------------------------
    // Output size is equal than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    hpx::partial_sort_copy(l.begin(), l.end(), &v1[0], &v1[9]);
    std::partial_sort_copy(l.begin(), l.end(), &v2[0], &v2[9]);

    for (int i = 0; i < 9; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 9; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };

    //------------------------------------------------------------------------
    // Output size is greater than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    hpx::partial_sort_copy(l.begin(), l.end(), &v1[0], &v1[20]);
    std::partial_sort_copy(l.begin(), l.end(), &v2[0], &v2[20]);

    for (int i = 0; i < 9; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 9; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
}

template <typename ExPolicy, typename IteratorTag>
void test_partial_sort(ExPolicy policy, IteratorTag)
{
    std::list<uint64_t> l = {9, 7, 6, 8, 5, 4, 1, 2, 3};
    uint64_t v1[20], v2[20];

    //------------------------------------------------------------------------
    // Output size is smaller than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    hpx::partial_sort_copy(policy, l.begin(), l.end(), &v1[0], &v1[4]);
    std::partial_sort_copy(l.begin(), l.end(), &v2[0], &v2[4]);

    for (int i = 0; i < 4; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 4; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };

    //------------------------------------------------------------------------
    // Output size is equal than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    hpx::partial_sort_copy(policy, l.begin(), l.end(), &v1[0], &v1[9]);
    std::partial_sort_copy(l.begin(), l.end(), &v2[0], &v2[9]);

    for (int i = 0; i < 9; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 9; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };

    //------------------------------------------------------------------------
    // Output size is greater than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    hpx::partial_sort_copy(policy, l.begin(), l.end(), &v1[0], &v1[20]);
    std::partial_sort_copy(l.begin(), l.end(), &v2[0], &v2[20]);

    for (int i = 0; i < 9; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 9; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
}

template <typename ExPolicy, typename IteratorTag>
void test_partial_sort_async(ExPolicy p, IteratorTag)
{
    std::list<uint64_t> l = {9, 7, 6, 8, 5, 4, 1, 2, 3};
    std::vector<uint64_t> v1(20);
    std::vector<uint64_t> v2(20);

    //------------------------------------------------------------------------
    // Output size is smaller than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    auto f = hpx::partial_sort_copy(
        p, l.begin(), l.end(), v1.begin(), v1.begin() + 4);
    f.wait();
    std::partial_sort_copy(l.begin(), l.end(), v2.begin(), v2.begin() + 4);

    for (int i = 0; i < 4; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 4; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };

    //------------------------------------------------------------------------
    // Output size is equal than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    auto f1 = hpx::partial_sort_copy(
        p, l.begin(), l.end(), v1.begin(), v1.begin() + 9);
    f1.wait();
    std::partial_sort_copy(l.begin(), l.end(), v2.begin(), v2.begin() + 9);

    for (int i = 0; i < 9; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 9; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };

    //------------------------------------------------------------------------
    // Output size is greater than input size
    //------------------------------------------------------------------------
    for (int i = 0; i < 20; ++i)
        v1[i] = v2[i] = 999;

    auto f2 =
        hpx::partial_sort_copy(p, l.begin(), l.end(), v1.begin(), v1.end());
    f2.wait();
    std::partial_sort_copy(l.begin(), l.end(), v2.begin(), v2.end());

    for (int i = 0; i < 9; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
    for (int i = 9; i < 20; ++i)
    {
        HPX_TEST(v1[i] == v2[i]);
    };
}

template <typename IteratorTag>
void test_partial_sort()
{
    using namespace hpx::execution;
    test_partial_sort(IteratorTag());
    test_partial_sort(seq, IteratorTag());
    test_partial_sort(par, IteratorTag());
    test_partial_sort(par_unseq, IteratorTag());

    test_partial_sort_async(seq(task), IteratorTag());
    test_partial_sort_async(par(task), IteratorTag());
}

void partial_sort_test()
{
    test_partial_sort<std::random_access_iterator_tag>();
    test_partial_sort<std::forward_iterator_tag>();
}

/*
//---------------------------------------------------------------------------
// Check with the three cases
// a) Output size smaller than input size
// b) Output size equal than input size
// c) Output size greter than input size
//---------------------------------------------------------------------------
void function01(void)
{

};
//---------------------------------------------------------------------------
// This function check all the sizes in a list of 10000 elements, and checks
// with the version of the standard library
//---------------------------------------------------------------------------
void function02 (void)
{
    typedef std::less<uint64_t>   compare_t;
    std::list <uint64_t> lst;
    std::mt19937 my_rand (0);
    std::vector<uint64_t> A, B;
    const uint32_t NELEM = 10000;
    A.reserve(NELEM);
    B.reserve(NELEM);
   

    for ( uint64_t i = 0; i < NELEM; ++i)
    {
    	A.emplace_back (i);
    	B.emplace_back (0);
    }

    std::shuffle( A.begin(), A.end(), my_rand);
    lst.insert (lst.end(), A.begin(), A.end() );

    for (uint64_t i = 0; i <= NELEM; ++i)
    {	A = B;

    	hpxp::partial_sort_copy (::hpx::execution::seq,
    			                 lst.begin(), lst.end(),
    			                 A.begin(), A.begin() + i, compare_t());

    	for ( uint64_t j =0 ; j < i; ++j)
		{
    		assert (A[j] == j);
		};
    };
};
//-----------------------------------------------------------------------------
// This function check the partial_sort_copy from a list to several output
// sizes nd compare with the standard library implementation
//-----------------------------------------------------------------------------

void function03 ( void)
{
    typedef std::less<uint64_t>   compare_t;
    std::list <uint64_t> lst;
    std::mt19937 my_rand (0);
    std::vector<uint64_t> A, B, C;
    const uint32_t NELEM = 1000000;
    A.reserve(NELEM);
    B.reserve(NELEM);
    C.reserve(NELEM);

    for ( uint64_t i = 0; i < NELEM; ++i)
    {
    	A.emplace_back (i);
    	B.emplace_back (0);
    }

    std::shuffle( A.begin(), A.end(), my_rand);
    lst.insert (lst.end(), A.begin(), A.end() );

	const uint32_t STEP = NELEM / 20 ;

	for (uint64_t i = 0; i <= NELEM; i += STEP)
    {	A = B ;
    	hpxp::partial_sort_copy (::hpx::execution::seq,
    			                 lst.begin(), lst.end(),
    			                 A.begin() , A.begin() + i, compare_t());

    	for ( uint64_t j =0 ; j < i; ++j)
		{
    		assert (A[j] == j);
		};
    };
};*/

int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    partial_sort_test();

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
