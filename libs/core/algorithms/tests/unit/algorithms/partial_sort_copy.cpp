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
#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/iostream.hpp>
#include <hpx/execution.hpp>
#include <ciso646>
#include <hpx/assert.hpp>
#include <chrono>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <hpx/parallel/algorithms/partial_sort_copy.hpp>

namespace hpp = ::hpx::parallel;

//---------------------------------------------------------------------------
// Check with the three cases
// a) Output size smaller than input size
// b) Output size equal than input size
// c) Output size greter than input size
//---------------------------------------------------------------------------
void function01(void)
{
	std::list<uint64_t> l = {9, 7, 6, 8, 5, 4, 1, 2, 3};
	uint64_t v1[20], v2[20];

	//------------------------------------------------------------------------
	// Output size is smaller than input size
	//------------------------------------------------------------------------
	for ( int i = 0; i < 20; ++i) 	v1[i] = v2[i] = 999;

	hpxp::partial_sort_copy (l.begin(), l.end(), &v1[0], & v1[4]);

	std::partial_sort_copy (l.begin(), l.end(), &v2[0], & v2[4]);
	//hpx::cout<<"Expected : 1, 2, 3, 4 \n";
	for ( int i =0 ; i < 4; ++i)
	{
		assert (v1[i] == v2[i]);
	};
	for ( int i = 4 ; i < 20; ++i)
	{
		assert (v1[i] == v2[i]);
	};

	//------------------------------------------------------------------------
	// Output size is equal than input size
	//------------------------------------------------------------------------
	for ( int i = 0; i < 20; ++i) 	v1[i] = v2[i] = 999;

	hpxp::partial_sort_copy (l.begin(), l.end(), &v1[0], & v1[9]);
	std::partial_sort_copy (l.begin(), l.end(), &v2[0], & v2[9]);
	//hpx::cout<<"Expected : 1, 2, 3, 4, 5, 6, 7, 8, 9\n";
	for ( int i =0 ; i < 9; ++i)
	{
		assert (v1[i] == v2[i]);
	};
	for ( int i = 9 ; i < 20; ++i)
	{
		assert (v1[i] == v2[i]);
	};

	//------------------------------------------------------------------------
	// Output size is greater than input size
	//------------------------------------------------------------------------
	for ( int i = 0; i < 20; ++i) 	v1[i] = v2[i] = 999;

	hpxp::partial_sort_copy (l.begin(), l.end(), &v1[0], & v1[20]);
	std::partial_sort_copy (l.begin(), l.end(), &v2[0], & v2[20]);
	//hpx::cout<<"Expected : 1, 2, 3, 4, 5, 6, 7, 8, 9\n";
	for ( int i =0 ; i < 9; ++i)
	{
		assert (v1[i] == v2[i]);
	};
	for ( int i = 9 ; i < 20; ++i)
	{
		assert (v1[i] == v2[i]);
	};

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
};

int test_main (void)
{
	hpx::cout<<"----------------- test_partial_sort_copy -------------------\n";
	function01();
    function02();
    function03();
    hpx::cout<<"------------------------ end -------------------------------\n";
    return 0;
}
int main(int argc, char* argv[])
{
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=all");

    // Initialize and run HPX.
    return hpx::init(argc, argv, cfg);
}
int hpx_main(boost::program_options::variables_map&)
{
	{	test_main() ;
    };
    // Initiate shutdown of the runtime systems on all localities.
    return hpx::finalize();
};
