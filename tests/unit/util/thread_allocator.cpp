// Copyright (c) 2018 Hartmut Kaiser.
//
// Distributed under the Boost Software License, Version 1.0. (See
// accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>

#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_for_loop.hpp>

#include <hpx/util/thread_allocator.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using hpx::util::thread_alloc;
using hpx::util::thread_free;

HPX_CONSTEXPR std::size_t ARRAY_SIZE = 10000;
HPX_CONSTEXPR std::size_t NUM_INTS = 200;

///////////////////////////////////////////////////////////////////////////////
template <typename Allocator>
int* test_malloc_no_free(
    Allocator& a, std::size_t num_ints, std::size_t num_alloc)
{
    int* p = reinterpret_cast<int*>(a.allocate(sizeof(int) * num_ints));
    HPX_TEST(p != nullptr || num_ints == 0);

    for (std::size_t i = 0; i != num_ints; ++i)
    {
        p[i] = static_cast<int>(num_alloc);
    }

    for (std::size_t i = 0; i != num_ints; ++i)
    {
        HPX_TEST_EQ(p[i], num_alloc);
    }

    return p;
}

///////////////////////////////////////////////////////////////////////////////
template <typename Allocator>
void test_malloc(Allocator& a, std::size_t num_ints, std::size_t num_alloc)
{
    int* p = test_malloc_no_free(a, num_ints, num_alloc);
    a.deallocate(p, num_ints);
}

template <typename Allocator>
void test_mallocs_seq(Allocator& a)
{
    for (std::size_t i = 0; i != ARRAY_SIZE; ++i)
    {
        for (std::size_t num_ints = 0; num_ints != NUM_INTS; ++num_ints)
        {
            test_malloc(a, num_ints, num_ints);
        }
    }
}

template <typename Allocator>
void test_mallocs_par(Allocator& a)
{
    for (std::size_t i = 0; i != ARRAY_SIZE; ++i)
    {
        hpx::parallel::for_loop(
            hpx::parallel::execution::par, std::size_t(0), NUM_INTS,
            [&](std::size_t num_ints)
            {
                test_malloc(a, num_ints, i);
            });
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Allocator>
void test_malloc_seq(
    Allocator& a, std::vector<std::size_t> const& indices, std::size_t num_ints)
{
    std::vector<int*> ptrs(ARRAY_SIZE, nullptr);

    for (std::size_t i = 0; i != ptrs.capacity(); ++i)
    {
        ptrs[i] = test_malloc_no_free(a, num_ints, i);
    }

    // free pointers in random order
    for (auto idx: indices)
    {
        a.deallocate(ptrs[idx], num_ints);
    }
}

template <typename Allocator>
void test_malloc_par(
    Allocator& a, std::vector<std::size_t> const& indices, std::size_t num_ints)
{
    std::vector<int*> ptrs(ARRAY_SIZE, nullptr);

    hpx::parallel::for_loop(
        hpx::parallel::execution::par, std::size_t(0), ptrs.capacity(),
        [&](std::size_t j)
        {
            ptrs[j] = test_malloc_no_free(a, num_ints, j);
        });

    // free pointers in random order possibly from various threads
    hpx::parallel::for_each(
        hpx::parallel::execution::par, indices.begin(), indices.end(),
        [&](std::size_t idx)
        {
            a.deallocate(ptrs[idx], num_ints);
        });
}

template <typename Allocator>
void test_mallocs_seq(Allocator& a, std::vector<std::size_t> const& indices)
{
    for (std::size_t num_ints = 1; num_ints != NUM_INTS; ++num_ints)
    {
        test_malloc_seq(a, indices, num_ints);
    }
}

template <typename Allocator>
void test_mallocs_par(Allocator& a, std::vector<std::size_t> const& indices)
{
    for (std::size_t num_ints = 1; num_ints != NUM_INTS; ++num_ints)
    {
        test_malloc_par(a, indices, num_ints);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Allocator>
void test_allocator(
    char const* allocator_name, Allocator& a, std::mt19937& g)
{
    // warm up caches
    test_mallocs_seq(a);

    // fill array with randomized indices
    std::vector<std::size_t> indices(ARRAY_SIZE);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);

    std::cout << "Testing: " << allocator_name << "\n";

    double elapsed1, elapsed2, elapsed3, elapsed4;

    // simple tests on a single thread
    {
        hpx::util::high_resolution_timer t;
        test_mallocs_seq(a);
        elapsed1 = t.elapsed();
    }
    {
        hpx::util::high_resolution_timer t;
        test_mallocs_seq(a, indices);
        elapsed2 = t.elapsed();
    }

    // simple tests on many threads
    {
        hpx::util::high_resolution_timer t;
        test_mallocs_par(a);
        elapsed3 = t.elapsed();
    }
    {
        hpx::util::high_resolution_timer t;
        test_mallocs_par(a, indices);
        elapsed4 = t.elapsed();
    }

    std::cout << "Sequential                  : " << elapsed1 << " [s]\n";
    std::cout << "Sequential (randomized free): " << elapsed2 << " [s]\n";
    std::cout << "Parallel                    : " << elapsed3 << " [s]\n";
    std::cout << "Parallel (randomized free)  : " << elapsed4 << " [s]\n";
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    std::random_device rd;
    std::mt19937 g(rd());

    test_allocator("std::allocator", std::allocator<int>{}, g);
    test_allocator(
        "hpx::util::thread_allocator", hpx::util::thread_allocator<int>{}, g);

    return hpx::util::report_errors();
}

