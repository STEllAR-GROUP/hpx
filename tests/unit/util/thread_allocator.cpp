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
#include <random>

using hpx::util::thread_alloc;
using hpx::util::thread_free;

///////////////////////////////////////////////////////////////////////////////
int* test_malloc_no_free(std::size_t num_ints, std::size_t num_alloc)
{
    int* p = reinterpret_cast<int*>(thread_alloc(sizeof(int) * num_ints));
    HPX_TEST(p != nullptr);

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

void test_malloc(std::size_t num_ints, std::size_t num_alloc)
{
    int* p = test_malloc_no_free(num_ints, num_alloc);
    thread_free(p);
}

template <typename Policy>
void test_mallocs(Policy && policy)
{
    hpx::parallel::for_loop(
        policy, std::size_t(0), std::size_t(256),
        [](std::size_t num_ints)
        {
            test_malloc(num_ints, num_ints);
        });
}

///////////////////////////////////////////////////////////////////////////////
template <typename Policy>
void test_malloc(Policy&& policy, std::mt19937& g, std::size_t num_ints)
{
    std::vector<int*> ptrs(10000, nullptr);

    hpx::parallel::for_loop(
        policy, std::size_t(0), ptrs.capacity(),
        [&](std::size_t j)
        {
            ptrs[j] = test_malloc_no_free(num_ints, j);
        });

    // free pointers in random order possibly from various threads
    std::shuffle(ptrs.begin(), ptrs.end(), g);

    hpx::parallel::for_each(
        policy, ptrs.begin(), ptrs.end(),
        [](int* p)
        {
            thread_free(p);
        });
}

template <typename Policy>
void test_mallocs(Policy&& policy, std::mt19937& g)
{
    for (std::size_t num_ints = 1; num_ints != 256; ++num_ints)
    {
        test_malloc(policy, g, num_ints);
    }
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    std::random_device rd;
    std::mt19937 g(rd());

    // simple tests on a single thread
    test_mallocs(hpx::parallel::execution::seq);
    test_mallocs(hpx::parallel::execution::seq, g);

    // simple tests on many threads
    test_mallocs(hpx::parallel::execution::par);
    test_mallocs(hpx::parallel::execution::par, g);

    return hpx::util::report_errors();
}

