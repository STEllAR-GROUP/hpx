//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/local_lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <vector>

#define NUM_THREADS std::size_t(100)

///////////////////////////////////////////////////////////////////////////////
boost::atomic<std::size_t> num_threads(0);

///////////////////////////////////////////////////////////////////////////////
void test_count_down_and_wait(hpx::lcos::local::latch& l)
{
    ++num_threads;

    HPX_TEST(!l.is_ready());
    l.count_down_and_wait();
}

void test_count_down(hpx::lcos::local::latch& l)
{
    ++num_threads;

    HPX_TEST(!l.is_ready());
    l.count_down(NUM_THREADS);
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    // count_down_and_wait
    {
        hpx::lcos::local::latch l(NUM_THREADS+1);
        HPX_TEST(!l.is_ready());

        std::vector<hpx::future<void> > results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
            results.push_back(hpx::async(&test_count_down_and_wait, std::ref(l)));

        HPX_TEST(!l.is_ready());

        // Wait for all threads to reach this point.
        l.count_down_and_wait();

        hpx::wait_all(results);

        HPX_TEST(l.is_ready());
        HPX_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    // count_down
    {
        num_threads.store(0);

        hpx::lcos::local::latch l(NUM_THREADS+1);
        HPX_TEST(!l.is_ready());

        hpx::future<void> f = hpx::async(&test_count_down, std::ref(l));

        HPX_TEST(!l.is_ready());
        l.count_down_and_wait();

        f.get();

        HPX_TEST(l.is_ready());
        HPX_TEST_EQ(num_threads.load(), std::size_t(1));
    }

    // wait
    {
        num_threads.store(0);

        hpx::lcos::local::latch l(NUM_THREADS);
        HPX_TEST(!l.is_ready());

        std::vector<hpx::future<void> > results;
        for (std::ptrdiff_t i = 0; i != NUM_THREADS; ++i)
            results.push_back(hpx::async(&test_count_down_and_wait, std::ref(l)));

        hpx::wait_all(results);

        l.wait();

        HPX_TEST(l.is_ready());
        HPX_TEST_EQ(num_threads.load(), NUM_THREADS);
    }

    HPX_TEST_EQ(hpx::finalize(), 0);
    return 0;
}

int main(int argc, char* argv[])
{
    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
