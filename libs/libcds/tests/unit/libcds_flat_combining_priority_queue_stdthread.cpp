//  Copyright (c) 2020 Weile Wei
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/assert.hpp>
#include <hpx/hpx_init.hpp>

#include <hpx/libcds/hpx_tls_manager.hpp>
#include <hpx/modules/testing.hpp>
//
#include <cds/container/fcpriority_queue.h>
#include <cds/init.h>    // for cds::Initialize and cds::Terminate

#include <algorithm>
#include <cstddef>
#include <deque>
#include <functional>
#include <iterator>
#include <random>
#include <vector>

static const std::size_t vec_test_size = 10000;

template <class PQueue>
void run(PQueue& pq, std::size_t i, std::vector<std::size_t>& nums)
{
    // generate a list of numbers in shuffled order
    nums.resize(vec_test_size, 0);
    std::iota(nums.begin(), nums.end(), i);
    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(nums), std::end(nums), rng);

    // insert random numbers into queue
    for (auto i : nums)
    {
        pq.push(i);
    }
}

// this test does not start the HPX runtime
int main(int argc, char* argv[])
{
    // Initialize libcds
    hpx::cds::libcds_wrapper cds_init_wrapper(hpx::cds::smr_t::rcu);

    {
        // Create flat-combining priority queue object
        using fc_pqueue_type = cds::container::FCPriorityQueue<std::size_t,
            std::priority_queue<std::size_t, std::deque<std::size_t>,
                std::greater<std::size_t>>>;

        fc_pqueue_type pq;

        // max threads we are allowed using one per core
        auto maxThreads = std::thread::hardware_concurrency();

        //  a vector of threads
        std::vector<std::thread> threads;
        //  a vector of vectors to store numbers (one number list per thread)
        std::vector<std::vector<std::size_t>> numbers_vec(maxThreads);

        std::size_t i = 0;
        for (auto& v : numbers_vec)
        {
            threads.emplace_back(
                std::thread(run<fc_pqueue_type>, std::ref(pq), i, std::ref(v)));
            i += vec_test_size;
        }

        // wait for all threads to complete
        for (auto& t : threads)
        {
            if (t.joinable())
                t.join();
        }

        // for each thread we generated, collect all the numbers into one huge list
        // then sort and check that the queue has the numbers in the right order
        std::vector<std::size_t> final_vector;
        for (auto& v : numbers_vec)
        {
            final_vector.insert(final_vector.end(), v.begin(), v.end());
        }
        std::sort(
            final_vector.begin(), final_vector.end(), std::less<std::size_t>());

        HPX_TEST_EQ(final_vector.size(), vec_test_size * maxThreads);
        HPX_TEST_EQ(pq.size(), vec_test_size * maxThreads);

        std::size_t vec_val, queue_val;
        // valid if fc priority queue sorts elements in order
        for (unsigned int i = 0; i < final_vector.size(); ++i)
        {
            vec_val = final_vector[i];
            pq.pop(queue_val);
            HPX_TEST_EQ(vec_val, queue_val);
        }
    }

    return hpx::util::report_errors();
}
