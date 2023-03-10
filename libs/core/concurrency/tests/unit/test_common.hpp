//  Copyright (C) 2011 Tim Blechmann
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/algorithms.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/modules/threading.hpp>

#include "test_helpers.hpp"

#include <array>
#include <atomic>
#include <functional>
#include <set>

namespace tests {

    template <bool Bounded = false>
    struct queue_stress_tester
    {
        static constexpr unsigned int buckets = 1 << 13;
        static constexpr long node_count = 10000;

        int const reader_threads;
        int const writer_threads;

        std::atomic<int> writers_finished;

        static_hashed_set<long, buckets> data;
        static_hashed_set<long, buckets> dequeued;
        std::array<std::set<long>, buckets> returned;

        std::atomic<int> push_count, pop_count;

        queue_stress_tester(int reader, int writer)
          : reader_threads(reader)
          , writer_threads(writer)
          , push_count(0)
          , pop_count(0)
        {
        }

        template <typename Queue>
        void add_items(Queue& stk)
        {
            for (long i = 0; i != node_count; ++i)
            {
                long id = generate_id<long>();

                bool inserted = data.insert(id);
                HPX_TEST(inserted);

                if constexpr (Bounded)
                {
                    while (!stk.bounded_push(id))
                    {
                        hpx::this_thread::yield();
                    }
                }
                else
                {
                    while (!stk.push(id))
                    {
                        hpx::this_thread::yield();
                    }
                }
                ++push_count;

                hpx::this_thread::yield();
            }
            writers_finished += 1;
        }

        std::atomic<bool> running;

        template <typename Queue>
        bool consume_element(Queue& q)
        {
            long id;
            bool ret = q.pop(id);

            if (!ret)
                return false;

            bool erased = data.erase(id);
            bool inserted = dequeued.insert(id);
            HPX_TEST(erased);
            HPX_TEST(inserted);

            ++pop_count;
            return true;
        }

        template <typename Queue>
        void get_items(Queue& q)
        {
            for (;;)
            {
                bool received_element = consume_element(q);
                if (received_element)
                    continue;

                if (writers_finished.load() == writer_threads)
                    break;

                hpx::this_thread::yield();
            }

            while (consume_element(q))
                ;
        }

        template <typename Queue>
        void run(Queue& stk)
        {
            writers_finished.store(0);

            hpx::experimental::task_group writer;
            hpx::experimental::task_group reader;

            HPX_TEST(stk.empty());

            for (int i = 0; i != reader_threads; ++i)
            {
                reader.run(&queue_stress_tester::template get_items<Queue>,
                    this, std::ref(stk));
            }

            for (int i = 0; i != writer_threads; ++i)
            {
                writer.run(&queue_stress_tester::template add_items<Queue>,
                    this, std::ref(stk));
            }

            writer.wait();
            reader.wait();

            HPX_TEST_EQ(data.count_nodes(), (size_t) 0);
            HPX_TEST(stk.empty());

            HPX_TEST_EQ(push_count, pop_count);
            HPX_TEST_EQ(push_count, writer_threads * node_count);
        }
    };
}    // namespace tests

using tests::queue_stress_tester;
