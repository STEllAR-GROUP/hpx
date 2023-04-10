//  Copyright (c) 2020-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Parts of this code were inspired by https://github.com/josuttis/jthread. The
//  original code was published by Nicolai Josuttis and Lewis Baker under the
//  Creative Commons Attribution 4.0 International License
//  (http://creativecommons.org/licenses/by/4.0/).

#include <hpx/init.hpp>
#include <hpx/modules/synchronization.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/thread.hpp>

#include <chrono>
#include <mutex>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
void producer_consumer(double prod_sec, double cons_sec, bool interrupt)
{
    // duration producer creates new values
    auto prod_sleep = std::chrono::duration<double, std::milli>{prod_sec};
    // duration producer deals with new values
    auto cons_sleep = std::chrono::duration<double, std::milli>{cons_sec};

    std::vector<int> items;
    hpx::mutex items_mtx;
    hpx::condition_variable_any items_cv;
    hpx::stop_source ssource;
    hpx::stop_token stoken{ssource.get_token()};
    constexpr size_t max_queue_size = 100;

    hpx::thread producer{[&] {
        auto next_value = [val = 0]() mutable { return ++val; };

        std::unique_lock<hpx::mutex> lock{items_mtx};
        while (!stoken.stop_requested())
        {
            if (!items_cv.wait(lock, stoken,
                    [&] { return items.size() < max_queue_size; }))
            {
                return;
            }

            while (items.size() < max_queue_size && !stoken.stop_requested())
            {
                // Ok to produce a value.

                // Don't hold mutex while working to allow consumer to run.
                int item;

                {
                    hpx::unlock_guard<hpx::mutex> ul(items_mtx);
                    item = next_value();
                    if (prod_sec > 0)
                    {
                        hpx::this_thread::sleep_for(prod_sleep);
                    }
                }

                // Enqueue a value
                items.push_back(item);
                items_cv.notify_all();    // notify that items were added
            }
        }
    }};

    hpx::thread consumer{[&] {
        std::unique_lock<hpx::mutex> lock{items_mtx};
        for (;;)
        {
            {
                hpx::unlock_guard<hpx::mutex> ul(items_mtx);
                if (cons_sec > 0)
                {
                    hpx::this_thread::sleep_for(cons_sleep);
                }
            }

            if (!items_cv.wait(lock, stoken, [&] { return !items.empty(); }))
            {
                // returned false, so it was interrupted
                return;
            }

            // process current items (note: items_mtx is locked):
            for (int item : items)
            {
                if (item == 42)
                {
                    // Found the item I'm looking for. Cancel producer.
                    ssource.request_stop();
                    return;
                }
            }
            items.clear();
            items_cv.notify_all();    // notify that items were removed
        }
    }};

    // Let the producer/consumer run for a little while
    // Interrupt if they don't find a result quickly enough.

    if (interrupt)
    {
        if (prod_sec > 0)
        {
            hpx::this_thread::sleep_for(prod_sleep * 10);
        }
        ssource.request_stop();
    }

    consumer.join();
    producer.join();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main()
{
    std::set_terminate([]() { HPX_TEST(false); });

    try
    {
        producer_consumer(0, 0, false);
        producer_consumer(0.1, 0, false);
        producer_consumer(0, 0.1, false);
        producer_consumer(0.1, 0.9, false);
        producer_consumer(0, 5.0, false);
        producer_consumer(0.05, 5.0, false);

        producer_consumer(0, 0, true);
        producer_consumer(0.1, 0, true);
        producer_consumer(0, 0.1, true);
        producer_consumer(0.1, 0.9, true);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
