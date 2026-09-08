//  Copyright (c) 2026 The STE||AR-Group
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

// Regression test: concurrent first-use of hostname_print_helper must not
// race or produce inconsistent/garbled output (see call_once guard added
// to print.cpp).

#include <hpx/modules/debugging.hpp>
#include <hpx/modules/testing.hpp>

#include <atomic>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

int main()
{
#if defined(__FreeBSD__)
    freebsd_environ = environ;
#endif

    constexpr int num_threads = 32;

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::vector<std::string> results(num_threads);
    std::atomic<int> ready{0};
    std::atomic<bool> start{false};

    for (int i = 0; i != num_threads; ++i)
    {
        threads.emplace_back([i, &results, &ready, &start]() {
            // Signal that this worker is ready, then spin until every
            // thread has reached the gate so initialization is actually
            // contended rather than serialized by scheduling.
            ready.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire))
            {
                // Yield to reduce busy-wait overhead on CI.
                std::this_thread::yield();
            }

            std::ostringstream os;
            os << hpx::debug::detail::hostname_print_helper();
            results[i] = os.str();
        });
    }

    // Wait for every worker to reach the gate before releasing.
    while (ready.load(std::memory_order_acquire) < num_threads)
    {
        std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);

    for (auto& t : threads)
    {
        t.join();
    }

    // The hostname must be identical and non-empty
    // for every thread; a data race in lazy init could otherwise yield a
    // partially written / truncated buffer for some threads.
    std::string const first = results[0];
    HPX_TEST(!first.empty());

    for (int i = 1; i != num_threads; ++i)
    {
        std::string const current = results[i];
        HPX_TEST_EQ(current, first);
        HPX_TEST(!current.empty());
    }

    return hpx::util::report_errors();
}
