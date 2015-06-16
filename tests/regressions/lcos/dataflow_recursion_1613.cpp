//  Copyright (c) 2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This test case demonstrates the issue described in #1613: Dataflow causes
// stack overflow

#include <hpx/hpx.hpp>
#include <hpx/hpx_main.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/atomic.hpp>

#define NUM_FUTURES std::size_t(1000)

int main()
{
    hpx::lcos::local::promise<void> first_promise;

    std::vector<hpx::shared_future<void> > results;
    results.reserve(NUM_FUTURES+1);

    boost::atomic<std::size_t> executed_dataflow(0);

    results.push_back(first_promise.get_future());
    for (std::size_t i = 0; i != NUM_FUTURES; ++i)
    {
        results.push_back(
            hpx::lcos::local::dataflow(
                hpx::launch::sync,
                [&](hpx::shared_future<void> &&)
                {
                    ++executed_dataflow;
                },
                results.back()
            )
        );
    }

    // make futures ready in backwards sequence
    hpx::apply([&first_promise]() { first_promise.set_value(); });

    hpx::wait_all(results);
    HPX_TEST_EQ(executed_dataflow.load(), NUM_FUTURES);

    return hpx::util::report_errors();
}
