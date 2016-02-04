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

#define NUM_FUTURES std::size_t(300)

// One way to force recursion is to make all futures depend on the next and
// make the last of the futures ready, triggering a chain of continuations.
void force_recursion_test1()
{
    hpx::lcos::local::promise<void> first_promise;

    std::vector<hpx::shared_future<void> > results;
    results.reserve(NUM_FUTURES+1);

    boost::atomic<std::size_t> executed_dataflow(0);

    results.push_back(first_promise.get_future());
    for (std::size_t i = 0; i != NUM_FUTURES; ++i)
    {
        results.push_back(
            hpx::dataflow(
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
}

// Another way to force recursion is to make all futures ready from the start
// (continuations will be executed at the point where they are being attached
// to the future), and attach the next continuation from inside a continuation.
// This will trigger a chain of continuations as well.
void make_ready_continue(
    std::size_t i,
    std::vector<hpx::lcos::local::promise<void> >& promises,
    std::vector<hpx::shared_future<void> > & futures,
    boost::atomic<std::size_t> & executed_continuations)
{
    if (i >= NUM_FUTURES)
        return;

    ++executed_continuations;
    promises[i].set_value();
    futures[i].then(
        hpx::util::bind(
            &make_ready_continue, i+1, std::ref(promises), std::ref(futures),
            std::ref(executed_continuations)
        )
    );
}

void force_recursion_test2()
{
    std::vector<hpx::lcos::local::promise<void> > promises;
    promises.reserve(NUM_FUTURES);

    std::vector<hpx::shared_future<void> > futures;
    futures.reserve(NUM_FUTURES);

    for (std::size_t i = 0; i != NUM_FUTURES; ++i)
    {
        promises.push_back(hpx::lcos::local::promise<void>());
        futures.push_back(promises[i].get_future());
    }

    boost::atomic<std::size_t> executed_continuations(0);
    make_ready_continue(0, promises, futures, executed_continuations);

    hpx::wait_all(futures);
    HPX_TEST_EQ(executed_continuations.load(), NUM_FUTURES);
}

int main()
{
    force_recursion_test1();
    force_recursion_test2();

    return hpx::util::report_errors();
}
