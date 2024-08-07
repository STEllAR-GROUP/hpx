//  Copyright (c) 2015 Daniel Bourgeois
//  Copyright (c) 2022-2024 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/execution.hpp>
#include <hpx/functional.hpp>
#include <hpx/future.hpp>
#include <hpx/init.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using hpx::util::deferred_call;
using iter = std::vector<int>::iterator;

////////////////////////////////////////////////////////////////////////////////
// A parallel executor that returns void for bulk_execute and hpx::future<void>
// for bulk_async_execute
struct void_parallel_executor : hpx::execution::parallel_executor
{
    using base_type = hpx::execution::parallel_executor;

    template <typename F, typename Shape, typename... Ts>
    friend auto tag_invoke(hpx::parallel::execution::bulk_async_execute_t,
        void_parallel_executor const& exec, F&& f, Shape const& shape,
        Ts&&... ts)
    {
        std::vector<hpx::future<void>> results;
        for (auto const& elem : shape)
        {
            results.push_back(hpx::parallel::execution::async_execute(
                static_cast<base_type const&>(exec), f, elem, ts...));
        }
        return results;
    }

    template <typename F, typename Shape, typename... Ts>
    friend auto tag_invoke(hpx::parallel::execution::bulk_sync_execute_t,
        void_parallel_executor const& exec, F&& f, Shape const& shape,
        Ts&&... ts)
    {
        return hpx::unwrap(hpx::parallel::execution::bulk_async_execute(
            exec, std::forward<F>(f), shape, std::forward<Ts>(ts)...));
    }
};

namespace hpx::parallel::execution {
    template <>
    struct is_bulk_one_way_executor<void_parallel_executor> : std::true_type
    {
    };

    template <>
    struct is_bulk_two_way_executor<void_parallel_executor> : std::true_type
    {
    };
}    // namespace hpx::parallel::execution

////////////////////////////////////////////////////////////////////////////////
// Tests to void_parallel_executor behavior for the bulk executes

template <typename Executor>
decltype(auto) disable_run_as_child(Executor&& exec)
{
    auto hint = hpx::execution::experimental::get_hint(exec);
    hint.runs_as_child_mode(hpx::threads::thread_execution_hint::none);

    return hpx::experimental::prefer(hpx::execution::experimental::with_hint,
        HPX_FORWARD(Executor, exec), hint);
}

void bulk_test(int, hpx::thread::id const& tid, int passed_through)    //-V813
{
    HPX_TEST_NEQ(tid, hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

void test_void_bulk_sync()
{
    using executor = void_parallel_executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    executor exec;
    hpx::parallel::execution::bulk_sync_execute(
        disable_run_as_child(exec), hpx::bind(&bulk_test, _1, tid, _2), v, 42);
    hpx::parallel::execution::bulk_sync_execute(
        disable_run_as_child(exec), &bulk_test, v, tid, 42);
}

void test_void_bulk_async()
{
    using executor = void_parallel_executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::placeholders::_1;
    using hpx::placeholders::_2;

    executor exec;
    hpx::when_all(
        hpx::parallel::execution::bulk_async_execute(disable_run_as_child(exec),
            hpx::bind(&bulk_test, _1, tid, _2), v, 42))
        .get();
    hpx::when_all(hpx::parallel::execution::bulk_async_execute(
                      disable_run_as_child(exec), &bulk_test, v, tid, 42))
        .get();
}

////////////////////////////////////////////////////////////////////////////////
// Sum using hpx's parallel_executor and the above void_parallel_executor

// Create shape argument for parallel_executor
std::vector<hpx::util::iterator_range<iter>> split(
    iter first, iter const& last, int parts)
{
    using sz_type = std::iterator_traits<iter>::difference_type;
    sz_type const count = std::distance(first, last);
    sz_type const increment = count / parts;

    std::vector<hpx::util::iterator_range<iter>> results;
    while (first != last)
    {
        iter const prev = first;
        std::advance(first, (std::min)(increment, std::distance(first, last)));
        results.push_back(hpx::util::iterator_range(prev, first));
    }
    return results;
}

// parallel sum using hpx's parallel executor
int parallel_sum(iter const& first, iter const& last, int num_parts)
{
    hpx::execution::parallel_executor exec;

    std::vector<hpx::util::iterator_range<iter>> input =
        split(first, last, num_parts);

    std::vector<hpx::future<int>> v =
        hpx::parallel::execution::bulk_async_execute(
            exec,
            [](hpx::util::iterator_range<iter> const& rng) -> int {
                return std::accumulate(std::begin(rng), std::end(rng), 0);
            },
            input);

    return std::accumulate(std::begin(v), std::end(v), 0,
        [](int a, hpx::future<int>& b) -> int { return a + b.get(); });
}

// parallel sum using void parallel executor
int void_parallel_sum(iter const& first, iter const& last, int num_parts)
{
    void_parallel_executor exec;

    std::vector<int> temp(num_parts + 1, 0);
    std::iota(std::begin(temp), std::end(temp), 0);

    std::ptrdiff_t const section_size = std::distance(first, last) / num_parts;

    std::vector<hpx::future<void>> f =
        hpx::parallel::execution::bulk_async_execute(
            exec,
            [&](const int& i) {
                iter const b = first + i * section_size;    //-V104
                iter const e = first +
                    (std::min)(std::distance(first, last),
                        static_cast<std::ptrdiff_t>(
                            (i + 1) * section_size)    //-V104
                    );
                temp[i] = std::accumulate(b, e, 0);    //-V108
            },
            temp);

    hpx::when_all(f).get();

    return std::accumulate(std::begin(temp), std::end(temp), 0);
}

void sum_test()
{
    std::vector<int> vec(10007);
    auto random_num = []() { return std::rand() % 50 - 25; };
    std::generate(std::begin(vec), std::end(vec), random_num);

    int const sum = std::accumulate(std::begin(vec), std::end(vec), 0);
    int num_parts = std::rand() % 5 + 3;

    // Return futures holding results of parallel_sum and void_parallel_sum
    hpx::execution::parallel_executor exec;

    hpx::future<int> f_par = hpx::parallel::execution::async_execute(
        exec, &parallel_sum, std::begin(vec), std::end(vec), num_parts);

    hpx::future<int> f_void_par = hpx::parallel::execution::async_execute(
        exec, &void_parallel_sum, std::begin(vec), std::end(vec), num_parts);

    HPX_TEST_EQ(f_par.get(), sum);
    HPX_TEST_EQ(f_void_par.get(), sum);
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_void_bulk_sync();
    test_void_bulk_async();
    sum_test();
    return hpx::local::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace hpx::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run");

    // By default, this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
