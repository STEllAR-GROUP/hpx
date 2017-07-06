//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/deferred_call.hpp>

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

#include <boost/range/iterator_range.hpp>

using hpx::util::deferred_call;
typedef std::vector<int>::iterator iter;

////////////////////////////////////////////////////////////////////////////////
// A parallel executor that returns void for bulk_execute and hpx::future<void>
// for bulk_async_execute
struct void_parallel_executor
  : hpx::parallel::execution::parallel_executor
{
    template <typename F, typename Shape, typename ... Ts>
    std::vector<hpx::future<void> >
    bulk_async_execute(F && f, Shape const& shape, Ts &&... ts)
    {
        std::vector<hpx::future<void> > results;
        for(auto const& elem: shape)
        {
            results.push_back(this->parallel_executor::async_execute(
                f, elem, ts...));
        }
        return results;
    }

    template <typename F, typename Shape, typename ... Ts>
    void bulk_sync_execute(F && f, Shape const& shape, Ts &&... ts)
    {
        return hpx::util::unwrapped(
            bulk_async_execute(std::forward<F>(f), shape,
                std::forward<Ts>(ts)...));
    }
};

namespace hpx { namespace traits
{
    template <>
    struct is_two_way_executor<void_parallel_executor>
        : std::true_type
    {};

    template <>
    struct is_bulk_two_way_executor<void_parallel_executor>
        : std::true_type
    {};
}}

////////////////////////////////////////////////////////////////////////////////
// Tests to void_parallel_executor behavior for the bulk executes

void bulk_test(int value, hpx::thread::id tid, int passed_through) //-V813
{
    HPX_TEST(tid != hpx::this_thread::get_id());
    HPX_TEST_EQ(passed_through, 42);
}

void test_void_bulk_sync()
{
    typedef void_parallel_executor executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    executor exec;
    hpx::parallel::execution::bulk_sync_execute(
        exec, hpx::util::bind(&bulk_test, _1, tid, _2), v, 42);
    hpx::parallel::execution::bulk_sync_execute(
        exec, &bulk_test, v, tid, 42);
}

void test_void_bulk_async()
{
    typedef void_parallel_executor executor;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(std::begin(v), std::end(v), std::rand());

    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;

    executor exec;
    hpx::when_all(
        hpx::parallel::execution::bulk_async_execute(exec,
            hpx::util::bind(&bulk_test, _1, tid, _2), v, 42)
    ).get();
    hpx::when_all(
        hpx::parallel::execution::bulk_async_execute(exec,
            &bulk_test, v, tid, 42)).get();
}

////////////////////////////////////////////////////////////////////////////////
// Sum using hpx's parallel_executor and the above void_parallel_executor

// Create shape argument for parallel_executor
std::vector<boost::iterator_range<iter> >
split(iter first, iter last, int parts)
{
    typedef std::iterator_traits<iter>::difference_type sz_type;
    sz_type count = std::distance(first, last);
    sz_type increment = count/parts;

    std::vector<boost::iterator_range<iter> > results;
    while(first != last)
    {
        iter prev = first;
        std::advance(
            first,
            (std::min)(increment, std::distance(first,last))
        );
        results.push_back(boost::make_iterator_range(prev, first));
    }
    return results;
}

// parallel sum using hpx's parallel executor
int parallel_sum(iter first, iter last, int num_parts)
{
    hpx::parallel::execution::parallel_executor exec;

    std::vector<boost::iterator_range<iter> > input =
        split(first, last, num_parts);

    std::vector<hpx::future<int> > v =
        hpx::parallel::execution::bulk_async_execute(exec,
            [](boost::iterator_range<iter> const& rng) -> int
            {
                return std::accumulate(std::begin(rng), std::end(rng), 0);
            },
            input);

    return std::accumulate(
        std::begin(v), std::end(v), 0,
        [](int a, hpx::future<int>& b) -> int
        {
            return a + b.get();
        });
}

// parallel sum using void parallel executer
int void_parallel_sum(iter first, iter last, int num_parts)
{
    void_parallel_executor exec;

    std::vector<int> temp(num_parts + 1, 0);
    std::iota(std::begin(temp), std::end(temp), 0);

    std::ptrdiff_t section_size = std::distance(first, last) / num_parts;

    std::vector<hpx::future<void> > f =
        hpx::parallel::execution::bulk_async_execute(exec,
            [&](const int& i)
            {
                iter b = first + i*section_size; //-V104
                iter e = first + (std::min)(
                        std::distance(first, last),
                        static_cast<std::ptrdiff_t>((i+1)*section_size) //-V104
                    );
                temp[i] = std::accumulate(b, e, 0); //-V108
            },
            temp);

    hpx::when_all(f).get();

    return std::accumulate(std::begin(temp), std::end(temp), 0);
}

void sum_test()
{
    std::vector<int> vec(10007);
    auto random_num = [](){ return std::rand() % 50 - 25; };
    std::generate(std::begin(vec), std::end(vec), random_num);

    int sum = std::accumulate(std::begin(vec), std::end(vec), 0);
    int num_parts = std::rand() % 5 + 3;

    // Return futures holding results of parallel_sum and void_parallel_sum
    hpx::parallel::execution::parallel_executor exec;

    hpx::future<int> f_par =
        hpx::parallel::execution::async_execute(exec, &parallel_sum,
            std::begin(vec), std::end(vec), num_parts);

    hpx::future<int> f_void_par =
        hpx::parallel::execution::async_execute(exec, &void_parallel_sum,
            std::begin(vec), std::end(vec), num_parts);

    HPX_TEST(f_par.get() == sum);
    HPX_TEST(f_void_par.get() == sum);
}

////////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = static_cast<unsigned int>(std::time(nullptr));
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    test_void_bulk_sync();
    test_void_bulk_async();
    sum_test();
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {
        "hpx.os_threads=all"
    };

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
