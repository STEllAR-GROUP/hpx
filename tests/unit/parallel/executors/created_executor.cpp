//  Copyright (c) 2015 Daniel Bourgeois
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/deferred_call.hpp>

#include <iostream>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iterator>

using hpx::parallel::parallel_executor;
using hpx::util::deferred_call
using iter = std::vector<int>::iterator;
using std::begin;
using std::end;

////////////////////////////////////////////////////////////////////////////////
// A parallel executor that returns void for bulk_execute and hpx::future<void>
// for bulk_async_execute
struct void_parallel_executor
    : public parallel_executor
{
    void_parallel_executor() {}

    template <typename F, typename Shape>
    static hpx::future<void> bulk_async_execute(F && f, Shape const& shape)
    {
        std::vector<hpx::future<void> > results;
        for(auto const& elem: shape)
        {
            results.push_back(
                parallel_executor::async_execute(deferred_call(f, elem))
            );
        }
        return hpx::when_all(results);
    }

    template <typename F, typename Shape>
    static void bulk_execute(F && f, Shape const& shape)
    {
        return hpx::util::unwrapped(
            bulk_async_execute(std::forward<F>(f), shape));
    }
};

////////////////////////////////////////////////////////////////////////////////
// Tests to void_parallel_executor behavior for the bulk executes

void bulk_test(hpx::thread::id tid, int value)
{
    HPX_TEST(tid != hpx::this_thread::get_id());
}

void test_void_bulk_sync()
{
    typedef void_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;

    executor exec;
    traits::execute(exec, hpx::util::bind(&bulk_test, tid, _1), v);
}

void test_void_bulk_async()
{
    typedef void_parallel_executor executor;
    typedef hpx::parallel::executor_traits<executor> traits;

    hpx::thread::id tid = hpx::this_thread::get_id();

    std::vector<int> v(107);
    std::iota(boost::begin(v), boost::end(v), std::rand());

    using hpx::util::placeholders::_1;

    executor exec;
    traits::async_execute(exec, hpx::util::bind(&bulk_test, tid, _1), v).get();
}

////////////////////////////////////////////////////////////////////////////////
// Sum using hpx's parallel_executor and the above void_parallel_executor

struct range
{
    range(iter first, iter last): first_(first), last_(last) {}
    iter begin() const { return first_; }
    iter end() const { return last_; }
private:
    iter first_;
    iter last_;
};

// Create shape argument for parallel_executor
std::vector<range> split(iter first, iter last, int parts)
{
    typedef typename std::iterator_traits<iter>::difference_type sz_type;
    sz_type count = std::distance(first, last);
    sz_type increment = count/parts;
    std::vector<range> results;
    while(first != last)
    {
        iter prev = first;
        std::advance(
            first,
            (std::min)(increment, std::distance(first,last))
        );
        results.push_back(range(prev, first));
    }
    return std::move(results);
}

// parallel sum using hpx's parallel executor
int parallel_sum(iter first, iter last, int num_parts)
{
    parallel_executor exec;
    typedef hpx::parallel::executor_traits<parallel_executor> traits;

    std::vector<hpx::future<int> > v =
        traits::async_execute(exec, [](const range& rng)
        {
            return std::accumulate(begin(rng), end(rng), 0);
        }, split(first, last, num_parts));

    return std::accumulate(begin(v), end(v), 0,
        [](int a, hpx::future<int>& b)
        {
            return a + b.get();
        });
}

// parallel sum using void parallel executer
int void_parallel_sum(iter first, iter last, int num_parts)
{
    void_parallel_executor exec;
    typedef hpx::parallel::executor_traits<void_parallel_executor> traits;

    std::vector<int> temp(num_parts + 1, 0);
    std::iota(begin(temp), end(temp), 0);

    int section_size = std::distance(first,last)/num_parts;

    hpx::future<void> f = traits::async_execute(exec, [&](const int& i)
    {
        iter b = first + i*section_size;
        iter e = first + (std::min)(int(std::distance(first, last)), (i+1)*section_size);
        temp[i] = std::accumulate(b, e, 0);
    }, temp);

    f.get();

    return std::accumulate(begin(temp), end(temp), 0);
}

void sum_test()
{
    std::vector<int> vec(10007);
    auto random_num = [](){ return std::rand() % 50 - 25; };
    std::generate(begin(vec), end(vec), random_num);

    int sum = std::accumulate(begin(vec), end(vec), 0);
    int num_parts = std::rand() % 5 + 3;

    // Return futures holding results of parallel_sum and void_parallel_sum
    parallel_executor exec;
    hpx::future<int> f_par = exec.async_execute(deferred_call(
        parallel_sum, begin(vec), end(vec), num_parts));
    hpx::future<int> f_void_par = exec.async_execute(deferred_call(
        void_parallel_sum, begin(vec), end(vec), num_parts));

    HPX_TEST(f_par.get() == sum);
    HPX_TEST(f_void_par.get() == sum);
}

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
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
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        boost::lexical_cast<std::string>(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
