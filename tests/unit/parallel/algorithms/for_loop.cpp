//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <hpx/include/parallel_for_loop.hpp>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

template <typename ExPolicy, typename IteratorTag>
void test_for_loop(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    hpx::parallel::for_loop(
        std::forward<ExPolicy>(policy),
        iterator(std::begin(c)), iterator(std::end(c)),
        [](iterator it)
        {
            *it = 42;
        });

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_async(ExPolicy && p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    auto f =
        hpx::parallel::for_loop(
            std::forward<ExPolicy>(p),
            iterator(std::begin(c)), iterator(std::end(c)),
            [](iterator it)
            {
                *it = 42;
            });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename IteratorTag>
void test_for_loop()
{
    using namespace hpx::parallel;

    test_for_loop(execution::seq, IteratorTag());
    test_for_loop(execution::par, IteratorTag());
    test_for_loop(execution::par_unseq, IteratorTag());

    test_for_loop_async(execution::seq(execution::task), IteratorTag());
    test_for_loop_async(execution::par(execution::task), IteratorTag());
}

void for_loop_test()
{
    test_for_loop<std::random_access_iterator_tag>();
    test_for_loop<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_for_loop<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_for_loop_idx(ExPolicy && policy)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    hpx::parallel::for_loop(
        std::forward<ExPolicy>(policy),
        0, c.size(),
        [&c](std::size_t i)
        {
            c[i] = 42;
        });

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy>
void test_for_loop_idx_async(ExPolicy && p)
{
    typedef std::vector<std::size_t>::iterator base_iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    auto f =
        hpx::parallel::for_loop(
            std::forward<ExPolicy>(p),
            0, c.size(),
            [&c](std::size_t i)
            {
                c[i] = 42;
            });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(std::begin(c), std::end(c),
        [&count](std::size_t v) -> void
        {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

void for_loop_test_idx()
{
    using namespace hpx::parallel;

    test_for_loop_idx(execution::seq);
    test_for_loop_idx(execution::par);
    test_for_loop_idx(execution::par_unseq);

    test_for_loop_idx_async(execution::seq(execution::task));
    test_for_loop_idx_async(execution::par(execution::task));
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    for_loop_test();
    for_loop_test_idx();

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
