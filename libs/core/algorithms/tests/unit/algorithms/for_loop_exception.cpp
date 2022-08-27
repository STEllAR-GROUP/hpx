//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/algorithm.hpp>
#include <hpx/local/init.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
struct throw_always
{
    throw_always(std::size_t throw_after)
    {
        throw_after_ = throw_after;
    }

    template <typename T>
    void operator()(T)
    {
        if (--throw_after_ == 0)
            throw std::runtime_error("test");
    }

    static std::atomic<std::size_t> throw_after_;
};

std::atomic<std::size_t> throw_always::throw_after_(0);

struct throw_bad_alloc
{
    throw_bad_alloc(std::size_t throw_after)
    {
        throw_after_ = throw_after;
    }

    template <typename T>
    void operator()(T)
    {
        if (--throw_after_ == 0)
            throw std::bad_alloc();
    }

    static std::atomic<std::size_t> throw_after_;
};

std::atomic<std::size_t> throw_bad_alloc::throw_after_(0);

///////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::uniform_int_distribution<std::size_t> dis(1, c.size());

    bool caught_exception = false;
    try
    {
        hpx::experimental::for_loop(std::forward<ExPolicy>(policy),
            iterator(std::begin(c)), iterator(std::end(c)),
            throw_always(dis(gen)));

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_exception_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::uniform_int_distribution<std::size_t> dis(1, c.size());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::experimental::for_loop(std::forward<ExPolicy>(p),
            iterator(std::begin(c)), iterator(std::end(c)),
            throw_always(dis(gen)));
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_loop_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::uniform_int_distribution<std::size_t> dis(1, c.size());

    bool caught_exception = false;
    try
    {
        hpx::experimental::for_loop(std::forward<ExPolicy>(policy),
            iterator(std::begin(c)), iterator(std::end(c)),
            throw_bad_alloc(dis(gen)));

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::uniform_int_distribution<std::size_t> dis(1, c.size());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::experimental::for_loop(std::forward<ExPolicy>(p),
            iterator(std::begin(c)), iterator(std::end(c)),
            throw_bad_alloc(dis(gen)));
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_for_loop_exception()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_for_loop_exception(seq, IteratorTag());
    test_for_loop_exception(par, IteratorTag());

    test_for_loop_bad_alloc(seq, IteratorTag());
    test_for_loop_bad_alloc(par, IteratorTag());

    test_for_loop_exception_async(seq(task), IteratorTag());
    test_for_loop_exception_async(par(task), IteratorTag());

    test_for_loop_bad_alloc_async(seq(task), IteratorTag());
    test_for_loop_bad_alloc_async(par(task), IteratorTag());
}

void for_loop_exception_test()
{
    test_for_loop_exception<std::random_access_iterator_tag>();
    test_for_loop_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_loop_idx_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::uniform_int_distribution<std::size_t> dis(1, c.size());

    bool caught_exception = false;
    try
    {
        hpx::experimental::for_loop(std::forward<ExPolicy>(policy), 0, c.size(),
            throw_always(dis(gen)));

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_for_loop_idx_exception_async(ExPolicy&& p, IteratorTag)
{
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::uniform_int_distribution<std::size_t> dis(1, c.size());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::experimental::for_loop(
            std::forward<ExPolicy>(p), 0, c.size(), throw_always(dis(gen)));
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy>
void test_for_loop_idx_bad_alloc(ExPolicy&& policy)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::uniform_int_distribution<std::size_t> dis(1, c.size());

    bool caught_exception = false;
    try
    {
        hpx::experimental::for_loop(std::forward<ExPolicy>(policy), 0, c.size(),
            throw_bad_alloc(dis(gen)));

        HPX_TEST(false);
    }
    catch (std::bad_alloc const& e)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy>
void test_for_loop_idx_bad_alloc_async(ExPolicy&& p)
{
    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), gen());

    std::uniform_int_distribution<std::size_t> dis(1, c.size());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::experimental::for_loop(
            std::forward<ExPolicy>(p), 0, c.size(), throw_bad_alloc(dis(gen)));
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (std::bad_alloc const& e)
    {
        caught_exception = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_for_loop_exception_idx()
{
    using namespace hpx::execution;

    test_for_loop_idx_exception(seq, IteratorTag());
    test_for_loop_idx_exception(par, IteratorTag());

    test_for_loop_idx_bad_alloc(seq);
    test_for_loop_idx_bad_alloc(par);

    test_for_loop_idx_exception_async(seq(task), IteratorTag());
    test_for_loop_idx_exception_async(par(task), IteratorTag());

    test_for_loop_idx_bad_alloc_async(seq(task));
    test_for_loop_idx_bad_alloc_async(par(task));
}

void for_loop_exception_test_idx()
{
    test_for_loop_exception_idx<std::random_access_iterator_tag>();
    test_for_loop_exception_idx<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    for_loop_exception_test();
    for_loop_exception_test_idx();

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

    // By default this test should run on all available cores
    std::vector<std::string> const cfg = {"hpx.os_threads=all"};

    // Initialize and run HPX
    hpx::local::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::local::init(hpx_main, argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
