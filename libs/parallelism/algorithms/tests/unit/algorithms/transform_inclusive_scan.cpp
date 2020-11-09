//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_transform_scan.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_inclusive_scan1(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };
    auto conv = [](std::size_t val) { return 2 * val; };

    hpx::parallel::transform_inclusive_scan(policy, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(d), op, conv, val);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_transform_inclusive_scan(
        std::begin(c), std::end(c), std::begin(e), conv, val, op);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(HPX_HAVE_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::transform_inclusive_scan(
        std::begin(c), std::end(c), std::begin(f), op, conv, val);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_inclusive_scan1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };
    auto conv = [](std::size_t val) { return 2 * val; };

    hpx::future<void> fut =
        hpx::parallel::transform_inclusive_scan(p, iterator(std::begin(c)),
            iterator(std::end(c)), std::begin(d), op, conv, val);
    fut.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_transform_inclusive_scan(
        std::begin(c), std::end(c), std::begin(e), conv, val, op);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(HPX_HAVE_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::transform_inclusive_scan(
        std::begin(c), std::end(c), std::begin(f), op, conv, val);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

template <typename IteratorTag>
void test_transform_inclusive_scan1()
{
    using namespace hpx::execution;

    test_transform_inclusive_scan1(seq, IteratorTag());
    test_transform_inclusive_scan1(par, IteratorTag());
    test_transform_inclusive_scan1(par_unseq, IteratorTag());

    test_transform_inclusive_scan1_async(seq(task), IteratorTag());
    test_transform_inclusive_scan1_async(par(task), IteratorTag());
}

void transform_inclusive_scan_test1()
{
    test_transform_inclusive_scan1<std::random_access_iterator_tag>();
    test_transform_inclusive_scan1<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_inclusive_scan2(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };
    auto conv = [](std::size_t val) { return 2 * val; };

    hpx::parallel::transform_inclusive_scan(policy, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(d), op, conv);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_transform_inclusive_scan_noinit(
        std::begin(c), std::end(c), std::begin(e), conv, op);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(HPX_HAVE_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::transform_inclusive_scan(
        std::begin(c), std::end(c), std::begin(f), op, conv);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_inclusive_scan2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    auto op = [](std::size_t v1, std::size_t v2) { return v1 + v2; };
    auto conv = [](std::size_t val) { return 2 * val; };

    hpx::future<void> fut =
        hpx::parallel::transform_inclusive_scan(p, iterator(std::begin(c)),
            iterator(std::end(c)), std::begin(d), op, conv);
    fut.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_transform_inclusive_scan_noinit(
        std::begin(c), std::end(c), std::begin(e), conv, op);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(e)));

#if defined(HPX_HAVE_CXX17_STD_TRANSFORM_SCAN_ALGORITHMS)
    std::vector<std::size_t> f(c.size());
    std::transform_inclusive_scan(
        std::begin(c), std::end(c), std::begin(f), op, conv);

    HPX_TEST(std::equal(std::begin(d), std::end(d), std::begin(f)));
#endif
}

template <typename IteratorTag>
void test_transform_inclusive_scan2()
{
    using namespace hpx::execution;

    test_transform_inclusive_scan2(seq, IteratorTag());
    test_transform_inclusive_scan2(par, IteratorTag());
    test_transform_inclusive_scan2(par_unseq, IteratorTag());

    test_transform_inclusive_scan2_async(seq(task), IteratorTag());
    test_transform_inclusive_scan2_async(par(task), IteratorTag());
}

void transform_inclusive_scan_test2()
{
    test_transform_inclusive_scan2<std::random_access_iterator_tag>();
    test_transform_inclusive_scan2<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_inclusive_scan_exception(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    bool caught_exception = false;
    try
    {
        hpx::parallel::transform_inclusive_scan(
            policy, iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(d),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
            },
            [](std::size_t val) { return val; }, std::size_t(0));

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
void test_transform_inclusive_scan_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<void> f = hpx::parallel::transform_inclusive_scan(
            p, iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
            },
            [](std::size_t val) { return val; }, std::size_t(0));

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

template <typename IteratorTag>
void test_transform_inclusive_scan_exception()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_inclusive_scan_exception(seq, IteratorTag());
    test_transform_inclusive_scan_exception(par, IteratorTag());

    test_transform_inclusive_scan_exception_async(seq(task), IteratorTag());
    test_transform_inclusive_scan_exception_async(par(task), IteratorTag());
}

void transform_inclusive_scan_exception_test()
{
    test_transform_inclusive_scan_exception<std::random_access_iterator_tag>();
    test_transform_inclusive_scan_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_inclusive_scan_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    bool caught_exception = false;
    try
    {
        hpx::parallel::transform_inclusive_scan(
            policy, iterator(std::begin(c)), iterator(std::end(c)),
            std::begin(d),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
            },
            [](std::size_t val) { return val; }, std::size_t(0));

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
void test_transform_inclusive_scan_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(std::begin(c), std::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<void> f = hpx::parallel::transform_inclusive_scan(
            p, iterator(std::begin(c)), iterator(std::end(c)), std::begin(d),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
            },
            [](std::size_t val) { return val; }, std::size_t(0));

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

template <typename IteratorTag>
void test_transform_inclusive_scan_bad_alloc()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_inclusive_scan_bad_alloc(seq, IteratorTag());
    test_transform_inclusive_scan_bad_alloc(par, IteratorTag());

    test_transform_inclusive_scan_bad_alloc_async(seq(task), IteratorTag());
    test_transform_inclusive_scan_bad_alloc_async(par(task), IteratorTag());
}

void transform_inclusive_scan_bad_alloc_test()
{
    test_transform_inclusive_scan_bad_alloc<std::random_access_iterator_tag>();
    test_transform_inclusive_scan_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_inclusive_scan_test1();
    transform_inclusive_scan_test2();

    transform_inclusive_scan_exception_test();
    transform_inclusive_scan_bad_alloc_test();

    return hpx::finalize();
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
    hpx::init_params init_args;
    init_args.desc_cmdline = desc_commandline;
    init_args.cfg = cfg;

    HPX_TEST_EQ_MSG(hpx::init(argc, argv, init_args), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
