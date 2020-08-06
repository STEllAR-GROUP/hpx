//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_equal.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);

template <typename IteratorTag>
void test_equal_binary1(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    {
        bool result = hpx::ranges::equal(
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)));

        bool expected =
            std::equal(std::begin(c1), std::end(c1), std::begin(c2));

        // verify values
        HPX_TEST_EQ(result, expected);
    }

    {
        std::uniform_int_distribution<> dis(0, c1.size() - 1);
        c1[dis(gen)] += 1;    //-V104
        bool result = hpx::ranges::equal(
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)));

        bool expected =
            std::equal(std::begin(c1), std::end(c1), std::begin(c2));

        // verify values
        HPX_TEST_EQ(result, expected);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_equal_binary1(ExPolicy&& policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    {
        bool result = hpx::ranges::equal(policy,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)));

        bool expected =
            std::equal(std::begin(c1), std::end(c1), std::begin(c2));

        // verify values
        HPX_TEST_EQ(result, expected);
    }

    {
        std::uniform_int_distribution<> dis(0, c1.size() - 1);
        c1[dis(gen)] += 1;    //-V104
        bool result = hpx::ranges::equal(policy,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)));

        bool expected =
            std::equal(std::begin(c1), std::end(c1), std::begin(c2));

        // verify values
        HPX_TEST_EQ(result, expected);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_equal_binary1_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    {
        hpx::future<bool> result = hpx::ranges::equal(p,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)));
        result.wait();

        bool expected =
            std::equal(std::begin(c1), std::end(c1), std::begin(c2));

        // verify values
        HPX_TEST_EQ(result.get(), expected);
    }

    {
        std::uniform_int_distribution<> dis(0, c1.size() - 1);
        ++c1[dis(gen)];    //-V104

        hpx::future<bool> result = hpx::ranges::equal(p,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)));
        result.wait();

        bool expected =
            std::equal(std::begin(c1), std::end(c1), std::begin(c2));

        // verify values
        HPX_TEST_EQ(result.get(), expected);
    }
}

template <typename IteratorTag>
void test_equal_binary1()
{
    using namespace hpx::parallel;

    test_equal_binary1(IteratorTag());

    test_equal_binary1(execution::seq, IteratorTag());
    test_equal_binary1(execution::par, IteratorTag());
    test_equal_binary1(execution::par_unseq, IteratorTag());

    test_equal_binary1_async(execution::seq(execution::task), IteratorTag());
    test_equal_binary1_async(execution::par(execution::task), IteratorTag());
}

void equal_binary_test1()
{
    test_equal_binary1<std::random_access_iterator_tag>();
    test_equal_binary1<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_equal_binary2(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    {
        bool result = hpx::ranges::equal(
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            std::equal_to<std::size_t>());

        bool expected = std::equal(std::begin(c1), std::end(c1), std::begin(c2),
            std::equal_to<std::size_t>());

        // verify values
        HPX_TEST_EQ(result, expected);
    }

    {
        std::uniform_int_distribution<> dis(0, c1.size() - 1);
        ++c1[dis(gen)];    //-V104
        bool result = hpx::ranges::equal(
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            std::equal_to<std::size_t>());

        bool expected = std::equal(std::begin(c1), std::end(c1), std::begin(c2),
            std::equal_to<std::size_t>());

        // verify values
        HPX_TEST_EQ(result, expected);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_equal_binary2(ExPolicy&& policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    {
        bool result = hpx::ranges::equal(policy,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            std::equal_to<std::size_t>());

        bool expected = std::equal(std::begin(c1), std::end(c1), std::begin(c2),
            std::equal_to<std::size_t>());

        // verify values
        HPX_TEST_EQ(result, expected);
    }

    {
        std::uniform_int_distribution<> dis(0, c1.size() - 1);
        ++c1[dis(gen)];    //-V104
        bool result = hpx::ranges::equal(policy,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            std::equal_to<std::size_t>());

        bool expected = std::equal(std::begin(c1), std::end(c1), std::begin(c2),
            std::equal_to<std::size_t>());

        // verify values
        HPX_TEST_EQ(result, expected);
    }
}

template <typename ExPolicy, typename IteratorTag>
void test_equal_binary2_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    {
        hpx::future<bool> result = hpx::ranges::equal(p,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            std::equal_to<std::size_t>());
        result.wait();

        bool expected = std::equal(std::begin(c1), std::end(c1), std::begin(c2),
            std::equal_to<std::size_t>());

        // verify values
        HPX_TEST_EQ(result.get(), expected);
    }

    {
        std::uniform_int_distribution<> dis(0, c1.size() - 1);
        ++c1[dis(gen)];    //-V104

        hpx::future<bool> result = hpx::ranges::equal(p,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            std::equal_to<std::size_t>());
        result.wait();

        bool expected = std::equal(std::begin(c1), std::end(c1), std::begin(c2),
            std::equal_to<std::size_t>());

        // verify values
        HPX_TEST_EQ(result.get(), expected);
    }
}

template <typename IteratorTag>
void test_equal_binary2()
{
    using namespace hpx::parallel;

    test_equal_binary2(IteratorTag());

    test_equal_binary2(execution::seq, IteratorTag());
    test_equal_binary2(execution::par, IteratorTag());
    test_equal_binary2(execution::par_unseq, IteratorTag());

    test_equal_binary2_async(execution::seq(execution::task), IteratorTag());
    test_equal_binary2_async(execution::par(execution::task), IteratorTag());
}

void equal_binary_test2()
{
    test_equal_binary2<std::random_access_iterator_tag>();
    test_equal_binary2<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_equal_binary_exception(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_exception = false;
    try
    {
        hpx::ranges::equal(
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), true;
            });

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<hpx::parallel::execution::sequenced_policy,
            IteratorTag>::call(hpx::parallel::execution::seq, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_equal_binary_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_exception = false;
    try
    {
        hpx::ranges::equal(policy,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), true;
            });

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
void test_equal_binary_exception_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::equal(p,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), true;
            });
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
void test_equal_binary_exception()
{
    using namespace hpx::parallel;

    test_equal_binary_exception(IteratorTag());

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_equal_binary_exception(execution::seq, IteratorTag());
    test_equal_binary_exception(execution::par, IteratorTag());

    test_equal_binary_exception_async(
        execution::seq(execution::task), IteratorTag());
    test_equal_binary_exception_async(
        execution::par(execution::task), IteratorTag());
}

void equal_binary_exception_test()
{
    test_equal_binary_exception<std::random_access_iterator_tag>();
    test_equal_binary_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_equal_binary_bad_alloc(ExPolicy&& policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_bad_alloc = false;
    try
    {
        hpx::ranges::equal(policy,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), true;
            });

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_equal_binary_bad_alloc_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef hpx::util::iterator_range<base_iterator> base_range;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;
    typedef hpx::util::iterator_range<iterator> range;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<bool> f = hpx::ranges::equal(p,
            range(iterator(std::begin(c1)), iterator(std::end(c1))),
            base_range(std::begin(c2), std::end(c2)),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), true;
            });
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch (std::bad_alloc const&)
    {
        caught_bad_alloc = true;
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_equal_binary_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_equal_binary_bad_alloc(execution::seq, IteratorTag());
    test_equal_binary_bad_alloc(execution::par, IteratorTag());

    test_equal_binary_bad_alloc_async(
        execution::seq(execution::task), IteratorTag());
    test_equal_binary_bad_alloc_async(
        execution::par(execution::task), IteratorTag());
}

void equal_binary_bad_alloc_test()
{
    test_equal_binary_bad_alloc<std::random_access_iterator_tag>();
    test_equal_binary_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    equal_binary_test1();
    equal_binary_test2();
    equal_binary_exception_test();
    equal_binary_bad_alloc_test();
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
    HPX_TEST_EQ_MSG(hpx::init(desc_commandline, argc, argv, cfg), 0,
        "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
