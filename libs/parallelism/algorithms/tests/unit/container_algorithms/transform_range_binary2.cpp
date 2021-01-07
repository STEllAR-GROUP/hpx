//  Copyright (c) 2014-2017 Hartmut Kaiser
//  Copyright (c) 2021 Giannis Gonidelis
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_transform.hpp>
#include <hpx/iterator_support/iterator_range.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_binary(IteratorTag)
{
    typedef test::test_container<std::vector<int>, IteratorTag> test_vector;

    test_vector c1(10007);
    test_vector c2(10007);
    std::vector<std::size_t> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    auto add = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    auto result = hpx::ranges::transform(c1, c2, std::begin(d1), add);

    HPX_TEST(result.in1 == std::end(c1));
    HPX_TEST(result.in2 == std::end(c2));
    HPX_TEST(result.out == std::end(d1));

    // verify values
    std::vector<std::size_t> d2(c1.size());
    std::transform(
        std::begin(c1), std::end(c1), std::begin(c2), std::begin(d2), add);

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef test::test_container<std::vector<int>, IteratorTag> test_vector;

    test_vector c1(10007);
    test_vector c2(10007);
    std::vector<std::size_t> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    auto add = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    auto result = hpx::ranges::transform(policy, c1, c2, std::begin(d1), add);

    HPX_TEST(result.in1 == std::end(c1));
    HPX_TEST(result.in2 == std::end(c2));
    HPX_TEST(result.out == std::end(d1));

    // verify values
    std::vector<std::size_t> d2(c1.size());
    std::transform(
        std::begin(c1), std::end(c1), std::begin(c2), std::begin(d2), add);

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_async(ExPolicy p, IteratorTag)
{
    typedef test::test_container<std::vector<int>, IteratorTag> test_vector;

    test_vector c1(10007);
    test_vector c2(10007);
    std::vector<std::size_t> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    auto add = [](std::size_t v1, std::size_t v2) { return v1 + v2; };

    auto f = hpx::ranges::transform(p, c1, c2, std::begin(d1), add);
    f.wait();

    auto result = f.get();
    HPX_TEST(result.in1 == std::end(c1));
    HPX_TEST(result.in2 == std::end(c2));
    HPX_TEST(result.out == std::end(d1));

    // verify values
    std::vector<std::size_t> d2(c1.size());
    std::transform(
        std::begin(c1), std::end(c1), std::begin(c2), std::begin(d2), add);

    std::size_t count = 0;
    HPX_TEST(std::equal(std::begin(d1), std::end(d1), std::begin(d2),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

template <typename IteratorTag>
void test_transform_binary()
{
    using namespace hpx::execution;

    test_transform_binary(IteratorTag());
    test_transform_binary(seq, IteratorTag());
    test_transform_binary(par, IteratorTag());
    test_transform_binary(par_unseq, IteratorTag());

    test_transform_binary_async(seq(task), IteratorTag());
    test_transform_binary_async(par(task), IteratorTag());
}

void transform_binary_test()
{
    test_transform_binary<std::random_access_iterator_tag>();
    test_transform_binary<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_transform_binary_exception(IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_exception = false;
    try
    {
        hpx::ranges::transform(
            hpx::util::make_iterator_range(
                iterator(std::begin(c1)), iterator(std::end(c1))),
            c2, std::begin(d1), [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
            });

        HPX_TEST(false);
    }
    catch (hpx::exception_list const& e)
    {
        caught_exception = true;
        test::test_num_exceptions<hpx::execution::sequenced_policy,
            IteratorTag>::call(hpx::execution::seq, e);
    }
    catch (...)
    {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_exception(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_exception = false;
    try
    {
        hpx::ranges::transform(policy,
            hpx::util::make_iterator_range(
                iterator(std::begin(c1)), iterator(std::end(c1))),
            hpx::util::make_iterator_range(std::begin(c2), std::end(c2)),
            std::begin(d1), [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
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
void test_transform_binary_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::ranges::transform(p,
            hpx::util::make_iterator_range(
                iterator(std::begin(c1)), iterator(std::end(c1))),
            hpx::util::make_iterator_range(std::begin(c2), std::end(c2)),
            std::begin(d1), [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
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
void test_transform_binary_exception()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_binary_exception(IteratorTag());
    test_transform_binary_exception(seq, IteratorTag());
    test_transform_binary_exception(par, IteratorTag());

    test_transform_binary_exception_async(seq(task), IteratorTag());
    test_transform_binary_exception_async(par(task), IteratorTag());
}

void transform_binary_exception_test()
{
    test_transform_binary_exception<std::random_access_iterator_tag>();
    test_transform_binary_exception<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_bad_alloc = false;
    try
    {
        hpx::ranges::transform(policy,
            hpx::util::make_iterator_range(
                iterator(std::begin(c1)), iterator(std::end(c1))),
            hpx::util::make_iterator_range(std::begin(c2), std::end(c2)),
            std::begin(d1), [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
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
void test_transform_binary_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());
    std::vector<std::size_t> d1(c1.size());    //-V656
    std::iota(std::begin(c1), std::end(c1), std::rand());
    std::iota(std::begin(c2), std::end(c2), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        auto f = hpx::ranges::transform(p,
            hpx::util::make_iterator_range(
                iterator(std::begin(c1)), iterator(std::end(c1))),
            hpx::util::make_iterator_range(std::begin(c2), std::end(c2)),
            std::begin(d1), [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
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
void test_transform_binary_bad_alloc()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_transform_binary_bad_alloc(seq, IteratorTag());
    test_transform_binary_bad_alloc(par, IteratorTag());

    test_transform_binary_bad_alloc_async(seq(task), IteratorTag());
    test_transform_binary_bad_alloc_async(par(task), IteratorTag());
}

void transform_binary_bad_alloc_test()
{
    test_transform_binary_bad_alloc<std::random_access_iterator_tag>();
    test_transform_binary_bad_alloc<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int) std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    transform_binary_test();
    transform_binary_exception_test();
    transform_binary_bad_alloc_test();
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
