//  Copyright (c) 2014-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/include/datapar.hpp>
#include <hpx/local/init.hpp>

#include <iostream>
#include <string>
#include <vector>

#include "../algorithms/equal_tests.hpp"

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_equal1()
{
    using namespace hpx::execution;

    test_equal1(simd, IteratorTag());
    test_equal1(par_simd, IteratorTag());

    test_equal1_async(simd(task), IteratorTag());
    test_equal1_async(par_simd(task), IteratorTag());
}

void equal_test1()
{
    test_equal1<std::random_access_iterator_tag>();
    test_equal1<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_equal2()
{
    using namespace hpx::execution;

    test_equal2(simd, IteratorTag());
    test_equal2(par_simd, IteratorTag());

    test_equal2_async(simd(task), IteratorTag());
    test_equal2_async(par_simd(task), IteratorTag());
}

void equal_test2()
{
    test_equal2<std::random_access_iterator_tag>();
    test_equal2<std::forward_iterator_tag>();
}

////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_equal_exception_datapar(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_exception = false;
    try
    {
        hpx::equal(policy, iterator(std::begin(c1)), iterator(std::end(c1)),
            std::begin(c2), [](auto a, auto) {
                return throw std::runtime_error("test"), (a == a);
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
void test_equal_exception_async_datapar(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<bool> f = hpx::equal(p, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2), [](auto a, auto) {
                return throw std::runtime_error("test"), (a == a);
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

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_equal_exception_datapar()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_equal_exception_datapar(simd, IteratorTag());
    test_equal_exception_datapar(par_simd, IteratorTag());

    test_equal_exception_async_datapar(simd(task), IteratorTag());
    test_equal_exception_async_datapar(par_simd(task), IteratorTag());
}

void equal_exception_test_datapar()
{
    test_equal_exception_datapar<std::random_access_iterator_tag>();
    test_equal_exception_datapar<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_equal_bad_alloc_datapar(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_bad_alloc = false;
    try
    {
        hpx::equal(policy, iterator(std::begin(c1)), iterator(std::end(c1)),
            std::begin(c2),
            [](auto a, auto) { return throw std::bad_alloc(), (a == a); });

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
void test_equal_bad_alloc_async_datapar(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c1(10007);
    std::vector<std::size_t> c2(c1.size());

    std::size_t first_value = gen();    //-V101
    std::iota(std::begin(c1), std::end(c1), first_value);
    std::iota(std::begin(c2), std::end(c2), first_value);

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        hpx::future<bool> f = hpx::equal(p, iterator(std::begin(c1)),
            iterator(std::end(c1)), std::begin(c2),
            [](auto a, auto) { return throw std::bad_alloc(), (a == a); });
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

////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_equal_bad_alloc_datapar()
{
    using namespace hpx::execution;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_equal_bad_alloc_datapar(simd, IteratorTag());
    test_equal_bad_alloc_datapar(par_simd, IteratorTag());

    test_equal_bad_alloc_async_datapar(simd(task), IteratorTag());
    test_equal_bad_alloc_async_datapar(par_simd(task), IteratorTag());
}

void equal_bad_alloc_test_datapar()
{
    test_equal_bad_alloc_datapar<std::random_access_iterator_tag>();
    test_equal_bad_alloc_datapar<std::forward_iterator_tag>();
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(hpx::program_options::variables_map& vm)
{
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    gen.seed(seed);

    equal_test1();
    equal_test2();
    equal_exception_test_datapar();
    equal_bad_alloc_test_datapar();
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
