//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_reduce.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce1(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t val(42);
    auto op =
        [val](std::size_t v1, std::size_t v2) {
            return v1 + v2 + val;
        };

    std::size_t r1 = hpx::parallel::reduce(policy,
        iterator(std::begin(c)), iterator(std::end(c)), val, op);

    // verify values
    std::size_t r2 = std::accumulate(std::begin(c), std::end(c), val, op);
    HPX_TEST_EQ(r1, r2);
}

template <typename ExPolicy, typename IteratorTag>
void test_reduce1_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t val(42);
    auto op =
        [val](std::size_t v1, std::size_t v2) {
            return v1 + v2 + val;
        };

    hpx::future<std::size_t> f =
        hpx::parallel::reduce(p,
            iterator(std::begin(c)), iterator(std::end(c)), val, op);
    f.wait();

    // verify values
    std::size_t r2 = std::accumulate(std::begin(c), std::end(c), val, op);
    HPX_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_reduce1()
{
    using namespace hpx::parallel;

    test_reduce1(execution::seq, IteratorTag());
    test_reduce1(execution::par, IteratorTag());
    test_reduce1(execution::par_unseq, IteratorTag());

    test_reduce1_async(execution::seq(execution::task), IteratorTag());
    test_reduce1_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_reduce1(execution_policy(execution::seq), IteratorTag());
    test_reduce1(execution_policy(execution::par), IteratorTag());
    test_reduce1(execution_policy(execution::par_unseq), IteratorTag());

    test_reduce1(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_reduce1(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void reduce_test1()
{
    test_reduce1<std::random_access_iterator_tag>();
    test_reduce1<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_reduce1<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t const val(42);
    std::size_t r1 = hpx::parallel::reduce(policy,
        iterator(std::begin(c)), iterator(std::end(c)), val);

    // verify values
    std::size_t r2 = std::accumulate(std::begin(c), std::end(c), val);
    HPX_TEST_EQ(r1, r2);
}

template <typename ExPolicy, typename IteratorTag>
void test_reduce2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t const val(42);
    hpx::future<std::size_t> f =
        hpx::parallel::reduce(p,
            iterator(std::begin(c)), iterator(std::end(c)), val);
    f.wait();

    // verify values
    std::size_t r2 = std::accumulate(std::begin(c), std::end(c), val);
    HPX_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_reduce2()
{
    using namespace hpx::parallel;

    test_reduce2(execution::seq, IteratorTag());
    test_reduce2(execution::par, IteratorTag());
    test_reduce2(execution::par_unseq, IteratorTag());

    test_reduce2_async(execution::seq(execution::task), IteratorTag());
    test_reduce2_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_reduce2(execution_policy(execution::seq), IteratorTag());
    test_reduce2(execution_policy(execution::par), IteratorTag());
    test_reduce2(execution_policy(execution::par_unseq), IteratorTag());

    test_reduce2(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_reduce2(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void reduce_test2()
{
    test_reduce2<std::random_access_iterator_tag>();
    test_reduce2<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_reduce2<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce3(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    std::size_t r1 = hpx::parallel::reduce(policy,
        iterator(std::begin(c)), iterator(std::end(c)));

    // verify values
    std::size_t r2 = std::accumulate(std::begin(c), std::end(c), std::size_t(0));
    HPX_TEST_EQ(r1, r2);
}

template <typename ExPolicy, typename IteratorTag>
void test_reduce3_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    hpx::future<std::size_t> f =
        hpx::parallel::reduce(p,
            iterator(std::begin(c)), iterator(std::end(c)));
    f.wait();

    // verify values
    std::size_t r2 = std::accumulate(std::begin(c), std::end(c), std::size_t(0));
    HPX_TEST_EQ(f.get(), r2);
}

template <typename IteratorTag>
void test_reduce3()
{
    using namespace hpx::parallel;

    test_reduce3(execution::seq, IteratorTag());
    test_reduce3(execution::par, IteratorTag());
    test_reduce3(execution::par_unseq, IteratorTag());

    test_reduce3_async(execution::seq(execution::task), IteratorTag());
    test_reduce3_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_reduce3(execution_policy(execution::seq), IteratorTag());
    test_reduce3(execution_policy(execution::par), IteratorTag());
    test_reduce3(execution_policy(execution::par_unseq), IteratorTag());

    test_reduce3(execution_policy(execution::seq(execution::task)), IteratorTag());
    test_reduce3(execution_policy(execution::par(execution::task)), IteratorTag());
#endif
}

void reduce_test3()
{
    test_reduce3<std::random_access_iterator_tag>();
    test_reduce3<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_reduce3<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::reduce(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                return throw std::runtime_error("test"), v1 + v2;
            });

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_reduce_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::reduce(p,
                iterator(std::begin(c)), iterator(std::end(c)),
                std::size_t(42),
                [](std::size_t v1, std::size_t v2) {
                    return throw std::runtime_error("test"), v1 + v2;
                });
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(hpx::exception_list const& e) {
        caught_exception = true;
        test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_reduce_exception()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_reduce_exception(execution::seq, IteratorTag());
    test_reduce_exception(execution::par, IteratorTag());

    test_reduce_exception_async(execution::seq(execution::task), IteratorTag());
    test_reduce_exception_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_reduce_exception(execution_policy(execution::seq), IteratorTag());
    test_reduce_exception(execution_policy(execution::par), IteratorTag());

    test_reduce_exception(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_reduce_exception(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void reduce_exception_test()
{
    test_reduce_exception<std::random_access_iterator_tag>();
    test_reduce_exception<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_reduce_exception<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_reduce_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::reduce(policy,
            iterator(std::begin(c)), iterator(std::end(c)),
            std::size_t(42),
            [](std::size_t v1, std::size_t v2) {
                return throw std::bad_alloc(), v1 + v2;
            });

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_reduce_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::iota(std::begin(c), std::end(c), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::reduce(p,
                iterator(std::begin(c)), iterator(std::end(c)),
                std::size_t(42),
                [](std::size_t v1, std::size_t v2) {
                    return throw std::bad_alloc(), v1 + v2;
                });
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_exception = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_exception);
    HPX_TEST(returned_from_algorithm);
}

template <typename IteratorTag>
void test_reduce_bad_alloc()
{
    using namespace hpx::parallel;

    // If the execution policy object is of type vector_execution_policy,
    // std::terminate shall be called. therefore we do not test exceptions
    // with a vector execution policy
    test_reduce_bad_alloc(execution::seq, IteratorTag());
    test_reduce_bad_alloc(execution::par, IteratorTag());

    test_reduce_bad_alloc_async(execution::seq(execution::task), IteratorTag());
    test_reduce_bad_alloc_async(execution::par(execution::task), IteratorTag());

#if defined(HPX_HAVE_GENERIC_EXECUTION_POLICY)
    test_reduce_bad_alloc(execution_policy(execution::seq), IteratorTag());
    test_reduce_bad_alloc(execution_policy(execution::par), IteratorTag());

    test_reduce_bad_alloc(execution_policy(execution::seq(execution::task)),
        IteratorTag());
    test_reduce_bad_alloc(execution_policy(execution::par(execution::task)),
        IteratorTag());
#endif
}

void reduce_bad_alloc_test()
{
    test_reduce_bad_alloc<std::random_access_iterator_tag>();
    test_reduce_bad_alloc<std::forward_iterator_tag>();
#if defined(HPX_HAVE_ALGORITHM_INPUT_ITERATOR_SUPPORT)
    test_reduce_bad_alloc<std::input_iterator_tag>();
#endif
}

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    reduce_test1();
    reduce_test2();
    reduce_test3();

    reduce_exception_test();
    reduce_bad_alloc_test();
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
