//  Copyright (c) 2016 Zahra Khatami, Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_FOREACH_MAY24_16)
#define HPX_PARALLEL_TEST_FOREACH_MAY24_16

#include <hpx/include/parallel_for_each.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <numeric>
#include <vector>

#include "test_utils.hpp"
///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef hpx::parallel::util::detail::prefetching_iterator<double> iterator;
    typedef hpx::parallel::util::detail::make_prefetcher_context<double> mpctx;

    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007,1.0);
    auto ctx = mpctx(0,10007,{c.data()},prefetch_distance_factor);

    hpx::parallel::for_each(std::forward<ExPolicy>(policy),
        ctx.begin(), ctx.end(),
        [&](std::size_t i) {
            c[i] = 42.1;
        });

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching_async(ExPolicy && p, IteratorTag)
{
    typedef hpx::parallel::util::detail::prefetching_iterator<double> iterator;
    typedef hpx::parallel::util::detail::make_prefetcher_context<double> mpctx;

    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007,1.0);
    auto ctx = mpctx(0,10007,{c.data()},prefetch_distance_factor);

    hpx::future<iterator> f =
        hpx::parallel::for_each(std::forward<ExPolicy>(p),
        ctx.begin(), ctx.end(),
        [&](std::size_t i)
        {
            c[i] = 42.1;
        });
    f.wait();

    // verify values
    std::size_t count = 0;
    std::for_each(boost::begin(c), boost::end(c),
        [&count](std::size_t v) -> void {
            HPX_TEST_EQ(v, std::size_t(42));
            ++count;
        });
    HPX_TEST_EQ(count, c.size());
}

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef hpx::parallel::util::detail::prefetching_iterator<double> iterator;
    typedef hpx::parallel::util::detail::make_prefetcher_context<double> mpctx;

    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007,1.0);
    auto ctx = mpctx(0,10007,{c.data()},prefetch_distance_factor);

    bool caught_exception = false;
    try {
        hpx::parallel::for_each(policy,
            ctx.begin(), ctx.end(),
            [](std::size_t i) { throw std::runtime_error("test"); });

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
void test_for_each_prefetching_exception_async(ExPolicy p, IteratorTag)
{
    typedef hpx::parallel::util::detail::prefetching_iterator<double> iterator;
    typedef hpx::parallel::util::detail::make_prefetcher_context<double> mpctx;

    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007,1.0);
    auto ctx = mpctx(0,10007,{c.data()},prefetch_distance_factor);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<iterator> f =
            hpx::parallel::for_each(p,
                ctx.begin(), ctx.end(),
                [](std::size_t i) { throw std::runtime_error("test"); });
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

////////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_for_each_prefetching_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef hpx::parallel::util::detail::prefetching_iterator<double> iterator;
    typedef hpx::parallel::util::detail::make_prefetcher_context<double> mpctx;

    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007,1.0);
    auto ctx = mpctx(0,10007,{c.data()},prefetch_distance_factor);

    bool caught_exception = false;
    try {
        hpx::parallel::for_each(policy,
            ctx.begin(), ctx.end(),
            [](std::size_t i) { throw std::bad_alloc(); });

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
void test_for_each_prefetching_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef hpx::parallel::util::detail::prefetching_iterator<double> iterator;
    typedef hpx::parallel::util::detail::make_prefetcher_context<double> mpctx;

    std::size_t prefetch_distance_factor = 2;
    std::vector<double> c(10007,1.0);
    auto ctx = mpctx(0,10007,{c.data()},prefetch_distance_factor);

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {

        hpx::future<iterator> f =
            hpx::parallel::for_each(p,
                ctx.begin(), ctx.end(),
                [](std::size_t i) { throw std::bad_alloc(); });
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

#endif
