//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_INCLUSIVE_SCAN_MAY28_1321)
#define HPX_PARALLEL_TEST_INCLUSIVE_SCAN_MAY28_1321

#include <hpx/hpx_init.hpp>
#include <hpx/hpx.hpp>
#include <hpx/include/parallel_executors.hpp>
#include <hpx/include/parallel_scan.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/iterator/counting_iterator.hpp>
#include <boost/range/functions.hpp>

#include <cstddef>
#include <iostream>
#include <utility>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
void inclusive_scan_benchmark()
{
    try {
      std::vector<double> c(100000000);
      std::vector<double> d(c.size());
      std::fill(boost::begin(c), boost::end(c), std::size_t(1));

      double const val(0);
      auto op =
          [](double v1, double v2) {
              return v1 + v2;
          };

      hpx::util::high_resolution_timer t;
      hpx::parallel::inclusive_scan(hpx::parallel::execution::par,
          boost::begin(c), boost::end(c), boost::begin(d),
          val, op);
      double elapsed = t.elapsed();

      // verify values
      std::vector<double> e(c.size());
      hpx::parallel::v1::detail::sequential_inclusive_scan(
          boost::begin(c), boost::end(c), boost::begin(e), val, op);

      bool ok = std::equal(boost::begin(d), boost::end(d), boost::begin(e));
      HPX_TEST(ok);
      if (ok) {
          std::cout << "<DartMeasurement name=\"InclusiveScanTime\" \n"
              << "type=\"numeric/double\">" << elapsed << "</DartMeasurement> \n";
      }
    }
    catch (...)
    {
      HPX_TEST(false);
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan1(ExPolicy && policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op =
        [](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    hpx::parallel::inclusive_scan(std::forward<ExPolicy>(policy),
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
        val, op);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val, op);

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan1_async(ExPolicy && p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    auto op =
        [](std::size_t v1, std::size_t v2) {
            return v1 + v2;
        };

    hpx::future<void> f =
        hpx::parallel::inclusive_scan(std::forward<ExPolicy>(p),
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
            val, op);
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val, op);

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan2(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    hpx::parallel::inclusive_scan(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
        val);

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val,
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan2_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    std::size_t const val(0);
    hpx::future<void> f =
        hpx::parallel::inclusive_scan(p,
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d),
            val);
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), val,
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan3(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    hpx::parallel::inclusive_scan(policy,
        iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d));

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), std::size_t(),
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan3_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    hpx::future<void> f =
        hpx::parallel::inclusive_scan(p,
            iterator(boost::begin(c)), iterator(boost::end(c)), boost::begin(d));
    f.wait();

    // verify values
    std::vector<std::size_t> e(c.size());
    hpx::parallel::v1::detail::sequential_inclusive_scan(
        boost::begin(c), boost::end(c), boost::begin(e), std::size_t(),
        std::plus<std::size_t>());

    HPX_TEST(std::equal(boost::begin(d), boost::end(d), boost::begin(e)));
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    try {
        hpx::parallel::inclusive_scan(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d), std::size_t(0),
            [](std::size_t v1, std::size_t v2)
            {
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
void test_inclusive_scan_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::inclusive_scan(p,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                boost::begin(d), std::size_t(0),
                [](std::size_t v1, std::size_t v2)
                {
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

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_inclusive_scan_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::execution::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::execution::is_execution_policy<ExPolicy>::value");

    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    try {
        hpx::parallel::inclusive_scan(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            boost::begin(d), std::size_t(0),
            [](std::size_t v1, std::size_t v2)
            {
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
void test_inclusive_scan_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<std::size_t>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<std::size_t> c(10007);
    std::vector<std::size_t> d(c.size());
    std::fill(boost::begin(c), boost::end(c), std::size_t(1));

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        hpx::future<void> f =
            hpx::parallel::inclusive_scan(p,
                iterator(boost::begin(c)), iterator(boost::end(c)),
                boost::begin(d), std::size_t(0),
                [](std::size_t v1, std::size_t v2)
                {
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

#define FILL_VALUE 10
#define ARRAY_SIZE 10000

// n'th value of sum of 1+2+3+...
int check_n_triangle(int n) {
    return n<0 ? 0 : (n)*(n+1)/2;
}

// n'th value of sum of x+x+x+...
int check_n_const(int n, int x) {
    return n<0 ? 0 : n*x;
}

// run scan algorithm, validate that output array hold expected answers.
template <typename ExPolicy>
void test_inclusive_scan_validate(ExPolicy p, std::vector<int> &a, std::vector<int> &b)
{
    using namespace hpx::parallel;
    typedef std::vector<int>::iterator Iter;

    // test 1, fill array with numbers counting from 0, then run scan algorithm
    a.clear();
    std::copy(boost::counting_iterator<int>(0),
        boost::counting_iterator<int>(ARRAY_SIZE), std::back_inserter(a));
    b.resize(a.size());
    hpx::parallel::inclusive_scan(p, a.begin(), a.end(), b.begin(), 0,
                                  [](int bar, int baz){ return bar+baz; });
    //
    for (int i=0; i<static_cast<int>(b.size()); ++i) {
        // counting from zero,
        int value = b[i]; //-V108
        int expected_value  = check_n_triangle(i);
        if (!HPX_TEST(value == expected_value)) break;
    }

    // test 2, fill array with numbers counting from 1, then run scan algorithm
    a.clear();
    std::copy(boost::counting_iterator<int>(1),
        boost::counting_iterator<int>(ARRAY_SIZE), std::back_inserter(a));
    b.resize(a.size());
    hpx::parallel::inclusive_scan(p, a.begin(), a.end(), b.begin(), 0,
                                  [](int bar, int baz){ return bar+baz; });
    //
    for (int i=0; i<static_cast<int>(b.size()); ++i) {
        // counting from 1, use i+1
        int value = b[i]; //-V108
        int expected_value  = check_n_triangle(i+1);
        if (!HPX_TEST(value == expected_value)) break;
    }

    // test 3, fill array with constant
    a.clear();
    std::fill_n(std::back_inserter(a), ARRAY_SIZE, FILL_VALUE);
    b.resize(a.size());
    hpx::parallel::inclusive_scan(p, a.begin(), a.end(), b.begin(), 0,
                                  [](int bar, int baz){ return bar+baz; });
    //
    for (int i=0; i<static_cast<int>(b.size()); ++i) {
        int value = b[i]; //-V108
        int expected_value  = check_n_const(i+1, FILL_VALUE);
        if (!HPX_TEST(value == expected_value)) break;
    }
}

#endif
