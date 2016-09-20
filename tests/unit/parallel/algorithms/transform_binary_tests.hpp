//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_TRANSFORM_SEP_08_2014_0927AM)
#define HPX_PARALLEL_TEST_TRANSFORM_SEP_08_2014_0927AM

#include <hpx/include/parallel_transform.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

struct add
{
    template <typename T1, typename T2>
    auto operator()(T1 const& v1, T2 const& v2) const -> decltype(v1 + v2)
    {
        return v1 + v2;
    }
};

struct throw_always
{
    template <typename T1, typename T2>
    auto operator()(T1 const& v1, T2 const& v2) const -> decltype(v1 + v2)
    {
        throw std::runtime_error("test");
    }
};

struct throw_bad_alloc
{
    template <typename T1, typename T2>
    auto operator()(T1 const& v1, T2 const& v2) const -> decltype(v1 + v2)
    {
        throw std::bad_alloc();
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    auto result =
        hpx::parallel::transform(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            boost::begin(c2), boost::begin(d1), add());

    HPX_TEST(hpx::util::get<0>(result) == iterator(boost::end(c1)));
    HPX_TEST(hpx::util::get<1>(result) == boost::end(c2));
    HPX_TEST(hpx::util::get<2>(result) == boost::end(d1));

    // verify values
    std::vector<int> d2(c1.size());
    std::transform(boost::begin(c1), boost::end(c1),
        boost::begin(c2), boost::begin(d2), add());

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(d1), boost::end(d1), boost::begin(d2),
        [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    auto f =
        hpx::parallel::transform(p,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            boost::begin(c2), boost::begin(d1), add());
    f.wait();

    hpx::util::tuple<iterator, base_iterator, base_iterator> result = f.get();
    HPX_TEST(hpx::util::get<0>(result) == iterator(boost::end(c1)));
    HPX_TEST(hpx::util::get<1>(result) == boost::end(c2));
    HPX_TEST(hpx::util::get<2>(result) == boost::end(d1));

    // verify values
    std::vector<int> d2(c1.size());
    std::transform(boost::begin(c1), boost::end(c1),
        boost::begin(c2), boost::begin(d2), add());

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(d1), boost::end(d1), boost::begin(d2),
        [&count](int v1, int v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d2.size());
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_exception = false;
    try {
        hpx::parallel::transform(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            boost::begin(c2), boost::begin(d1),
            throw_always());

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
void test_transform_binary_exception_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::transform(p,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                boost::begin(c2), boost::begin(d1),
                throw_always());
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
void test_transform_binary_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_bad_alloc = false;
    try {
        hpx::parallel::transform(policy,
            iterator(boost::begin(c1)), iterator(boost::end(c1)),
            boost::begin(c2), boost::begin(d1),
            throw_bad_alloc());

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_transform_binary_bad_alloc_async(ExPolicy p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c1(10007);
    std::vector<int> c2(c1.size());
    std::vector<int> d1(c1.size()); //-V656
    std::iota(boost::begin(c1), boost::end(c1), std::rand());
    std::iota(boost::begin(c2), boost::end(c2), std::rand());

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try {
        auto f =
            hpx::parallel::transform(p,
                iterator(boost::begin(c1)), iterator(boost::end(c1)),
                boost::begin(c2), boost::begin(d1),
                throw_bad_alloc());
        returned_from_algorithm = true;
        f.get();

        HPX_TEST(false);
    }
    catch(std::bad_alloc const&) {
        caught_bad_alloc = true;
    }
    catch(...) {
        HPX_TEST(false);
    }

    HPX_TEST(caught_bad_alloc);
    HPX_TEST(returned_from_algorithm);
}

#endif
