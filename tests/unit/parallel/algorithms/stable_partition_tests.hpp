//  Copyright (c) 2014-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_TEST_STABLE_PARTITION_SEP_24_2016_1210PM)
#define HPX_PARALLEL_TEST_STABLE_PARTITION_SEP_24_2016_1210PM

#include <hpx/include/parallel_partition.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <boost/range/functions.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "test_utils.hpp"

///////////////////////////////////////////////////////////////////////////////
struct less_than
{
    less_than(int partition_at)
      : partition_at_(partition_at)
    {}

    template <typename T>
    bool operator()(T const& val)
    {
        return val < partition_at_;
    }

    int partition_at_;
};

struct great_equal_than
{
    great_equal_than(int partition_at)
      : partition_at_(partition_at)
    {}

    template <typename T>
    bool operator()(T const& val)
    {
        return val >= partition_at_;
    }

    int partition_at_;
};

struct throw_always
{
    template <typename T>
    T operator()(T)
    {
        throw std::runtime_error("test");
    }
};

struct throw_bad_alloc
{
    template <typename T>
    T operator()(T) const
    {
        throw std::bad_alloc();
    }
};

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_stable_partition(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    std::vector<int> c(10007);
    std::vector<int> d(c.size());
    std::iota(boost::begin(c), boost::end(c), std::rand());
    std::copy(boost::begin(c), boost::end(c), boost::begin(d));

    int partition_at = std::rand();

    auto result =
        hpx::parallel::stable_partition(policy,
            iterator(boost::begin(c)), iterator(boost::end(c)),
            less_than(partition_at));

    auto partition_pt = std::find_if(boost::begin(c), boost::end(c),
        great_equal_than(partition_at));
    HPX_TEST(result.base() == partition_pt);

    // verify values
    std::stable_partition(boost::begin(d), boost::end(d), less_than(partition_at));

    std::size_t count = 0;
    HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
        [&count](std::size_t v1, std::size_t v2) -> bool {
            HPX_TEST_EQ(v1, v2);
            ++count;
            return v1 == v2;
        }));
    HPX_TEST_EQ(count, d.size());
}

template <typename ExPolicy, typename IteratorTag>
void test_stable_partition_async(ExPolicy p, IteratorTag)
{
//     typedef std::vector<int>::iterator base_iterator;
//     typedef test::test_iterator<base_iterator, IteratorTag> iterator;
//
//     std::vector<int> c(10007);
//     std::vector<int> d(c.size());
//     std::iota(boost::begin(c), boost::end(c), std::rand());
//
//     auto f =
//         hpx::parallel::stable_partition(p,
//             iterator(boost::begin(c)), iterator(boost::end(c)),
//             boost::begin(d),
//             add_one());
//     f.wait();
//
//     hpx::util::tuple<iterator, base_iterator> result = f.get();
//     HPX_TEST(hpx::util::get<0>(result) == iterator(boost::end(c)));
//     HPX_TEST(hpx::util::get<1>(result) == boost::end(d));
//
//     // verify values
//     std::size_t count = 0;
//     HPX_TEST(std::equal(boost::begin(c), boost::end(c), boost::begin(d),
//         [&count](std::size_t v1, std::size_t v2) -> bool {
//             HPX_TEST_EQ(v1 + 1, v2);
//             ++count;
//             return v1 + 1 == v2;
//         }));
//     HPX_TEST_EQ(count, d.size());
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_stable_partition_exception(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

//     typedef std::vector<int>::iterator base_iterator;
//     typedef test::test_iterator<base_iterator, IteratorTag> iterator;
//
//     std::vector<int> c(10007);
//     std::vector<int> d(c.size());
//     std::iota(boost::begin(c), boost::end(c), std::rand());
//
//     bool caught_exception = false;
//     try {
//         hpx::parallel::stable_partition(policy,
//             iterator(boost::begin(c)), iterator(boost::end(c)),
//             boost::begin(d),
//             throw_always());
//
//         HPX_TEST(false);
//     }
//     catch(hpx::exception_list const& e) {
//         caught_exception = true;
//         test::test_num_exceptions<ExPolicy, IteratorTag>::call(policy, e);
//     }
//     catch(...) {
//         HPX_TEST(false);
//     }
//
//     HPX_TEST(caught_exception);
}

template <typename ExPolicy, typename IteratorTag>
void test_stable_partition_exception_async(ExPolicy p, IteratorTag)
{
//     typedef std::vector<int>::iterator base_iterator;
//     typedef test::test_iterator<base_iterator, IteratorTag> iterator;
//
//     std::vector<int> c(10007);
//     std::vector<int> d(c.size());
//     std::iota(boost::begin(c), boost::end(c), std::rand());
//
//     bool caught_exception = false;
//     bool returned_from_algorithm = false;
//     try {
//         auto f =
//             hpx::parallel::stable_partition(p,
//                 iterator(boost::begin(c)), iterator(boost::end(c)),
//                 boost::begin(d),
//                 throw_always());
//         returned_from_algorithm = true;
//         f.get();
//
//         HPX_TEST(false);
//     }
//     catch(hpx::exception_list const& e) {
//         caught_exception = true;
//         test::test_num_exceptions<ExPolicy, IteratorTag>::call(p, e);
//     }
//     catch(...) {
//         HPX_TEST(false);
//     }
//
//     HPX_TEST(caught_exception);
//     HPX_TEST(returned_from_algorithm);
}

///////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_stable_partition_bad_alloc(ExPolicy policy, IteratorTag)
{
    static_assert(
        hpx::parallel::is_execution_policy<ExPolicy>::value,
        "hpx::parallel::is_execution_policy<ExPolicy>::value");

//     typedef std::vector<std::size_t>::iterator base_iterator;
//     typedef test::test_iterator<base_iterator, IteratorTag> iterator;
//
//     std::vector<std::size_t> c(10007);
//     std::vector<std::size_t> d(c.size());
//     std::iota(boost::begin(c), boost::end(c), std::rand());
//
//     bool caught_bad_alloc = false;
//     try {
//         hpx::parallel::stable_partition(policy,
//             iterator(boost::begin(c)), iterator(boost::end(c)),
//             boost::begin(d),
//             throw_bad_alloc());
//
//         HPX_TEST(false);
//     }
//     catch(std::bad_alloc const&) {
//         caught_bad_alloc = true;
//     }
//     catch(...) {
//         HPX_TEST(false);
//     }
//
//     HPX_TEST(caught_bad_alloc);
}

template <typename ExPolicy, typename IteratorTag>
void test_stable_partition_bad_alloc_async(ExPolicy p, IteratorTag)
{
//     typedef std::vector<std::size_t>::iterator base_iterator;
//     typedef test::test_iterator<base_iterator, IteratorTag> iterator;
//
//     std::vector<std::size_t> c(10007);
//     std::vector<std::size_t> d(c.size());
//     std::iota(boost::begin(c), boost::end(c), std::rand());
//
//     bool caught_bad_alloc = false;
//     bool returned_from_algorithm = false;
//     try {
//         auto f =
//             hpx::parallel::stable_partition(p,
//                 iterator(boost::begin(c)), iterator(boost::end(c)),
//                 boost::begin(d),
//                 throw_bad_alloc());
//         returned_from_algorithm = true;
//         f.get();
//
//         HPX_TEST(false);
//     }
//     catch(std::bad_alloc const&) {
//         caught_bad_alloc = true;
//     }
//     catch(...) {
//         HPX_TEST(false);
//     }
//
//     HPX_TEST(caught_bad_alloc);
//     HPX_TEST(returned_from_algorithm);
}

#endif
