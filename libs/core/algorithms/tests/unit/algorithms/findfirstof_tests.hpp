//  Copyright (c) 2021 Srinivas Yadav
//  copyright (c) 2014 Grant Mercer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/testing.hpp>
#include <hpx/parallel/algorithms/find.hpp>

#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "test_utils.hpp"

////////////////////////////////////////////////////////////////////////////
unsigned int seed = std::random_device{}();
std::mt19937 gen(seed);
std::uniform_int_distribution<> dis(0, 10006);
std::uniform_int_distribution<> dist(0, 2);

template <typename IteratorTag>
void test_find_first_of(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    int find_first_of_pos = dis(gen);
    int random_sub_seq_pos = dist(gen);

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 19);
    int h[] = {1, 7, 18, 3};
    c[find_first_of_pos] = h[random_sub_seq_pos];    //-V108

    iterator index = hpx::find_first_of(iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    base_iterator test_index = std::begin(c) + find_first_of_pos;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_first_of(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    int find_first_of_pos = dis(gen);
    int random_sub_seq_pos = dist(gen);

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 19);
    int h[] = {1, 7, 18, 3};
    c[find_first_of_pos] = h[random_sub_seq_pos];    //-V108

    iterator index = hpx::find_first_of(policy, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));

    base_iterator test_index = std::begin(c) + find_first_of_pos;

    HPX_TEST(index == iterator(test_index));
}

template <typename ExPolicy, typename IteratorTag>
void test_find_first_of_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::test_iterator<base_iterator, IteratorTag> iterator;

    int find_first_of_pos = dis(gen);
    int random_sub_seq_pos = dist(gen);

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 19);
    int h[] = {1, 7, 18, 3};
    c[find_first_of_pos] = h[random_sub_seq_pos];    //-V108

    hpx::future<iterator> f = hpx::find_first_of(p, iterator(std::begin(c)),
        iterator(std::end(c)), std::begin(h), std::end(h));
    f.wait();

    // create iterator at position of value to be found
    base_iterator test_index = std::begin(c) + find_first_of_pos;

    HPX_TEST(f.get() == iterator(test_index));
}

///////////////////////////////////////////////////////////////////////////////
template <typename IteratorTag>
void test_find_first_of_exception(IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 1;

    bool caught_exception = false;
    try
    {
        int h[] = {1, 2};

        hpx::find_first_of(decorated_iterator(std::begin(c),
                               []() { throw std::runtime_error("test"); }),
            decorated_iterator(std::end(c)), std::begin(h), std::end(h));
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
void test_find_first_of_exception(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;
    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 1;

    bool caught_exception = false;
    try
    {
        int h[] = {1, 2};

        hpx::find_first_of(policy,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(std::end(c)), std::begin(h), std::end(h));
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
void test_find_first_of_exception_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 1;

    bool caught_exception = false;
    bool returned_from_algorithm = false;
    try
    {
        int h[] = {1, 2};

        hpx::future<decorated_iterator> f = hpx::find_first_of(p,
            decorated_iterator(
                std::begin(c), []() { throw std::runtime_error("test"); }),
            decorated_iterator(std::end(c)), std::begin(h), std::end(h));
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

//////////////////////////////////////////////////////////////////////////////
template <typename ExPolicy, typename IteratorTag>
void test_find_first_of_bad_alloc(ExPolicy&& policy, IteratorTag)
{
    static_assert(hpx::is_execution_policy<ExPolicy>::value,
        "hpx::is_execution_policy<ExPolicy>::value");

    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(100007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 1;

    bool caught_bad_alloc = false;
    try
    {
        int h[] = {1, 2};

        hpx::find_first_of(policy,
            decorated_iterator(std::begin(c), []() { throw std::bad_alloc(); }),
            decorated_iterator(std::end(c)), std::begin(h), std::end(h));
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
void test_find_first_of_bad_alloc_async(ExPolicy&& p, IteratorTag)
{
    typedef std::vector<int>::iterator base_iterator;
    typedef test::decorated_iterator<base_iterator, IteratorTag>
        decorated_iterator;

    std::vector<int> c(10007);
    std::iota(std::begin(c), std::end(c), gen() + 1);
    c[c.size() / 2] = 1;

    bool caught_bad_alloc = false;
    bool returned_from_algorithm = false;
    try
    {
        int h[] = {1, 2};

        hpx::future<decorated_iterator> f = hpx::find_first_of(p,
            decorated_iterator(std::begin(c), []() { throw std::bad_alloc(); }),
            decorated_iterator(std::end(c)), std::begin(h), std::end(h));
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
