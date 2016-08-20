//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Copyright (C) 2013 Cromwell D. Enage

#ifndef HPX_CONTAINER_TEST_INSERT_TEST_HPP
#define HPX_CONTAINER_TEST_INSERT_TEST_HPP

#include <hpx/config.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <deque>

#include "check_equal_containers.hpp"

namespace test {

template <class SeqContainer>
void test_insert_range(std::deque<int> &std_deque, SeqContainer &seq_container,
    std::deque<int> const &input_deque, std::size_t index)
{
    HPX_TEST(CheckEqualContainers(std_deque, seq_container));

    std_deque.insert(
        std_deque.begin() + index, input_deque.begin(), input_deque.end());

    seq_container.insert(
        seq_container.begin() + index, input_deque.begin(), input_deque.end());
    HPX_TEST(CheckEqualContainers(std_deque, seq_container));
}

template <class SeqContainer>
bool test_range_insertion()
{
    int err_count = hpx::util::report_errors();

    typedef typename SeqContainer::value_type value_type;
    std::deque<int> input_deque;
    for (int element = -10; element < 10; ++element)
    {
        input_deque.push_back(element + 20);
    }

    for (std::size_t i = 0; i <= input_deque.size(); ++i)
    {
        std::deque<int> std_deque;
        std::unique_ptr<SeqContainer> const pseqcontainer(new SeqContainer());
        SeqContainer &seq_container = *pseqcontainer;

        for (int element = -10; element < 10; ++element)
        {
            std_deque.push_back(element);
            seq_container.push_back(value_type(element));
        }
        test_insert_range(std_deque, seq_container, input_deque, i);
    }

    return err_count == hpx::util::report_errors();
}

}    //namespace test {

#endif    //#ifndef BOOST_CONTAINER_TEST_INSERT_TEST_HPP
