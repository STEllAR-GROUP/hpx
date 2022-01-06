///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <forward_list>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename IterTag, typename DataType = int, typename Enable = void>
struct test_container;

template <typename RandIterTag, typename DataType>
struct test_container<RandIterTag, DataType,
    typename std::enable_if<std::is_same<RandIterTag,
        std::random_access_iterator_tag>::value>::type>
{
    typedef std::vector<DataType> type;

    static type get_container(std::size_t size)
    {
        return type(size);
    }
};

template <typename BidirIterTag, typename DataType>
struct test_container<BidirIterTag, DataType,
    typename std::enable_if<std::is_same<BidirIterTag,
        std::bidirectional_iterator_tag>::value>::type>
{
    typedef std::list<DataType> type;

    static type get_container(std::size_t size)
    {
        return type(size);
    }
};

template <typename FwdIterTag, typename DataType>
struct test_container<FwdIterTag, DataType,
    typename std::enable_if<
        std::is_same<FwdIterTag, std::forward_iterator_tag>::value>::type>
{
    typedef std::forward_list<DataType> type;

    static type get_container(std::size_t size)
    {
        return type(size);
    }
};
