///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2017 Taeguk Kwon
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <cstddef>
#include <forward_list>
#include <iterator>
#include <list>
#include <type_traits>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
template <typename IterTag, typename Enable = void>
struct test_container;

template <typename RandIterTag>
struct test_container<RandIterTag,
    typename std::enable_if<
        std::is_same<RandIterTag, std::random_access_iterator_tag>::value
    >::type>
{
    typedef std::vector<int> type;

    static type get_container(std::size_t size)
    {
        return type(size);
    }
};

template <typename BidirIterTag>
struct test_container<BidirIterTag,
    typename std::enable_if<
    std::is_same<BidirIterTag, std::bidirectional_iterator_tag>::value
>::type>
{
    typedef std::list<int> type;

    static type get_container(std::size_t size)
    {
        return type(size);
    }
};

template <typename FwdIterTag>
struct test_container<FwdIterTag,
    typename std::enable_if<
        std::is_same<FwdIterTag, std::forward_iterator_tag>::value
    >::type>
{
    typedef std::forward_list<int> type;

    static type get_container(std::size_t size)
    {
        return type(size);
    }
};
