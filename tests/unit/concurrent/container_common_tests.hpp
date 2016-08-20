//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// (C) Copyright Ion Gaztanaga 2004-2013.

#ifndef HPX_CONTAINER_TEST_CONTAINER_COMMON_TESTS_HPP
#define HPX_CONTAINER_TEST_CONTAINER_COMMON_TESTS_HPP

#include <algorithm>

namespace test {

template <class Container>
const Container &as_const(Container &c)
{
    return c;
}

template <typename Container>
typename Container::iterator
nth(Container& c, typename Container::size_type n)
{
    typename Container::iterator it = c.begin();
    std::advance(it, n);
    return it;
}

template <typename Container>
typename Container::const_iterator
nth(Container const& c, typename Container::size_type n)
{
    typename Container::iterator it = c.begin();
    std::advance(it, n);
    return it;
}

template <typename Container>
typename Container::size_type
index_of(Container& c, typename Container::iterator it)
{
    return std::distance(c.begin(), it);
}

template <typename Container>
typename Container::size_type
index_of(Container const& c, typename Container::const_iterator it)
{
    return std::distance(c.cbegin(), it);
}


// nth, index_of
template <class Container>
bool test_nth_index_of(Container &c)
{
    typename Container::iterator it;
    typename Container::const_iterator cit;
    typename Container::size_type sz, csz;

    // index 0
    it = nth(c, 0);
    sz = index_of(c, it);
    cit = nth(as_const(c), 0);
    csz = index_of(as_const(c), cit);

    if (it != c.begin())
        return false;
    if (cit != c.cbegin())
        return false;
    if (sz != 0)
        return false;
    if (csz != 0)
        return false;

    // index size()/2
    const typename Container::size_type sz_div_2 = c.size() / 2;
    it = nth(c, sz_div_2);
    sz = index_of(c, it);
    cit = nth(as_const(c), sz_div_2);
    csz = index_of(as_const(c), cit);

    if (it != (c.begin() + sz_div_2))
        return false;
    if (cit != (c.cbegin() + sz_div_2))
        return false;
    if (sz != sz_div_2)
        return false;
    if (csz != sz_div_2)
        return false;

    // index size()
    it = nth(c, c.size());
    sz = index_of(c, it);
    cit = nth(as_const(c), c.size());
    csz = index_of(as_const(c), cit);

    if (it != c.end())
        return false;
    if (cit != c.cend())
        return false;
    if (sz != c.size())
        return false;
    if (csz != c.size())
        return false;
    return true;
}

}

#endif
