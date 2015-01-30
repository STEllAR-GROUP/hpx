//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/components/unordered/unordered_map.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>
#include <functional>
#include <string>

#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_UNORDERED_MAP(std::string, double);

///////////////////////////////////////////////////////////////////////////////
template <typename Key, typename Value>
void test_global_iteration(hpx::unordered_map<Key, Value>& m,
    std::size_t size, Value const& val)
{
//     typedef typename hpx::unordered_map<Key, Value>::iterator iterator;
//     typedef hpx::traits::segmented_iterator_traits<iterator> traits;
//     HPX_TEST(traits::is_segmented_iterator::value);

//     typedef typename hpx::unordered_map<Key, Value>::const_iterator const_iterator;
//     typedef hpx::traits::segmented_iterator_traits<const_iterator> const_traits;
//     HPX_TEST(const_traits::is_segmented_iterator::value);

    HPX_TEST_EQ(m.size(), size);
    for(std::size_t i = 0; i != size; ++i)
    {
        std::string idx = boost::lexical_cast<std::string>(i);
        HPX_TEST_EQ(m[idx], val);
        m[idx] = Value(i+1);
        HPX_TEST_EQ(m[idx], Value(i+1));
    }

//     // test normal iteration
//     std::size_t count = 0;
//     std::size_t i = 42;
//     for (iterator it = v.begin(); it != v.end(); ++it, ++i, ++count)
//     {
//         HPX_TEST_NEQ(*it, val);
//         *it = T(i);
//         HPX_TEST_EQ(*it, T(i));
//     }
//     HPX_TEST_EQ(count, size);
//
//     count = 0;
//     i = 42;
//     for (const_iterator cit = v.cbegin(); cit != v.cend(); ++cit, ++i, ++count)
//     {
//         HPX_TEST_EQ(*cit, T(i));
//     }
//     HPX_TEST_EQ(count, size);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Key, typename Value>
void trivial_tests()
{
    {
        hpx::unordered_map<Key, Value> m;

        test_global_iteration(m, 0, Value());
    }
}

int main()
{
    trivial_tests<std::string, double>();

    return 0;
}

