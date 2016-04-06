//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/unordered_map.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>
#include <functional>
#include <string>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_UNORDERED_MAP(std::string, double);

///////////////////////////////////////////////////////////////////////////////
template <typename Key, typename Value, typename Hash, typename KeyEqual>
void test_global_iteration(hpx::unordered_map<Key, Value, Hash, KeyEqual>& m,
    Value const& val = Value())
{
    std::size_t size = m.size();

//     typedef typename hpx::unordered_map<Key, Value>::iterator iterator;
//     typedef hpx::traits::segmented_iterator_traits<iterator> traits;
//     HPX_TEST(traits::is_segmented_iterator::value);

//     typedef typename hpx::unordered_map<Key, Value>::const_iterator const_iterator;
//     typedef hpx::traits::segmented_iterator_traits<const_iterator> const_traits;
//     HPX_TEST(const_traits::is_segmented_iterator::value);

    for(std::size_t i = 0; i != size; ++i)
    {
        std::string idx = std::to_string(i);
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

template <typename Key, typename Value, typename Hash, typename KeyEqual>
void fill_unordered_map(hpx::unordered_map<Key, Value, Hash, KeyEqual>& m,
    std::size_t count, Value const& val)
{
    for (std::size_t i = 0; i != count; ++i)
    {
        std::string idx = std::to_string(i);
        m[idx] = val;
    }
    HPX_TEST(m.size() == count);
}

///////////////////////////////////////////////////////////////////////////////
template <typename Key, typename Value, typename DistPolicy>
void trivial_tests(DistPolicy const& policy)
{
    // bucket_count
    {
        hpx::unordered_map<Key, Value> m(17, policy);
        test_global_iteration(m);

        fill_unordered_map(m, 107, Value(42));
        test_global_iteration(m, Value(42));
    }

    // bucket_count, hash
    {
        hpx::unordered_map<Key, Value> m(17, std::hash<std::string>(),
            std::equal_to<std::string>(), policy);
        test_global_iteration(m);

        fill_unordered_map(m, 107, Value(42));
        test_global_iteration(m, Value(42));
    }

    // bucket_count, hash, key_equal
    {
        hpx::unordered_map<Key, Value> m(17, std::hash<std::string>(), policy);
        test_global_iteration(m);

        fill_unordered_map(m, 107, Value(42));
        test_global_iteration(m, Value(42));
    }
}

template <typename Key, typename Value>
void trivial_tests()
{
    // default constructed
    {
        hpx::unordered_map<Key, Value> m;
        test_global_iteration(m);

        fill_unordered_map(m, 107, Value(42));
        test_global_iteration(m, Value(42));
    }

    // bucket_count
    {
        hpx::unordered_map<Key, Value> m(17);
        test_global_iteration(m);

        fill_unordered_map(m, 107, Value(42));
        test_global_iteration(m, Value(42));
    }

    // bucket_count, hash
    {
        hpx::unordered_map<Key, Value> m(17, std::hash<std::string>());
        test_global_iteration(m);

        fill_unordered_map(m, 107, Value(42));
        test_global_iteration(m, Value(42));
    }

    // bucket_count, hash, key_equal
    {
        hpx::unordered_map<Key, Value> m(17, std::hash<std::string>(),
            std::equal_to<std::string>());
        test_global_iteration(m);

        fill_unordered_map(m, 107, Value(42));
        test_global_iteration(m, Value(42));
    }
}

int main()
{
    trivial_tests<std::string, double>();

    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    trivial_tests<std::string, double>(hpx::container_layout);
    trivial_tests<std::string, double>(hpx::container_layout(3));
    trivial_tests<std::string, double>(hpx::container_layout(3, localities));
    trivial_tests<std::string, double>(hpx::container_layout(localities));

    return 0;
}

