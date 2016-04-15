//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_PARTITIONED_VECTOR(double);
HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void fill_vector(hpx::partitioned_vector<T>& v, T const& val)
{
    typename hpx::partitioned_vector<T>::iterator it = v.begin(), end = v.end();
    for (/**/; it != end; ++it)
        *it = val;
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void compare_vectors(hpx::partitioned_vector<T> const& v1,
    hpx::partitioned_vector<T> const& v2, bool must_be_equal = true)
{
    typedef typename hpx::partitioned_vector<T>::const_iterator const_iterator;

    HPX_TEST_EQ(v1.size(), v2.size());

    const_iterator it1 = v1.begin(), it2 = v2.begin();
    const_iterator end1 = v1.end(), end2 = v2.end();
    for (/**/; it1 != end1 && it2 != end2; ++it1, ++it2)
    {
        if (must_be_equal)
        {
            HPX_TEST_EQ(*it1, *it2);
        }
        else
        {
            HPX_TEST_NEQ(*it1, *it2);
        }
    }
}

template <typename T>
void move_tests(hpx::partitioned_vector<T> const& v1)
{
    hpx::partitioned_vector<T> v2(v1);
    compare_vectors(v1, v2);

    hpx::partitioned_vector<T> v3(std::move(v2));
    compare_vectors(v1, v3);

    fill_vector(v3, T(43));
    compare_vectors(v1, v3, false);

    hpx::partitioned_vector<T> v4;
    v4 = v1;
    compare_vectors(v1, v4);

    hpx::partitioned_vector<T> v5;
    v5 = std::move(v4);
    compare_vectors(v1, v5);

    fill_vector(v5, T(43));
    compare_vectors(v1, v5, false);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T, typename DistPolicy>
void move_tests_with_policy(std::size_t size, std::size_t localities,
    DistPolicy const& policy)
{
    hpx::partitioned_vector<T> v1(size, policy);

    hpx::partitioned_vector<T> v2(v1);
    compare_vectors(v1, v2);

    hpx::partitioned_vector<T> v3(std::move(v2));
    compare_vectors(v1, v3);

    fill_vector(v3, T(43));
    compare_vectors(v1, v3, false);

    hpx::partitioned_vector<T> v4;
    v4 = v1;
    compare_vectors(v1, v4);

    hpx::partitioned_vector<T> v5;
    v5 = std::move(v4);
    compare_vectors(v1, v5);

    fill_vector(v5, T(43));
    compare_vectors(v1, v5, false);
}

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void move_tests()
{
    std::size_t const length = 12;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    {
        hpx::partitioned_vector<T> v;
        move_tests(v);
    }

    {
        hpx::partitioned_vector<T> v(length);
        move_tests(v);
    }

    {
        hpx::partitioned_vector<T> v(length, T(42));
        move_tests(v);
    }

    move_tests_with_policy<T>(length, 1, hpx::container_layout);
    move_tests_with_policy<T>(length, 3, hpx::container_layout(3));
    move_tests_with_policy<T>(length, 3, hpx::container_layout(3, localities));
    move_tests_with_policy<T>(length, localities.size(),
        hpx::container_layout(localities));
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    move_tests<double>();
    move_tests<int>();

    return 0;
}

