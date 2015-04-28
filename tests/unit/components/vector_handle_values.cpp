//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/vector.hpp>
#include <hpx/include/parallel_for_each.hpp>

#include <hpx/util/lightweight_test.hpp>

#include <vector>
#include <iostream>


///////////////////////////////////////////////////////////////////////////////
// Define the vector types to be used.
HPX_REGISTER_VECTOR(double);
HPX_REGISTER_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void fill_vector(hpx::vector<T>& v, T const& val)
{
    typename hpx::vector<T>::iterator it = v.begin(), end = v.end();
    for (/**/; it != end; ++it)
        *it = val;
}

template <typename Vector, typename T1>
void fill_vector(Vector& v, T1 val, T1 dist)
{
    typename Vector::iterator it = v.begin(), end = v.end();
    for (/**/; it != end; ++it)
    {
        *it = val;
        val += dist;
    }
}

///////////////////////////////////////////////////////////////////////////////
template <typename Vector>
void compare_vectors(Vector const& v1, Vector const& v2,
    bool must_be_equal = true)
{
    typedef typename Vector::const_iterator const_iterator;

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

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void handle_values_tests(hpx::vector<T>& v)
{
    fill_vector(v, T(42));

    std::vector<std::size_t> positions(2);
    fill_vector(positions, 0, 2);

    std::vector<T> values(positions.size());
    fill_vector(values, T(48), T(3));

    v.set_values(0, positions, values);
    std::vector<T> result = v.get_values_sync(0, positions);

    compare_vectors(values, result);
}

template <typename T>
void handle_values_tests_distributed_access(hpx::vector<T>& v)
{
    fill_vector (v, T(42));

    std::vector<std::size_t> positions(5);
    fill_vector(positions, 0, 2);
    std::vector<std::size_t> positions2(5);
    fill_vector(positions2, 1, 2);

    std::vector<T> values(positions.size());
    fill_vector(values, T(48), T(3));
    std::vector<T> values2(positions2.size());
    fill_vector(values2, T(42), T(0));

    v.set_values(positions, values);
    std::vector<T> result  = v.get_values_sync(positions );
    std::vector<T> result2 = v.get_values_sync(positions2);

    compare_vectors(values , result);
    compare_vectors(values2, result2);
}

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename DistPolicy>
void handle_values_tests_with_policy(std::size_t size, std::size_t localities,
    DistPolicy const& policy)
{
    {
        hpx::vector<T> v(size, policy);
        handle_values_tests(v);
    }

    {
        hpx::vector<T> v(size, policy);
        handle_values_tests_distributed_access(v);
    }
}

template <typename T>
void handle_values_tests()
{
    std::size_t const length = 12;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    {
        hpx::vector<T> v(length);
        handle_values_tests(v);
    }
    {
        hpx::vector<T> v(length);
        handle_values_tests_distributed_access(v);
    }
    {
        hpx::vector<T> v(length, T(42));
        handle_values_tests(v);
    }

    handle_values_tests_with_policy<T>(length, 1, hpx::container_layout);
    handle_values_tests_with_policy<T>(length, 3, hpx::container_layout(3));
    handle_values_tests_with_policy<T>(length, 3,
        hpx::container_layout(3, localities));
    handle_values_tests_with_policy<T>(length, localities.size(),
        hpx::container_layout(localities));
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    handle_values_tests<double>();
    handle_values_tests<int>();

    return 0;
}

