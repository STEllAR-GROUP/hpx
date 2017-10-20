//  Copyright (c) 2014-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/include/traits.hpp>
#include <hpx/include/partitioned_vector_predef.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// The vector types to be used are defined in partitioned_vector module.
// HPX_REGISTER_PARTITIONED_VECTOR(double);
// HPX_REGISTER_PARTITIONED_VECTOR(int);

///////////////////////////////////////////////////////////////////////////////
template <typename T>
void test_global_iteration(hpx::partitioned_vector<T>& v, std::size_t size,
    T const& val)
{
    typedef typename hpx::partitioned_vector<T>::iterator iterator;
    typedef hpx::traits::segmented_iterator_traits<iterator> traits;
    HPX_TEST(traits::is_segmented_iterator::value);

    typedef typename hpx::partitioned_vector<T>::const_iterator const_iterator;
    typedef hpx::traits::segmented_iterator_traits<const_iterator> const_traits;
    HPX_TEST(const_traits::is_segmented_iterator::value);

    HPX_TEST_EQ(v.size(), size);
    for(std::size_t i = 0; i != size; ++i)
    {
        HPX_TEST_EQ(v[i], val);
        v[i] = T(i+1);
        HPX_TEST_EQ(v[i], T(i+1));
    }

    // test normal iteration
    std::size_t count = 0;
    std::size_t i = 42;
    for (iterator it = v.begin(); it != v.end(); ++it, ++i, ++count)
    {
        HPX_TEST_NEQ(*it, val);
        *it = T(i);
        HPX_TEST_EQ(*it, T(i));
    }
    HPX_TEST_EQ(count, size);

    count = 0;
    i = 42;
    for (const_iterator cit = v.cbegin(); cit != v.cend(); ++cit, ++i, ++count)
    {
        HPX_TEST_EQ(*cit, T(i));
    }
    HPX_TEST_EQ(count, size);
}

// test segmented iteration
template <typename T>
void test_segmented_iteration(hpx::partitioned_vector<T>& v, std::size_t size,
    std::size_t parts)
{
    typedef typename hpx::partitioned_vector<T>::iterator iterator;
    typedef hpx::traits::segmented_iterator_traits<iterator> traits;
    typedef typename traits::segment_iterator segment_iterator;
    typedef typename traits::local_segment_iterator local_segment_iterator;
    typedef typename traits::local_iterator local_iterator;
    typedef typename traits::local_raw_iterator local_raw_iterator;

    typedef typename hpx::partitioned_vector<T>::const_iterator const_iterator;
    typedef hpx::traits::segmented_iterator_traits<const_iterator> const_traits;
    typedef typename const_traits::segment_iterator const_segment_iterator;
    typedef typename const_traits::local_segment_iterator const_local_segment_iterator;
    typedef typename const_traits::local_iterator const_local_iterator;
    typedef typename const_traits::local_raw_iterator const_local_raw_iterator;

    HPX_TEST(!hpx::traits::segmented_iterator_traits<
            segment_iterator
        >::is_segmented_iterator::value);
    HPX_TEST(!hpx::traits::segmented_iterator_traits<
            const_segment_iterator
        >::is_segmented_iterator::value);

    HPX_TEST(!hpx::traits::segmented_iterator_traits<
            local_segment_iterator
        >::is_segmented_iterator::value);
    HPX_TEST(!hpx::traits::segmented_iterator_traits<
            const_local_segment_iterator
        >::is_segmented_iterator::value);

    HPX_TEST(!hpx::traits::segmented_iterator_traits<
            local_iterator
        >::is_segmented_iterator::value);
    HPX_TEST(!hpx::traits::segmented_iterator_traits<
            const_local_iterator
        >::is_segmented_iterator::value);

    // test segmented and local iteration
    std::size_t seg_count = 0;
    std::size_t count = 0;
    segment_iterator seg_end = traits::segment(v.end());
    for (segment_iterator seg_it = traits::segment(v.begin());
         seg_it != seg_end; ++seg_it, ++seg_count)
    {
        local_iterator loc_end = traits::end(seg_it);
        for (local_iterator lit = traits::begin(seg_it);
             lit != loc_end; ++lit, ++count)
        {
        }
    }
    HPX_TEST_EQ(count, size);
    HPX_TEST_EQ(seg_count, parts);

    // const
    count = 0;
    seg_count = 0;
    const_segment_iterator seg_cend = const_traits::segment(v.cend());
    for (const_segment_iterator seg_cit = const_traits::segment(v.cbegin());
         seg_cit != seg_cend; ++seg_cit, ++seg_count)
    {
        const_local_iterator loc_cend = const_traits::end(seg_cit);
        for (const_local_iterator lcit = const_traits::begin(seg_cit);
             lcit != loc_cend; ++lcit, ++count)
        {
        }
    }
    HPX_TEST_EQ(count, size);
    HPX_TEST_EQ(seg_count, parts);

    // test iteration over localities
    count = 0;
    for (hpx::id_type const& loc : hpx::find_all_localities())
    {
        std::uint32_t locality_id = hpx::naming::get_locality_id_from_id(loc);
        iterator end = v.end(locality_id);
        for (iterator it = v.begin(locality_id); it != end; ++it, ++count)
        {
            std::size_t i = 42;
            local_iterator loc_end = traits::end(traits::segment(it));
            for (local_iterator lit = traits::begin(traits::segment(it));
                 lit != loc_end; ++lit, ++i)
            {
                *lit = T(i);
                HPX_TEST_EQ(*lit, T(i));
            }
        }
    }
    HPX_TEST_EQ(count, size);

    count = 0;
    for (hpx::id_type const& loc : hpx::find_all_localities())
    {
        std::uint32_t locality_id = hpx::naming::get_locality_id_from_id(loc);
        const_iterator end = v.cend(locality_id);
        for (const_iterator it = v.cbegin(locality_id); it != end; ++it, ++count)
        {
            std::size_t i = 42;
            const_local_iterator loc_end =
                const_traits::end(const_traits::segment(it));
            for (const_local_iterator lcit =
                const_traits::begin(const_traits::segment(it));
                 lcit != loc_end; ++lcit, ++i)
            {
                HPX_TEST_EQ(*lcit, T(i));
            }
        }
    }
    HPX_TEST_EQ(count, size);

    // test segmented iteration over localities
    seg_count = 0;
    for (hpx::id_type const& loc : hpx::find_all_localities())
    {
        std::uint32_t locality_id = hpx::naming::get_locality_id_from_id(loc);
        local_segment_iterator seg_end = v.segment_end(locality_id);
        for (local_segment_iterator seg_it = v.segment_begin(locality_id);
             seg_it != seg_end; ++seg_it, ++seg_count)
        {
            // local raw iterators are valid locally only
            if (loc != hpx::find_here())
                continue;

            local_raw_iterator loc_end = traits::end(seg_it);
            for (local_raw_iterator lit = traits::begin(seg_it);
                 lit != loc_end; ++lit, ++count)
            {
            }
        }
    }
    HPX_TEST_EQ(seg_count, parts);

    seg_count = 0;
    for (hpx::id_type const& loc : hpx::find_all_localities())
    {
        std::uint32_t locality_id = hpx::naming::get_locality_id_from_id(loc);
        const_local_segment_iterator seg_cend = v.segment_cend(locality_id);
        for (const_local_segment_iterator seg_cit = v.segment_cbegin(locality_id);
             seg_cit != seg_cend; ++seg_cit, ++seg_count)
        {
            // local raw iterators are valid locally only
            if (loc != hpx::find_here())
                continue;

            const_local_raw_iterator loc_cend = const_traits::end(seg_cit);
            for (const_local_raw_iterator lcit = const_traits::begin(seg_cit);
                 lcit != loc_cend; ++lcit, ++count)
            {
            }
        }
    }
    HPX_TEST_EQ(seg_count, parts);

    // test iterator composition
    if (size != 0)
    {
        segment_iterator seg_it = traits::segment(v.begin());
        local_iterator lit1 = traits::local(v.begin());
        local_iterator lit2 = traits::begin(seg_it);
        HPX_TEST(lit1 == lit2);

        iterator it = traits::compose(seg_it, lit1);
        HPX_TEST(it == v.begin());

        const_segment_iterator seg_cit = const_traits::segment(v.cbegin());
        const_local_iterator lcit1 = const_traits::local(v.cbegin());
        const_local_iterator lcit2 = const_traits::begin(seg_cit);
        HPX_TEST(lcit1 == lcit2);

        const_iterator cit = const_traits::compose(seg_cit, lcit1);
        HPX_TEST(cit == v.cbegin());
    }
}

template <typename T>
void trivial_test_without_policy(std::size_t size, char const* prefix)
{
    std::string prefix_(prefix);

    {
        // create empty vector
        hpx::partitioned_vector<T> v;

        test_global_iteration(v, 0, T());
        test_segmented_iteration(v, 0, 0);
    }

    {
        // create and connect to empty vector
        std::string empty(prefix_ + "empty");

        hpx::partitioned_vector<T> base;
        base.register_as(hpx::launch::sync, empty);

        hpx::partitioned_vector<T> v;
        v.connect_to(hpx::launch::sync, empty);

        test_global_iteration(v, 0, T());
        test_segmented_iteration(v, 0, 0);
    }

    {
        // create vector with initial size != 0
        hpx::partitioned_vector<T> v(size);

        test_global_iteration(v, size, T());
        test_segmented_iteration(v, size, 1);
    }

    {
        // create vector with initial size != 0
        std::string size_(prefix_ + "size");

        hpx::partitioned_vector<T> base(size);
        base.register_as(hpx::launch::sync, size_);

        hpx::partitioned_vector<T> v;
        v.connect_to(hpx::launch::sync, size_);

        test_global_iteration(v, size, T());
        test_segmented_iteration(v, size, 1);
    }

    {
        // create vector with initial size and values
        hpx::partitioned_vector<T> v(size, T(999));

        test_global_iteration(v, size, T(999));
        test_segmented_iteration(v, size, 1);
    }

    {
        // create vector with initial size and values
        std::string size_value(prefix_ + "size_value");

        hpx::partitioned_vector<T> base(size, T(999));
        base.register_as(hpx::launch::sync, size_value);

        hpx::partitioned_vector<T> v;
        v.connect_to(hpx::launch::sync, size_value);

        test_global_iteration(v, size, T(999));
        test_segmented_iteration(v, size, 1);
    }
}

template <typename T, typename DistPolicy>
void trivial_test_with_policy(std::size_t size, std::size_t parts,
    DistPolicy const& policy, char const* prefix)
{
    std::string prefix_(prefix);

    {
        hpx::partitioned_vector<T> v(size, policy);

        test_global_iteration(v, size, T(0));
        test_segmented_iteration(v, size, parts);
    }

    {
        std::string policy_(prefix_ + "policy");

        hpx::partitioned_vector<T> base(size, policy);
        base.register_as(hpx::launch::sync, policy_);

        hpx::partitioned_vector<T> v;
        v.connect_to(hpx::launch::sync, policy_);

        test_global_iteration(v, size, T(0));
        test_segmented_iteration(v, size, parts);
    }

    {
        hpx::partitioned_vector<T> v(size, T(999), policy);

        test_global_iteration(v, size, T(999));
        test_segmented_iteration(v, size, parts);
    }

    {
        std::string policy_value(prefix_ + "policy_value");

        hpx::partitioned_vector<T> base(size, T(999), policy);
        base.register_as(hpx::launch::sync, policy_value);

        hpx::partitioned_vector<T> v;
        v.connect_to(hpx::launch::sync, policy_value);

        test_global_iteration(v, size, T(999));
        test_segmented_iteration(v, size, parts);
    }
}

template <typename T>
void trivial_tests()
{
    std::size_t const length = 12;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    trivial_test_without_policy<T>(length, "test1");

    trivial_test_with_policy<T>(length, 1, hpx::container_layout, "test1");
    trivial_test_with_policy<T>(length, 3, hpx::container_layout(3), "test2");
    trivial_test_with_policy<T>(length, 3,
        hpx::container_layout(3, localities), "test3");
    trivial_test_with_policy<T>(length, localities.size(),
        hpx::container_layout(localities), "test4");
}

int main()
{
    trivial_tests<double>();
    trivial_tests<int>();

    return 0;
}

