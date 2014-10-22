//  Copyright (c) 2014 Hartmut Kaiser
//
// (C) Copyright Ion Gaztanaga 2004-2012. Distributed under the Boost
// Software License, Version 1.0. (See accompanying file
// LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/components/vector/vector.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>
#include <functional>

#include <boost/foreach.hpp>

// #include "check_equal_containers.hpp"
// #include "movable_int.hpp"
// #include "expand_bwd_test_allocator.hpp"
// #include "expand_bwd_test_template.hpp"
// #include "dummy_test_allocator.hpp"
// #include "propagate_allocator_test.hpp"
// #include "vector_test.hpp"

// namespace hpx { namespace components
// {
//     // Explicit instantiation to detect compilation errors
//     template class boost::container::vector<test::movable_and_copyable_int,
//        test::simple_allocator<test::movable_and_copyable_int> >;
//
//     template class boost::container::vector<test::movable_and_copyable_int,
//        test::dummy_test_allocator<test::movable_and_copyable_int> >;
//
//     template class boost::container::vector<test::movable_and_copyable_int,
//        std::allocator<test::movable_and_copyable_int> >;
// }}
//
// int test_expand_bwd()
// {
//    // Now test all back insertion possibilities
//
//    // First raw ints
//    typedef test::expand_bwd_test_allocator<int>
//       int_allocator_type;
//    typedef vector<int, int_allocator_type>
//       int_vector;
//
//    if(!test::test_all_expand_bwd<int_vector>())
//       return 1;
//
//    //Now user defined wrapped int
//    typedef test::expand_bwd_test_allocator<test::int_holder>
//       int_holder_allocator_type;
//    typedef vector<test::int_holder, int_holder_allocator_type>
//       int_holder_vector;
//
//    if(!test::test_all_expand_bwd<int_holder_vector>())
//       return 1;
//
//    //Now user defined bigger wrapped int
//    typedef test::expand_bwd_test_allocator<test::triple_int_holder>
//       triple_int_holder_allocator_type;
//
//    typedef vector<test::triple_int_holder, triple_int_holder_allocator_type>
//       triple_int_holder_vector;
//
//    if(!test::test_all_expand_bwd<triple_int_holder_vector>())
//       return 1;
//
//    return 0;
// }
//
// class recursive_vector
// {
//    public:
//    int id_;
//    vector<recursive_vector> vector_;
// };
//
// void recursive_vector_test()//Test for recursive types
// {
//    vector<recursive_vector> recursive_vector_vector;
// }
//
// enum Test
// {
//    zero, one, two, three, four, five, six
// };

template <typename T>
void test_global_iteration(hpx::vector<T>& v, std::size_t size, T const& val)
{
    typedef hpx::vector<T>::iterator iterator;
    typedef hpx::vector<T>::const_iterator const_iterator;

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
void test_segmented_iteration(hpx::vector<T>& v, std::size_t size,
    std::size_t parts)
{
    typedef hpx::vector<T>::iterator iterator;
    typedef hpx::traits::segmented_iterator_traits<iterator> traits;
    typedef traits::segment_iterator segment_iterator;
    typedef traits::local_iterator local_iterator;

    typedef hpx::vector<T>::const_iterator const_iterator;
    typedef hpx::traits::segmented_iterator_traits<const_iterator> const_traits;
    typedef const_traits::segment_iterator const_segment_iterator;
    typedef const_traits::local_iterator const_local_iterator;

    HPX_TEST(!hpx::traits::segmented_iterator_traits<
            segment_iterator
        >::is_segmented_iterator::value);
    HPX_TEST(!hpx::traits::segmented_iterator_traits<
            const_segment_iterator
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

    // test segmented iteration over localities
    count = 0;
    seg_count = 0;
    BOOST_FOREACH(hpx::id_type const& loc, hpx::find_all_localities())
    {
        boost::uint32_t locality_id = hpx::naming::get_locality_id_from_id(loc);
        segment_iterator seg_end = v.segment_end(locality_id);
        for (segment_iterator seg_it = v.segment_begin(locality_id);
             seg_it != seg_end; ++seg_it, ++seg_count)
        {
            local_iterator loc_end = traits::end(seg_it);
            for (local_iterator lit = traits::begin(seg_it);
                 lit != loc_end; ++lit, ++count)
            {
            }
        }
    }
    HPX_TEST_EQ(count, size);
    HPX_TEST_EQ(seg_count, parts);

    count = 0;
    seg_count = 0;
    BOOST_FOREACH(hpx::id_type const& loc, hpx::find_all_localities())
    {
        boost::uint32_t locality_id = hpx::naming::get_locality_id_from_id(loc);
        const_segment_iterator seg_cend = v.segment_cend(locality_id);
        for (const_segment_iterator seg_cit = v.segment_cbegin(locality_id);
             seg_cit != seg_cend; ++seg_cit, ++seg_count)
        {
            const_local_iterator loc_cend = const_traits::end(seg_cit);
            for (const_local_iterator lcit = const_traits::begin(seg_cit);
                 lcit != loc_cend; ++lcit, ++count)
            {
            }
        }
    }
    HPX_TEST_EQ(count, size);
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
    typedef hpx::vector<T>::iterator iterator;
    typedef hpx::traits::segmented_iterator_traits<iterator> traits;
    HPX_TEST(traits::is_segmented_iterator::value);

    typedef hpx::vector<T>::const_iterator const_iterator;
    typedef hpx::traits::segmented_iterator_traits<const_iterator> const_traits;
    HPX_TEST(const_traits::is_segmented_iterator::value);

    std::string prefix_(prefix);

    {
        // create empty vector
        hpx::vector<T> v;

        test_global_iteration(v, 0, T());
        test_segmented_iteration(v, 0, 0);
    }

    {
        // create and connect to empty vector
        std::string empty(prefix_ + "empty");

        hpx::vector<T> base;
        base.register_as(empty);
        hpx::vector<T> v(empty);

        test_global_iteration(v, 0, T());
        test_segmented_iteration(v, 0, 0);
    }

    {
        // create vector with initial size != 0
        hpx::vector<T> v(size);

        test_global_iteration(v, size, T());
        test_segmented_iteration(v, size, 1);
    }

    {
        // create vector with initial size != 0
        std::string size_(prefix_ + "size");

        hpx::vector<T> base(size, size_);
        hpx::vector<T> v(size_);

        test_global_iteration(v, size, T());
        test_segmented_iteration(v, size, 1);
    }

    {
        // create vector with initial size and values
        hpx::vector<T> v(size, T(999));

        test_global_iteration(v, size, T(999));
        test_segmented_iteration(v, size, 1);
    }

    {
        // create vector with initial size and values
        std::string size_value(prefix_ + "size_value");

        hpx::vector<T> base(size, T(999), size_value);
        hpx::vector<T> v(size_value);

        test_global_iteration(v, size, T(999));
        test_segmented_iteration(v, size, 1);
    }
}

template <typename T, typename DistPolicy>
void trivial_test_with_policy(std::size_t size, std::size_t parts,
    DistPolicy const& policy, char const* prefix)
{
    typedef hpx::vector<T>::iterator iterator;
    typedef hpx::traits::segmented_iterator_traits<iterator> traits;
    typedef traits::segment_iterator segment_iterator;

    typedef hpx::vector<T>::const_iterator const_iterator;
    typedef hpx::traits::segmented_iterator_traits<const_iterator> const_traits;
    typedef const_traits::segment_iterator const_segment_iterator;

    std::string prefix_(prefix);

    {
        hpx::vector<T> v(size, policy);

        test_global_iteration(v, size, T(0));
        test_segmented_iteration(v, size, parts);
    }

    {
        std::string policy_(prefix_ + "policy");

        hpx::vector<T> base(size, policy, policy_);
        hpx::vector<T> v(policy_);

        test_global_iteration(v, size, T(0));
        test_segmented_iteration(v, size, parts);
    }

    {
        hpx::vector<T> v(size, T(999), policy);

        test_global_iteration(v, size, T(999));
        test_segmented_iteration(v, size, parts);
    }

    {
        std::string policy_value(prefix_ + "policy_value");

        hpx::vector<T> base(size, T(999), policy, policy_value);
        hpx::vector<T> v(policy_value);

        test_global_iteration(v, size, T(999));
        test_segmented_iteration(v, size, parts);
    }
}

template <typename T>
void trivial_tests()
{
    std::size_t const length = 12;
    std::vector<hpx::id_type> localities = hpx::find_all_localities();

    trivial_test_without_policy<double>(length, "1");

    trivial_test_with_policy<double>(length, 1, hpx::block, "1");
    trivial_test_with_policy<double>(length, 3, hpx::block(3), "2");
    trivial_test_with_policy<double>(length, 3, hpx::block(3, localities), "3");
    trivial_test_with_policy<double>(length, localities.size(),
        hpx::block(localities), "4");

    trivial_test_with_policy<double>(length, 1, hpx::cyclic, "5");
    trivial_test_with_policy<double>(length, 3, hpx::cyclic(3), "6");
    trivial_test_with_policy<double>(length, 3, hpx::cyclic(3, localities), "7");
    trivial_test_with_policy<double>(length, localities.size(),
        hpx::cyclic(localities), "8");

    trivial_test_with_policy<double>(length, 1, hpx::block_cyclic, "9");
    trivial_test_with_policy<double>(length, 3, hpx::block_cyclic(3), "10");
    trivial_test_with_policy<double>(length, 3,
        hpx::block_cyclic(3, localities), "11");
    trivial_test_with_policy<double>(length, localities.size(),
        hpx::block_cyclic(localities), "12");
    trivial_test_with_policy<double>(length, 4, hpx::block_cyclic(4, 3), "13");
    trivial_test_with_policy<double>(length, 4,
        hpx::block_cyclic(4, localities, 3), "14");
    trivial_test_with_policy<double>(length, localities.size(),
        hpx::block_cyclic(localities, 3), "15");
}

int main()
{
    trivial_tests<double>();

//    recursive_vector_test();
//    {
//       //Now test move semantics
//       vector<recursive_vector> original;
//       vector<recursive_vector> move_ctor(boost::move(original));
//       vector<recursive_vector> move_assign;
//       move_assign = boost::move(move_ctor);
//       move_assign.swap(original);
//    }
//    typedef vector<int> MyVector;
//    typedef vector<test::movable_int> MyMoveVector;
//    typedef vector<test::movable_and_copyable_int> MyCopyMoveVector;
//    typedef vector<test::copyable_int> MyCopyVector;
//    typedef vector<Test> MyEnumVector;
//
//    if(test::vector_test<MyVector>())
//       return 1;
//    if(test::vector_test<MyMoveVector>())
//       return 1;
//    if(test::vector_test<MyCopyMoveVector>())
//       return 1;
//    if(test::vector_test<MyCopyVector>())
//       return 1;
//    if(test_expand_bwd())
//       return 1;
//    if(!test::default_init_test< vector<int, test::default_init_allocator<int> > >()){
//       std::cerr << "Default init test failed" << std::endl;
//       return 1;
//    }
//
//    MyEnumVector v;
//    Test t;
//    v.push_back(t);
//    v.push_back(::boost::move(t));
//    v.push_back(Test());
//
//    const test::EmplaceOptions Options = (test::EmplaceOptions)(test::EMPLACE_BACK | test::EMPLACE_BEFORE);
//    if(!boost::container::test::test_emplace< vector<test::EmplaceInt>, Options>()){
//       return 1;
//    }
//
//    if(!boost::container::test::test_propagate_allocator<vector>()){
//       return 1;
//    }

   return 0;
}

