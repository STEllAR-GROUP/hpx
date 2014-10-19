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

int main()
{
    {
        std::size_t const positions_length = 10;
        std::size_t positions[positions_length] = { 0u };

        hpx::vector<double> v1;
        HPX_TEST_EQ(v1.size(), std::size_t(0));

        hpx::vector<double> v2(positions_length);
        HPX_TEST_EQ(v2.size(), positions_length);

        for(std::size_t i = 0, max = v2.size(); i != max; ++i)
        {
            HPX_TEST_EQ(v2[i], 0.);
            v2[i] = double(i);
            HPX_TEST_EQ(v2[i], double(i));
        }

        hpx::vector<double> v3(positions_length, 999.);
        HPX_TEST_EQ(v3.size(), positions_length);

        hpx::vector<double> v4(positions_length, hpx::block);
        HPX_TEST_EQ(v4.size(), positions_length);

        hpx::vector<double> v5(positions_length, 999., hpx::block);
        HPX_TEST_EQ(v5.size(), positions_length);


//       vector_double.insert(vector_double.begin(), 999.);

//       vector_int.insert_ordered_at(positions_length, positions + positions_length, vector_int2.end());
//
//       for(std::size_t i = 0, max = vector_int.size(); i != max; ++i){
//          std::cout << vector_int[i] << std::endl;
//       }
   }

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

