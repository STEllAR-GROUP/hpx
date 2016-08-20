//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// (C) Copyright Ion Gaztanaga 2004-2013.

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/concurrent_vector.hpp>
#include <hpx/util/lightweight_test.hpp>

#include "check_equal_containers.hpp"
#include "movable_int.hpp"
// #include "expand_bwd_test_allocator.hpp"
// #include "expand_bwd_test_template.hpp"
#include "dummy_test_allocator.hpp"
// #include "propagate_allocator_test.hpp"
#include "vector_test.hpp"
// #include "default_init_test.hpp"
// #include "../../intrusive/test/iterator_test.hpp"

#include <iostream>
#include <memory>

//Explicit instantiation to detect compilation errors
template class hpx::concurrent::vector<test::movable_and_copyable_int,
    test::simple_allocator<test::movable_and_copyable_int> >;

template class hpx::concurrent::vector<test::movable_and_copyable_int,
    test::dummy_test_allocator<test::movable_and_copyable_int> >;

template class hpx::concurrent::vector<test::movable_and_copyable_int,
    std::allocator<test::movable_and_copyable_int> >;

// int test_expand_bwd()
// {
//     //Now test all back insertion possibilities
//
//     //First raw ints
//     typedef test::expand_bwd_test_allocator<int> int_allocator_type;
//     typedef vector<int, int_allocator_type> int_vector;
//     if (!test::test_all_expand_bwd<int_vector>())
//         return 1;
//
//     //Now user defined copyable int
//     typedef test::expand_bwd_test_allocator<test::copyable_int>
//         copyable_int_allocator_type;
//     typedef vector<test::copyable_int, copyable_int_allocator_type>
//         copyable_int_vector;
//     if (!test::test_all_expand_bwd<copyable_int_vector>())
//         return 1;
//
//     return 0;
// }

class recursive_vector
{
public:
    int id_;
    hpx::concurrent::vector<recursive_vector> vector_;
    hpx::concurrent::vector<recursive_vector>::iterator it_;
    hpx::concurrent::vector<recursive_vector>::const_iterator cit_;
    hpx::concurrent::vector<recursive_vector>::reverse_iterator rit_;
    hpx::concurrent::vector<recursive_vector>::const_reverse_iterator crit_;
};

void recursive_vector_test()    //Test for recursive types
{
    hpx::concurrent::vector<recursive_vector> recursive_vector_vector;
}

// enum Test
// {
//     zero,
//     one,
//     two,
//     three,
//     four,
//     five,
//     six
// };
//
// template <class VoidAllocator>
// struct GetAllocatorCont
// {
//     template <class ValueType>
//     struct apply
//     {
//         typedef hpx::concurrent::vector<ValueType,
//             typename allocator_traits<VoidAllocator>::
//                 template portable_rebind_alloc<ValueType>::type
//             > type;
//     };
// };
//
// template <class VoidAllocator>
// int test_cont_variants()
// {
//     typedef typename GetAllocatorCont<VoidAllocator>::template apply<int>::type
//         MyCont;
//     typedef typename GetAllocatorCont<VoidAllocator>::template apply<
//         test::movable_int>::type MyMoveCont;
//     typedef typename GetAllocatorCont<VoidAllocator>::template apply<
//         test::movable_and_copyable_int>::type MyCopyMoveCont;
//     typedef typename GetAllocatorCont<VoidAllocator>::template apply<
//         test::copyable_int>::type MyCopyCont;
//
//     if (test::vector_test<MyCont>())
//         return 1;
//     if (test::vector_test<MyMoveCont>())
//         return 1;
//     if (test::vector_test<MyCopyMoveCont>())
//         return 1;
//     if (test::vector_test<MyCopyCont>())
//         return 1;
//
//     return 0;
// }

int hpx_main()
{
    {
        hpx::concurrent::vector<int> vector_int;

        HPX_TEST_EQ(vector_int.size(), std::size_t(0));
        vector_int.push_back(999);
        HPX_TEST_EQ(vector_int.size(), std::size_t(1));
        HPX_TEST_EQ(vector_int[0], 999);
    }
    {
        hpx::concurrent::vector<int> vector_int2(10);
        HPX_TEST_EQ(vector_int2.size(), std::size_t(10));

        for (std::size_t i = 0, max = vector_int2.size(); i != max; ++i)
        {
            vector_int2[i] = (int) i;
        }

        for (std::size_t i = 0, max = vector_int2.size(); i != max; ++i)
        {
            HPX_TEST_EQ(vector_int2[i], (int)i);
        }
    }
//     recursive_vector_test();
//     {
//         //Now test move semantics
//         hpx::concurrent::vector<recursive_vector> original;
//         hpx::concurrent::vector<recursive_vector> move_ctor(boost::move(original));
//         hpx::concurrent::vector<recursive_vector> move_assign;
//         move_assign = boost::move(move_ctor);
//         move_assign.swap(original);
//     }
//
//     ////////////////////////////////////
//     //    Testing allocator implementations
//     ////////////////////////////////////
//     //       std:allocator
//     if (test_cont_variants<std::allocator<void>>())
//     {
//         std::cerr << "test_cont_variants< std::allocator<void> > failed"
//                   << std::endl;
//         return 1;
//     }
//     //       hpx::concurrent::allocator
//     if (test_cont_variants<allocator<void>>())
//     {
//         std::cerr << "test_cont_variants< allocator<void> > failed"
//                   << std::endl;
//         return 1;
//     }
//     //       hpx::concurrent::node_allocator
//     if (test_cont_variants<node_allocator<void>>())
//     {
//         std::cerr << "test_cont_variants< node_allocator<void> > failed"
//                   << std::endl;
//         return 1;
//     }
//     //       hpx::concurrent::adaptive_pool
//     if (test_cont_variants<adaptive_pool<void>>())
//     {
//         std::cerr << "test_cont_variants< adaptive_pool<void> > failed"
//                   << std::endl;
//         return 1;
//     }
//
//     {
//         typedef hpx::concurrent::vector<Test, std::allocator<Test>> MyEnumCont;
//         MyEnumCont v;
//         Test t;
//         v.push_back(t);
//         v.push_back(::boost::move(t));
//         v.push_back(Test());
//     }
//
//     ////////////////////////////////////
//     //    Backwards expansion test
//     ////////////////////////////////////
//     if (test_expand_bwd())
//         return 1;
//
//     ////////////////////////////////////
//     //    Default init test
//     ////////////////////////////////////
//     if (!test::default_init_test<
//             hpx::concurrent::vector<int, test::default_init_allocator<int>>>())
//     {
//         std::cerr << "Default init test failed" << std::endl;
//         return 1;
//     }
//
//     ////////////////////////////////////
//     //    Emplace testing
//     ////////////////////////////////////
//     const test::EmplaceOptions Options =
//         (test::EmplaceOptions)(test::EMPLACE_BACK | test::EMPLACE_BEFORE);
//     if (!hpx::concurrent::test::test_emplace<hpx::concurrent::vector<test::EmplaceInt>,
//             Options>())
//     {
//         return 1;
//     }
//
//     ////////////////////////////////////
//     //    Allocator propagation testing
//     ////////////////////////////////////
//     if (!hpx::concurrent::test::test_propagate_allocator<
//             boost_container_vector>())
//     {
//         return 1;
//     }
//
//     ////////////////////////////////////
//     //    Initializer lists testing
//     ////////////////////////////////////
//     if (!hpx::concurrent::test::
//             test_vector_methods_with_initializer_list_as_argument_for<
//                 hpx::concurrent::vector<int>>())
//     {
//         return 1;
//     }
//
//     ////////////////////////////////////
//     //    Iterator testing
//     ////////////////////////////////////
//     {
//         typedef hpx::concurrent::vector<int> cont_int;
//         cont_int a;
//         a.push_back(0);
//         a.push_back(1);
//         a.push_back(2);
//         boost::intrusive::test::test_iterator_random<cont_int>(a);
//         if (boost::report_errors() != 0)
//         {
//             return 1;
//         }
//     }
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // By default this test should run on all available cores
    std::vector<std::string> cfg;
    cfg.push_back("hpx.os_threads=" +
        std::to_string(hpx::threads::hardware_concurrency()));

    // Initialize and run HPX
    HPX_TEST_EQ_MSG(
        hpx::init(argc, argv, cfg), 0, "HPX main exited with non-zero status");

    return hpx::util::report_errors();
}
