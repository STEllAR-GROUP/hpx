//  Copyright (c) 2014 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_STL_TEST_ITERATOR_MAY_29_2014_0110PM)
#define HPX_STL_TEST_ITERATOR_MAY_29_2014_0110PM

#include <boost/iterator/iterator_adaptor.hpp>

namespace test
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseIterator, typename IteratorTag>
    struct test_iterator
      : boost::iterator_adaptor<
            test_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
    {
    private:
        typedef boost::iterator_adaptor<
            test_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
        base_type;

    public:
        test_iterator() : base_type() {}
        test_iterator(BaseIterator base) : base_type(base) {};
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseIterator, typename IteratorTag>
    struct test_bad_alloc_iterator
      : boost::iterator_adaptor<
            test_bad_alloc_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
    {
    private:
        typedef boost::iterator_adaptor<
            test_bad_alloc_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
        base_type;

    public:
        int operator*() const {throw std::bad_alloc();}
        test_bad_alloc_iterator() : base_type() {}
        test_bad_alloc_iterator(BaseIterator base) : base_type(base) {};
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename BaseIterator, typename IteratorTag>
    struct test_runtime_error_iterator
      : boost::iterator_adaptor<
            test_runtime_error_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
    {
    private:
        typedef boost::iterator_adaptor<
            test_runtime_error_iterator<BaseIterator, IteratorTag>,
            BaseIterator, boost::use_default, IteratorTag>
        base_type;

    public:
        int operator*() const {throw std::runtime_error("test");}
        test_runtime_error_iterator() : base_type() {}
        test_runtime_error_iterator(BaseIterator base) : base_type(base) {};
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename IteratorTag>
    struct test_num_exeptions
    {
        static void call(ExPolicy const&, hpx::exception_list const& e)
        {
            // The static partitioner uses the number of threads/cores for the
            // number chunks to create.
            HPX_TEST_EQ(e.size(), hpx::get_num_worker_threads());
        }
    };

    template <typename IteratorTag>
    struct test_num_exeptions<
        hpx::parallel::sequential_execution_policy, IteratorTag>
    {
        static void call(hpx::parallel::sequential_execution_policy const&,
            hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1);
        }
    };

    template <typename ExPolicy>
    struct test_num_exeptions<ExPolicy, std::input_iterator_tag>
    {
        static void call(ExPolicy const&, hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1);
        }
    };

    template <>
    struct test_num_exeptions<
        hpx::parallel::sequential_execution_policy, std::input_iterator_tag>
    {
        static void call(hpx::parallel::sequential_execution_policy const&,
            hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1);
        }
    };

    template <typename IteratorTag>
    struct test_num_exeptions<hpx::parallel::execution_policy, IteratorTag>
    {
        static void call(hpx::parallel::execution_policy const& policy,
            hpx::exception_list const& e)
        {
            using namespace hpx::parallel::detail;

            if (which(policy) == static_cast<int>(execution_policy_enum::sequential)) {
                HPX_TEST_EQ(e.size(), 1);
            }
            else {
                // The static partitioner uses the number of threads/cores for
                // the number chunks to create.
                HPX_TEST_EQ(e.size(), hpx::get_num_worker_threads());
            }
        }
    };

    template <>
    struct test_num_exeptions<
        hpx::parallel::execution_policy, std::input_iterator_tag>
    {
        static void call(hpx::parallel::execution_policy const&,
            hpx::exception_list const& e)
        {
            HPX_TEST_EQ(e.size(), 1);
        }
    };
}

#endif
