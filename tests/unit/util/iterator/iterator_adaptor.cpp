//  (C) Copyright Thomas Witt 2003.
//
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_main.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/tuple.hpp>
#include <hpx/util/iterator_adaptor.hpp>

#include "iterator_tests.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <functional>
#include <list>
#include <numeric>
#include <set>
#include <type_traits>
#include <vector>

struct mult_functor
{
    typedef int result_type;
    typedef int argument_type;

    // Functors used with transform_iterator must be
    // DefaultConstructible, as the transform_iterator must be
    // DefaultConstructible to satisfy the requirements for
    // TrivialIterator.
    mult_functor()
    {
    }
    mult_functor(int aa)
      : a(aa)
    {
    }

    int operator()(int b) const
    {
        return a * b;
    }

    int a;
};

template <typename Pair>
struct select1st_ : public std::unary_function<Pair, typename Pair::first_type>
{
    const typename Pair::first_type& operator()(const Pair& x) const
    {
        return x.first;
    }

    typename Pair::first_type& operator()(Pair& x) const
    {
        return x.first;
    }
};

struct one_or_four
{
    bool operator()(tests::dummy_type x) const
    {
        return x.foo() == 1 || x.foo() == 4;
    }
};

typedef std::deque<int> storage;
typedef std::deque<int*> pointer_deque;
typedef std::set<storage::iterator> iterator_set;

template <class T>
struct foo;

void blah(int)
{
}

struct my_gen
{
    typedef int result_type;

    my_gen()
      : n(0)
    {
    }

    int operator()()
    {
        return ++n;
    }

    int n;
};

template <typename V>
struct ptr_iterator
  : hpx::util::iterator_adaptor<
        ptr_iterator<V>, V*, V, std::random_access_iterator_tag>
{
private:
    typedef hpx::util::iterator_adaptor<
            ptr_iterator<V>, V*, V, std::random_access_iterator_tag
        > base_adaptor_type;

public:
    ptr_iterator()
    {
    }

    ptr_iterator(V* d)
      : base_adaptor_type(d)
    {
    }

    template <typename V2>
    ptr_iterator(const ptr_iterator<V2>& x,
            typename std::enable_if<
                std::is_convertible<V2*, V*>::value
            >::type* = 0)
      : base_adaptor_type(x.base())
    {
    }
};

// Non-functional iterator for category modification checking
template <typename Iter, typename Category>
struct modify_traversal
  : hpx::util::iterator_adaptor<
        modify_traversal<Iter, Category>, Iter, void, Category>
{
};

template <typename T>
struct fwd_iterator
  : hpx::util::iterator_adaptor<
        fwd_iterator<T>, tests::forward_iterator_archetype<T> >
{
private:
    typedef hpx::util::iterator_adaptor<
            fwd_iterator<T>, tests::forward_iterator_archetype<T>
        > base_adaptor_type;

public:
    fwd_iterator()
    {
    }

    fwd_iterator(tests::forward_iterator_archetype<T> d)
      : base_adaptor_type(d)
    {
    }
};

template <typename T>
struct in_iterator
  : hpx::util::iterator_adaptor<
        in_iterator<T>, tests::input_iterator_archetype_no_proxy<T> >
{
private:
    typedef hpx::util::iterator_adaptor<
            in_iterator<T>, tests::input_iterator_archetype_no_proxy<T>
        > base_adaptor_type;

public:
    in_iterator()
    {
    }
    in_iterator(tests::input_iterator_archetype_no_proxy<T> d)
      : base_adaptor_type(d)
    {
    }
};

template <typename Iter>
struct constant_iterator
    : hpx::util::iterator_adaptor<
            constant_iterator<Iter>, Iter,
            typename std::iterator_traits<Iter>::value_type const
      >
{
    typedef hpx::util::iterator_adaptor<
            constant_iterator<Iter>, Iter,
            typename std::iterator_traits<Iter>::value_type const
        > base_adaptor_type;

    constant_iterator()
    {
    }
    constant_iterator(Iter it)
      : base_adaptor_type(it)
    {
    }
};

int main()
{
    tests::dummy_type array[] = {
        tests::dummy_type(0), tests::dummy_type(1), tests::dummy_type(2),
        tests::dummy_type(3), tests::dummy_type(4), tests::dummy_type(5)
    };
    const int N = sizeof(array) / sizeof(tests::dummy_type);

    // sanity check, if this doesn't pass the test is buggy
    tests::random_access_iterator_test(array, N, array);

    // Test the iterator_adaptor
    {
        ptr_iterator<tests::dummy_type> i(array);
        tests::random_access_iterator_test(i, N, array);

        ptr_iterator<const tests::dummy_type> j(array);
        tests::random_access_iterator_test(j, N, array);
        tests::const_nonconst_iterator_test(i, ++j);
    }

    // Test the iterator_traits
    {
        // Test computation of defaults
        typedef ptr_iterator<int> Iter1;

        // don't use std::iterator_traits here to avoid VC++ problems
        HPX_TEST((std::is_same<Iter1::value_type, int>::value));
        HPX_TEST((std::is_same<Iter1::reference, int&>::value));
        HPX_TEST((std::is_same<Iter1::pointer, int*>::value));
        HPX_TEST((std::is_same<Iter1::difference_type, std::ptrdiff_t>::value));

        HPX_TEST((std::is_convertible<Iter1::iterator_category,
            std::random_access_iterator_tag>::value));
    }

    {
        // Test computation of default when the Value is const
        typedef ptr_iterator<int const> Iter1;
        HPX_TEST((std::is_same<Iter1::value_type, int>::value));
        HPX_TEST((std::is_same<Iter1::reference, const int&>::value));

//         HPX_TEST(boost::is_readable_iterator<Iter1>::value);
//         HPX_TEST(boost::is_lvalue_iterator<Iter1>::value);

        HPX_TEST((std::is_same<Iter1::pointer, int const*>::value));
    }

    {
        // Test constant iterator idiom
        typedef ptr_iterator<int> BaseIter;
        typedef constant_iterator<BaseIter> Iter;

        Iter it;

        HPX_TEST((std::is_same<Iter::value_type, int>::value));
        HPX_TEST((std::is_same<Iter::reference, int const&>::value));
        HPX_TEST((std::is_same<Iter::pointer, int const*>::value));

//         HPX_TEST(boost::is_non_const_lvalue_iterator<BaseIter>::value);
//         HPX_TEST(boost::is_lvalue_iterator<Iter>::value);

        typedef modify_traversal<BaseIter, std::input_iterator_tag>
            IncrementableIter;

        HPX_TEST((std::is_same<
                BaseIter::iterator_category, std::random_access_iterator_tag
            >::value));
        HPX_TEST((std::is_same<
                IncrementableIter::iterator_category, std::input_iterator_tag
            >::value));
    }

    // Test the iterator_adaptor
    {
        ptr_iterator<tests::dummy_type> i(array);
        tests::random_access_iterator_test(i, N, array);

        ptr_iterator<const tests::dummy_type> j(array);
        tests::random_access_iterator_test(j, N, array);
        tests::const_nonconst_iterator_test(i, ++j);
    }

    // check operator-> with a forward iterator
    {
        tests::forward_iterator_archetype<tests::dummy_type> forward_iter;

        typedef fwd_iterator<tests::dummy_type> adaptor_type;

        adaptor_type i(forward_iter);
        int zero = 0;
        if (zero)    // don't do this, just make sure it compiles
        {
            HPX_TEST((*i).x_ == i->foo());
        }
    }

    // check operator-> with an input iterator
    {
        tests::input_iterator_archetype_no_proxy<tests::dummy_type> input_iter;
        typedef in_iterator<tests::dummy_type> adaptor_type;
        adaptor_type i(input_iter);
        int zero = 0;
        if (zero)    // don't do this, just make sure it compiles
        {
            HPX_TEST((*i).x_ == i->foo());
        }
    }

    // check that base_type is correct
    {
        // Test constant iterator idiom
        typedef ptr_iterator<int> BaseIter;

        HPX_TEST((std::is_same<BaseIter::base_type, int*>::value));
        HPX_TEST((std::is_same<
                constant_iterator<BaseIter>::base_type,
                BaseIter
            >::value));

        typedef modify_traversal<BaseIter, std::forward_iterator_tag>
            IncrementableIter;

        HPX_TEST((std::is_same<IncrementableIter::base_type, BaseIter>::value));
    }

    return hpx::util::report_errors();
}
