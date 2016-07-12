//  (C) Copyright Jeremy Siek 2002.
//  (C) Copyright David Abrahams 2003.
//
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  Revision History
//  22 Nov 2002 Thomas Witt
//       Added interoperability check.
//  28 Oct 2002   Jeremy Siek
//       Updated for new iterator adaptors.
//  08 Mar 2001   Jeremy Siek
//       Moved test of transform iterator into its own file. It to
//       to be in iterator_adaptor_test.cpp.

#include <hpx/hpx_main.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/util/transform_iterator.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>

struct mult_functor
{
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

    template <typename Iterator>
    int operator()(Iterator b) const
    {
        return a * *b;
    }

    int a;
};

struct adaptable_mult_functor : mult_functor
{
    typedef int result_type;
    typedef int argument_type;

    // Functors used with transform_iterator must be
    // DefaultConstructible, as the transform_iterator must be
    // DefaultConstructible to satisfy the requirements for
    // TrivialIterator.
    adaptable_mult_functor()
    {
    }

    adaptable_mult_functor(int aa)
      : mult_functor(aa)
    {
    }
};

struct const_select_first
{
    typedef int const& result_type;

    int const& operator()(std::pair<int, int> const* p) const
    {
        return p->first;
    }
};

struct select_first : const_select_first    // derivation to allow conversions
{
    typedef int& result_type;

    int& operator()(std::pair<int, int>* p) const
    {
        return p->first;
    }
};

struct select_second
{
    typedef int& result_type;

    int& operator()(std::pair<int, int>* p) const
    {
        return p->second;
    }
};

struct value_select_first
{
    typedef int result_type;

    int operator()(std::pair<int, int> const* p) const
    {
        return p->first;
    }
};

int mult_2(int* arg)
{
    return *arg * 2;
}

struct polymorphic_mult_functor
{
    template <typename T>
    T operator()(const T* _arg) const
    {
        return *_arg * 2;
    }

    template <typename T>
    T operator()(const T* _arg)
    {
        HPX_TEST(false);
        return *_arg * 2;
    }
};

///////////////////////////////////////////////////////////////////////////////
// Preconditions: *i == v1, *++i == v2
template <typename Iterator, typename T>
void input_iterator_test(Iterator i, T v1, T v2)
{
    Iterator i1(i);

    HPX_TEST(i == i1);
    HPX_TEST(!(i != i1));

    // I can see no generic way to create an input iterator
    // that is in the domain of== of i and != i.
    // The following works for istream_iterator but is not
    // guaranteed to work for arbitrary input iterators.
    //
    //   Iterator i2;
    //
    //   HPX_TEST(i != i2);
    //   HPX_TEST(!(i == i2));

    HPX_TEST(*i1 == v1);
    HPX_TEST(*i == v1);

    // we cannot test for equivalence of (void)++i & (void)i++
    // as i is only guaranteed to be single pass.
    HPX_TEST(*i++ == v1);

    i1 = i;

    HPX_TEST(i == i1);
    HPX_TEST(!(i != i1));

    HPX_TEST(*i1 == v2);
    HPX_TEST(*i == v2);

    // i is dereferencable, so it must be incrementable.
    ++i;

    // how to test for operator-> ?
}

namespace traits
{
    template <typename T, typename Enable = void>
    struct is_incrementable
      : std::false_type
    {};

    template <typename T>
    struct is_incrementable<T,
            typename hpx::util::always_void<
                decltype(++std::declval<T&>())
            >::type>
      : std::true_type
    {};

    template <typename T, typename Enable = void>
    struct is_postfix_incrementable
      : std::false_type
    {};

    template <typename T>
    struct is_postfix_incrementable<T,
            typename hpx::util::always_void<
                decltype(std::declval<T&>()++)
            >::type>
      : std::true_type
    {};
}

// Preconditions: *i == v
// Do separate tests for *i++ so we can treat, e.g., smart pointers,
// as readable and/or writable iterators.
template <typename Iterator, typename T>
void readable_iterator_traversal_test(Iterator i1, T v, std::true_type)
{
    T v2(*i1++);
    HPX_TEST(v == v2);
}

template <typename Iterator, typename T>
void readable_iterator_traversal_test(const Iterator i1, T v, std::false_type)
{
}

// Preconditions: *i == v
template <class Iterator, class T>
void readable_iterator_test(const Iterator i1, T v)
{
    Iterator i2(i1);    // Copy Constructible
    typedef typename std::iterator_traits<Iterator>::reference ref_t;
    ref_t r1 = *i1;
    ref_t r2 = *i2;
    T v1 = r1;
    T v2 = r2;
    HPX_TEST(v1 == v);
    HPX_TEST(v2 == v);

    readable_iterator_traversal_test(
        i1, v,
        typename std::integral_constant<bool,
            traits::is_postfix_incrementable<Iterator>::value
        >::type());

    // I think we don't really need this as it checks the same things as
    // the above code.
    HPX_TEST(!hpx::traits::is_output_iterator<Iterator>::value);
}

template <class Iterator, class T>
void forward_readable_iterator_test(Iterator i, Iterator j, T val1, T val2)
{
    Iterator i2;
    Iterator i3(i);
    i2 = i;
    HPX_TEST(i2 == i3);
    HPX_TEST(i != j);
    HPX_TEST(i2 != j);
    readable_iterator_test(i, val1);
    readable_iterator_test(i2, val1);
    readable_iterator_test(i3, val1);

    HPX_TEST(i == i2++);
    HPX_TEST(i != ++i3);

    readable_iterator_test(i2, val2);
    readable_iterator_test(i3, val2);

    readable_iterator_test(i, val1);
}

// bidirectional
// Preconditions: *i == v1, *++i == v2
template <class Iterator, class T>
void bidirectional_readable_iterator_test(Iterator i, T v1, T v2)
{
    Iterator j(i);
    ++j;
    forward_readable_iterator_test(i, j, v1, v2);
    ++i;

    Iterator i1 = i, i2 = i;

    HPX_TEST(i == i1--);
    HPX_TEST(i != --i2);

    readable_iterator_test(i, v2);
    readable_iterator_test(i1, v1);
    readable_iterator_test(i2, v1);

    --i;
    HPX_TEST(i == i1);
    HPX_TEST(i == i2);
    ++i1;
    ++i2;

    readable_iterator_test(i, v1);
    readable_iterator_test(i1, v2);
    readable_iterator_test(i2, v2);
}

// random access
// Preconditions: [i,i+N) is a valid range
template <class Iterator, class TrueVals>
void random_access_readable_iterator_test(Iterator i, int N, TrueVals vals)
{
    bidirectional_readable_iterator_test(i, vals[0], vals[1]);
    const Iterator j = i;
    int c;

    for (c = 0; c < N - 1; ++c)
    {
        HPX_TEST(i == j + c);
        HPX_TEST(*i == vals[c]);
        typename std::iterator_traits<Iterator>::value_type x = j[c];
        HPX_TEST(*i == x);
        HPX_TEST(*i == *(j + c));
        HPX_TEST(*i == *(c + j));
        ++i;
        HPX_TEST(i > j);
        HPX_TEST(i >= j);
        HPX_TEST(j <= i);
        HPX_TEST(j < i);
    }

    Iterator k = j + N - 1;
    for (c = 0; c < N - 1; ++c)
    {
        HPX_TEST(i == k - c);
        HPX_TEST(*i == vals[N - 1 - c]);
        typename std::iterator_traits<Iterator>::value_type x = j[N - 1 - c];
        HPX_TEST(*i == x);
        Iterator q = k - c;
        HPX_TEST(*i == *q);
        HPX_TEST(i > j);
        HPX_TEST(i >= j);
        HPX_TEST(j <= i);
        HPX_TEST(j < i);
        --i;
    }
}

template <typename Iterator, typename T>
void constant_lvalue_iterator_test(Iterator i, T v1)
{
    Iterator i2(i);
    typedef typename std::iterator_traits<Iterator>::value_type value_type;
    typedef typename std::iterator_traits<Iterator>::reference reference;
    HPX_TEST((std::is_same<const value_type&, reference>::value));
    const T& v2 = *i2;
    HPX_TEST(v1 == v2);
//     HPX_TEST(is_lvalue_iterator<Iterator>::value);
//     HPX_TEST(!is_non_const_lvalue_iterator<Iterator>::value);
}

template <typename Iterator, typename T>
void non_const_lvalue_iterator_test(Iterator i, T v1, T v2)
{
    Iterator i2(i);
    typedef typename std::iterator_traits<Iterator>::value_type value_type;
    typedef typename std::iterator_traits<Iterator>::reference reference;
    HPX_TEST((std::is_same<value_type&, reference>::value));
    T& v3 = *i2;
    HPX_TEST(v1 == v3);

    // A non-const lvalue iterator is not necessarily writable, but we
    // are assuming the value_type is assignable here
    *i = v2;

    T& v4 = *i2;
    HPX_TEST(v2 == v4);
//     HPX_TEST(is_lvalue_iterator<Iterator>::value);
//     HPX_TEST(is_non_const_lvalue_iterator<Iterator>::value);
}

// Precondition: i != j
template <typename Iterator, typename ConstIterator>
void const_nonconst_iterator_test(Iterator i, ConstIterator j)
{
    HPX_TEST(i != j);
    HPX_TEST(j != i);

    ConstIterator k(i);
    HPX_TEST(k == i);
    HPX_TEST(i == k);

    k = i;
    HPX_TEST(k == i);
    HPX_TEST(i == k);
}

///////////////////////////////////////////////////////////////////////////////
int main()
{
    const int N = 10;

    // Test transform_iterator
    {
        int x[N], y[N];
        for (int k = 0; k < N; ++k)
            x[k] = k;
        std::copy(x, x + N, y);

        for (int k2 = 0; k2 < N; ++k2)
            x[k2] = x[k2] * 2;

        typedef hpx::util::transform_iterator<int*, adaptable_mult_functor> iter_t;
        iter_t i(y, adaptable_mult_functor(2));
        input_iterator_test(i, x[0], x[1]);
        input_iterator_test(
            iter_t(&y[0], adaptable_mult_functor(2)), x[0], x[1]);

        random_access_readable_iterator_test(i, N, x);
    }

    // Test transform_iterator non adaptable functor
    {
        int x[N], y[N];
        for (int k = 0; k < N; ++k)
            x[k] = k;
        std::copy(x, x + N, y);

        for (int k2 = 0; k2 < N; ++k2)
            x[k2] = x[k2] * 2;

        typedef hpx::util::transform_iterator<int*, mult_functor, int> iter_t;
        iter_t i(y, mult_functor(2));
        input_iterator_test(i, x[0], x[1]);
        input_iterator_test(iter_t(&y[0], mult_functor(2)), x[0], x[1]);

        random_access_readable_iterator_test(i, N, x);
    }

    // Test transform_iterator default argument handling
    {
        {
            typedef hpx::util::transform_iterator<int*, adaptable_mult_functor, float>
                    iter_t;
            HPX_TEST((std::is_same<iter_t::reference, float>::value));
            HPX_TEST((std::is_same<iter_t::value_type, float>::value));
        }

        {
            typedef hpx::util::transform_iterator<
                    int*, adaptable_mult_functor, int&, float
                > iter_t;

            HPX_TEST((std::is_same<iter_t::reference, int&>::value));
            HPX_TEST((std::is_same<iter_t::value_type, float>::value));
        }

        {
            typedef hpx::util::transform_iterator<
                    int*, adaptable_mult_functor, float, double
                > iter_t;

            HPX_TEST((std::is_same<iter_t::reference, float>::value));
            HPX_TEST((std::is_same<iter_t::value_type, double>::value));
        }
    }

    // Test transform_iterator with function pointers
    {
        int x[N], y[N];
        for (int k = 0; k < N; ++k)
            x[k] = k;
        std::copy(x, x + N, y);

        for (int k2 = 0; k2 < N; ++k2)
            x[k2] = x[k2] * 2;

        input_iterator_test(
            hpx::util::make_transform_iterator(&y[0], &mult_2), x[0], x[1]);

        random_access_readable_iterator_test(
            hpx::util::make_transform_iterator(&y[0], &mult_2), N, x);
    }

    // Test transform_iterator as projection iterator
    {
        typedef std::pair<int, int> pair_t;

        int x[N];
        int y[N];
        pair_t values[N];

        for (int i = 0; i < N; ++i)
        {
            x[i] = i;
            y[i] = N - (i + 1);
        }

        std::copy(x, x + N,
            hpx::util::make_transform_iterator((pair_t*) values, select_first()));

        std::copy(y, y + N,
            hpx::util::make_transform_iterator((pair_t*) values, select_second()));

        random_access_readable_iterator_test(
            hpx::util::make_transform_iterator((pair_t*) values, value_select_first()),
            N, x);

        random_access_readable_iterator_test(
            hpx::util::make_transform_iterator((pair_t*) values, const_select_first()),
            N, x);

        constant_lvalue_iterator_test(
            hpx::util::make_transform_iterator((pair_t*) values, const_select_first()),
            x[0]);

        non_const_lvalue_iterator_test(
            hpx::util::make_transform_iterator((pair_t*) values, select_first()), x[0],
            17);

        const_nonconst_iterator_test(
            ++hpx::util::make_transform_iterator((pair_t*) values, select_first()),
            hpx::util::make_transform_iterator((pair_t*) values, const_select_first()));
    }

    // Test transform_iterator with polymorphic object function
    {
        int x[N], y[N];
        for (int k = 0; k < N; ++k)
            x[k] = k;
        std::copy(x, x + N, y);

        for (int k2 = 0; k2 < N; ++k2)
            x[k2] = x[k2] * 2;

        input_iterator_test(
            hpx::util::make_transform_iterator(&y[0], polymorphic_mult_functor()),
            x[0], x[1]);

        random_access_readable_iterator_test(
            hpx::util::make_transform_iterator(&y[0], polymorphic_mult_functor()),
            N, x);
    }

    return hpx::util::report_errors();
}
