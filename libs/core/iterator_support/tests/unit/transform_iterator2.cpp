//  (C) Copyright Jeremy Siek 2002.
//  (C) Copyright David Abrahams 2003.
//
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
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

#include <hpx/iterator_support/tests/iterator_tests.hpp>
#include <hpx/modules/iterator_support.hpp>
#include <hpx/modules/testing.hpp>

#include <algorithm>
#include <iterator>
#include <type_traits>
#include <utility>

struct mult_functor
{
    // Functors used with transform_iterator must be
    // DefaultConstructible, as the transform_iterator must be
    // DefaultConstructible to satisfy the requirements for
    // TrivialIterator.
    mult_functor() {}

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
    // Functors used with transform_iterator must be
    // DefaultConstructible, as the transform_iterator must be
    // DefaultConstructible to satisfy the requirements for
    // TrivialIterator.
    adaptable_mult_functor() {}

    adaptable_mult_functor(int aa)
      : mult_functor(aa)
    {
    }
};

struct const_select_first
{
    int const& operator()(std::pair<int, int> const* p) const
    {
        return p->first;
    }
};

struct select_first : const_select_first    // derivation to allow conversions
{
    int& operator()(std::pair<int, int>* p) const
    {
        return p->first;
    }
};

struct select_second
{
    int& operator()(std::pair<int, int>* p) const
    {
        return p->second;
    }
};

struct value_select_first
{
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

        typedef hpx::util::transform_iterator<int*, adaptable_mult_functor>
            iter_t;
        iter_t i(y, adaptable_mult_functor(2));

        tests::input_iterator_test(i, x[0], x[1]);
        tests::input_iterator_test(
            iter_t(&y[0], adaptable_mult_functor(2)), x[0], x[1]);

        tests::random_access_readable_iterator_test(i, N, x);
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
        tests::input_iterator_test(i, x[0], x[1]);
        tests::input_iterator_test(iter_t(&y[0], mult_functor(2)), x[0], x[1]);

        tests::random_access_readable_iterator_test(i, N, x);
    }

    // Test transform_iterator default argument handling
    {
        {
            typedef hpx::util::transform_iterator<int*, adaptable_mult_functor,
                float>
                iter_t;

            HPX_TEST((std::is_same<iter_t::reference, float>::value));
            HPX_TEST((std::is_same<iter_t::value_type, float>::value));
        }

        {
            typedef hpx::util::transform_iterator<int*, adaptable_mult_functor,
                int&, float>
                iter_t;

            HPX_TEST((std::is_same<iter_t::reference, int&>::value));
            HPX_TEST((std::is_same<iter_t::value_type, float>::value));
        }

        {
            typedef hpx::util::transform_iterator<int*, adaptable_mult_functor,
                float, double>
                iter_t;

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

        tests::input_iterator_test(
            hpx::util::transform_iterator(&y[0], &mult_2), x[0], x[1]);

        tests::random_access_readable_iterator_test(
            hpx::util::transform_iterator(&y[0], &mult_2), N, x);
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
            hpx::util::transform_iterator((pair_t*) values, select_first()));

        std::copy(y, y + N,
            hpx::util::transform_iterator((pair_t*) values, select_second()));

        tests::random_access_readable_iterator_test(
            hpx::util::transform_iterator(
                (pair_t*) values, value_select_first()),
            N, x);

        tests::random_access_readable_iterator_test(
            hpx::util::transform_iterator(
                (pair_t*) values, const_select_first()),
            N, x);

        tests::constant_lvalue_iterator_test(
            hpx::util::transform_iterator(
                (pair_t*) values, const_select_first()),
            x[0]);

        tests::non_const_lvalue_iterator_test(
            hpx::util::transform_iterator((pair_t*) values, select_first()),
            x[0], 17);

        tests::const_nonconst_iterator_test(
            ++hpx::util::transform_iterator((pair_t*) values, select_first()),
            hpx::util::transform_iterator(
                (pair_t*) values, const_select_first()));
    }

    // Test transform_iterator with polymorphic object function
    {
        int x[N], y[N];
        for (int k = 0; k < N; ++k)
            x[k] = k;
        std::copy(x, x + N, y);

        for (int k2 = 0; k2 < N; ++k2)
            x[k2] = x[k2] * 2;

        tests::input_iterator_test(
            hpx::util::transform_iterator(&y[0], polymorphic_mult_functor()),
            x[0], x[1]);

        tests::random_access_readable_iterator_test(
            hpx::util::transform_iterator(&y[0], polymorphic_mult_functor()), N,
            x);
    }

    return hpx::util::report_errors();
}
