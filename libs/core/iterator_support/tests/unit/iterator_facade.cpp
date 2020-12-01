//  Copyright David Abrahams 2001-2004.
//  Copyright (c) Jeremy Siek 2001-2003.
//  Copyright (c) Thomas Witt 2002.
//
//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is really an incomplete test; should be fleshed out.

#include <hpx/hpx_main.hpp>
#include <hpx/iterator_support/iterator_facade.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/modules/testing.hpp>
#include <hpx/type_support/always_void.hpp>

#include "iterator_tests.hpp"

#include <type_traits>
#include <utility>

// This is a really, really limited test so far.  All we're doing
// right now is checking that the postfix++ proxy for single-pass
// iterators works properly.
template <typename Ref>
class counter_iterator
  : public hpx::util::iterator_facade<counter_iterator<Ref>, int const,
        std::input_iterator_tag, Ref>
{
public:
    counter_iterator() {}
    counter_iterator(int* state)
      : state(state)
    {
    }

    void increment()
    {
        ++*state;
    }

    Ref dereference() const
    {
        return *state;
    }

    bool equal(counter_iterator const& y) const
    {
        return *this->state == *y.state;
    }

    int* state;
};

struct proxy
{
    proxy(proxy const& x) = default;

    proxy(int& x)
      : state(x)
    {
    }

    operator int const&() const
    {
        return state;
    }

    int& operator=(int x)
    {
        state = x;
        return state;
    }

    proxy& operator=(proxy const& other)
    {
        state = other.state;
        return *this;
    }

    int& state;
};

struct value
{
    void mutator() {}    // non-const member function
};

struct input_iter
  : hpx::util::iterator_facade<input_iter, value, std::forward_iterator_tag,
        value>
{
public:
    input_iter() {}

    void increment() {}

    value dereference() const
    {
        return value();
    }

    bool equal(input_iter const&) const
    {
        return false;
    }
};

template <typename T>
struct wrapper
{
    T m_x;

    template <typename T_, typename TD = typename std::decay<T_>::type,
        typename Enable =
            typename std::enable_if<!std::is_same<TD, wrapper<T>>::value>::type>
    explicit wrapper(T_&& x)
      : m_x(std::forward<T_>(x))
    {
    }

    template <typename U>
    wrapper(const wrapper<U>& other,
        typename std::enable_if<std::is_convertible<U, T>::value>::type* = 0)
      : m_x(other.m_x)
    {
    }
};

struct iterator_with_proxy_reference
  : hpx::util::iterator_facade<iterator_with_proxy_reference, wrapper<int>,
        std::forward_iterator_tag, wrapper<int&>>
{
    int& m_x;

    explicit iterator_with_proxy_reference(int& x)
      : m_x(x)
    {
    }

    void increment() {}

    reference dereference() const
    {
        return wrapper<int&>(m_x);
    }
};

template <typename T, typename U>
void same_type(U const&)
{
    HPX_TEST((std::is_same<T, U>::value));
}

int main()
{
    {
        int state = 0;
        tests::readable_iterator_test(counter_iterator<int const&>(&state), 0);

        state = 3;
        tests::readable_iterator_test(counter_iterator<proxy>(&state), 3);
        tests::writable_iterator_test(counter_iterator<proxy>(&state), 9, 7);

        HPX_TEST_EQ(state, 8);
    }

    {
        // These two lines should be equivalent (and both compile)
        input_iter p;
        (*p).mutator();
        p->mutator();

        same_type<input_iter::pointer>(p.operator->());
    }

    {
        int x = 0;
        iterator_with_proxy_reference i(x);
        HPX_TEST_EQ(x, 0);
        HPX_TEST_EQ(i.m_x, 0);
        ++(*i).m_x;
        HPX_TEST_EQ(x, 1);
        HPX_TEST_EQ(i.m_x, 1);
        ++i->m_x;
        HPX_TEST_EQ(x, 2);
        HPX_TEST_EQ(i.m_x, 2);
    }

    return hpx::util::report_errors();
}
