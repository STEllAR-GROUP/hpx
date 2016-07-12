//  Copyright David Abrahams 2001-2004.
//  Copyright (c) Jeremy Siek 2001-2003.
//  Copyright (c) Thomas Witt 2002.
//
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This is really an incomplete test; should be fleshed out.

#include <hpx/hpx_main.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/iterator_facade.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <type_traits>
#include <utility>

// This is a really, really limited test so far.  All we're doing
// right now is checking that the postfix++ proxy for single-pass
// iterators works properly.
template <typename Ref>
class counter_iterator
  : public hpx::util::iterator_facade<
        counter_iterator<Ref>
      , int const
      , std::input_iterator_tag
      , Ref
    >
{
 public:
    counter_iterator() {}
    counter_iterator(int* state) : state(state) {}

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
    proxy(int& x) : state(x) {}

    operator int const&() const
    {
        return state;
    }

    int& operator=(int x) { state = x; return state; }

    int& state;
};

struct value
{
    void mutator() {} // non-const member function
};

struct input_iter
  : hpx::util::iterator_facade<
        input_iter
      , value
      , std::forward_iterator_tag
      , value
    >
{
 public:
    input_iter() {}

    void increment()
    {
    }

    value dereference() const
    {
        return value();
    }

    bool equal(input_iter const& y) const
    {
        return false;
    }
};

template <typename T>
struct wrapper
{
    T m_x;

    template <typename T_>
    explicit wrapper(T_ && x)
        : m_x(std::forward<T_>(x))
    {}

    template <typename U>
    wrapper(const wrapper<U>& other,
            typename std::enable_if<std::is_convertible<U, T>::value>::type* = 0)
      : m_x(other.m_x)
    {}
};

struct iterator_with_proxy_reference
  : hpx::util::iterator_facade<
        iterator_with_proxy_reference
      , wrapper<int>
      , std::forward_iterator_tag
      , wrapper<int&>
    >
{
    int& m_x;

    explicit iterator_with_proxy_reference(int& x)
      : m_x(x)
    {}

    void increment()
    {}

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

template <typename Iterator, typename T>
void readable_iterator_test(const Iterator i1, T v)
{
    typedef typename std::iterator_traits<Iterator>::reference ref_t;

    Iterator i2(i1); // Copy Constructible
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

template <typename Iterator, typename T>
void writable_iterator_traversal_test(Iterator i1, T v, std::true_type)
{
    ++i1;           // we just wrote into that position
    *i1++ = v;

    Iterator x(i1++);
    (void)x;
}

template <class Iterator, class T>
void writable_iterator_traversal_test(const Iterator i1, T v, std::false_type)
{
}

template <class Iterator, class T>
void writable_iterator_test(Iterator i, T v, T v2)
{
    Iterator i2(i); // Copy Constructible
    *i2 = v;

    writable_iterator_traversal_test(
        i, v2,
        typename std::integral_constant<bool,
            traits::is_incrementable<Iterator>::value &&
            traits::is_postfix_incrementable<Iterator>::value
        >());
}

int main()
{
    {
        int state = 0;
        readable_iterator_test(counter_iterator<int const&>(&state), 0);

        state = 3;
        readable_iterator_test(counter_iterator<proxy>(&state), 3);
        writable_iterator_test(counter_iterator<proxy>(&state), 9, 7);

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
        HPX_TEST(x == 0);
        HPX_TEST(i.m_x == 0);
        ++(*i).m_x;
        HPX_TEST(x == 1);
        HPX_TEST(i.m_x == 1);
        ++i->m_x;
        HPX_TEST(x == 2);
        HPX_TEST(i.m_x == 2);
    }

    return hpx::util::report_errors();
}
