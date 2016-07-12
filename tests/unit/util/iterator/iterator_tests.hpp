//  Copyright David Abrahams 2001-2004.
//  Copyright (c) Jeremy Siek 2001-2003.
//  Copyright (c) Thomas Witt 2002.
//
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TESTS_UTIL_ITERATOR_TESTS_HPP)
#define HPX_TESTS_UTIL_ITERATOR_TESTS_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <iterator>
#include <type_traits>

namespace tests
{
    class dummy_constructor {};

    // use this for the value type
    struct dummy_type
    {
        dummy_type()
        {
        }

        dummy_type(dummy_constructor)
        {
        }

        dummy_type(int x)
          : x_(x)
        {
        }

        int foo() const
        {
            return x_;
        }

        bool operator==(const dummy_type& d) const
        {
            return x_ == d.x_;
        }

        int x_;
    };

    ///////////////////////////////////////////////////////////////////////////
    // Tests whether type Iterator satisfies the requirements for a
    // TrivialIterator.
    // Preconditions: i != j, *i == val
    template <class Iterator, class T>
    void trivial_iterator_test(const Iterator i, const Iterator j, T val)
    {
        Iterator k;
        HPX_TEST(i == i);
        HPX_TEST(j == j);
        HPX_TEST(i != j);
        typename std::iterator_traits<Iterator>::value_type v = *i;

        HPX_TEST(v == val);

        k = i;
        HPX_TEST(k == k);
        HPX_TEST(k == i);
        HPX_TEST(k != j);
        HPX_TEST(*k == val);
    }

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
                tests::traits::is_postfix_incrementable<Iterator>::value
            >::type());

        // I think we don't really need this as it checks the same things as
        // the above code.
        HPX_TEST(!hpx::traits::is_output_iterator<Iterator>::value);
    }

    template <typename Iterator, typename T>
    void forward_iterator_test(Iterator i, T v1, T v2)
    {
        input_iterator_test(i, v1, v2);

        Iterator i1 = i, i2 = i;

        HPX_TEST(i == i1++);
        HPX_TEST(i != ++i2);

        trivial_iterator_test(i, i1, v1);
        trivial_iterator_test(i, i2, v1);

        ++i;
        HPX_TEST(i == i1);
        HPX_TEST(i == i2);
        ++i1;
        ++i2;

        trivial_iterator_test(i, i1, v2);
        trivial_iterator_test(i, i2, v2);

//         lvalue_test<(std::is_pointer<Iterator>::value)>::check(i);
    }

    template <typename Iterator, typename T>
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

    // Preconditions: *i == v1, *++i == v2
    template <typename Iterator, typename T>
    void bidirectional_iterator_test(Iterator i, T v1, T v2)
    {
        forward_iterator_test(i, v1, v2);
        ++i;

        Iterator i1 = i, i2 = i;

        HPX_TEST(i == i1--);
        HPX_TEST(i != --i2);

        trivial_iterator_test(i, i1, v2);
        trivial_iterator_test(i, i2, v2);

        --i;
        HPX_TEST(i == i1);
        HPX_TEST(i == i2);
        ++i1;
        ++i2;

        trivial_iterator_test(i, i1, v1);
        trivial_iterator_test(i, i2, v1);
    }

    // bidirectional
    // Preconditions: *i == v1, *++i == v2
    template <typename Iterator, typename T>
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

    namespace detail
    {
        template <typename T>
        struct identity
        {
            typedef T type;
        };

        // implementation originally suggested by C. Green in
        // http://lists.boost.org/MailArchives/boost/msg00886.php

        // The use of identity creates a non-deduced form, so that the
        // explicit template argument must be supplied
        template <typename T>
        inline T implicit_cast(typename identity<T>::type x)
        {
            return x;
        }
    }

    // Preconditions: [i,i+N) is a valid range
    template <typename Iterator, typename TrueVals>
    void random_access_iterator_test(Iterator i, int N, TrueVals vals)
    {
        bidirectional_iterator_test(i, vals[0], vals[1]);
        const Iterator j = i;
        int c;

        typedef typename boost::detail::iterator_traits<Iterator>::value_type
            value_type;

        for (c = 0; c < N - 1; ++c)
        {
            HPX_TEST(i == j + c);
            HPX_TEST(*i == vals[c]);
            HPX_TEST(*i == detail::implicit_cast<value_type>(j[c]));
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
            HPX_TEST(*i == detail::implicit_cast<value_type>(j[N - 1 - c]));
            Iterator q = k - c;
            HPX_TEST(*i == *q);
            HPX_TEST(i > j);
            HPX_TEST(i >= j);
            HPX_TEST(j <= i);
            HPX_TEST(j < i);
            --i;
        }
    }

    // random access
    // Preconditions: [i,i+N) is a valid range
    template <typename Iterator, typename TrueVals>
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
                tests::traits::is_incrementable<Iterator>::value &&
                tests::traits::is_postfix_incrementable<Iterator>::value
            >());
    }

    template <class T>
    class static_object
    {
    public:
        static T& get()
        {
            static char d[sizeof(T)];
            return *reinterpret_cast<T*>(d);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class input_output_iterator_archetype
    {
    private:
        typedef input_output_iterator_archetype self;
        struct in_out_tag
          : public std::input_iterator_tag, public std::output_iterator_tag
        {
        };

    public:
        typedef in_out_tag iterator_category;
        typedef T value_type;
        struct reference
        {
            reference& operator=(const T&)
            {
                return *this;
            }
            operator value_type()
            {
                return static_object<T>::get();
            }
        };

        typedef const T* pointer;
        typedef std::ptrdiff_t difference_type;

        input_output_iterator_archetype()
        {
        }
        self& operator=(const self&)
        {
            return *this;
        }
        bool operator==(const self&) const
        {
            return true;
        }
        bool operator!=(const self&) const
        {
            return true;
        }
        reference operator*() const
        {
            return reference();
        }
        self& operator++()
        {
            return *this;
        }
        self operator++(int)
        {
            return *this;
        }
    };

    template <typename T>
    class input_iterator_archetype_no_proxy
    {
    private:
        typedef input_iterator_archetype_no_proxy self;

    public:
        typedef std::input_iterator_tag iterator_category;
        typedef T value_type;
        typedef const T& reference;
        typedef const T* pointer;
        typedef std::ptrdiff_t difference_type;
        self& operator=(const self&)
        {
            return *this;
        }
        bool operator==(const self&) const
        {
            return true;
        }
        bool operator!=(const self&) const
        {
            return true;
        }
        reference operator*() const
        {
            return static_object<T>::get();
        }
        self& operator++()
        {
            return *this;
        }
        self operator++(int)
        {
            return *this;
        }
    };

    template <typename T>
    class forward_iterator_archetype
    {
    public:
        typedef forward_iterator_archetype self;

    public:
        typedef std::forward_iterator_tag iterator_category;
        typedef T value_type;
        typedef const T& reference;
        typedef T const* pointer;
        typedef std::ptrdiff_t difference_type;
        forward_iterator_archetype()
        {
        }
        self& operator=(const self&)
        {
            return *this;
        }
        bool operator==(const self&) const
        {
            return true;
        }
        bool operator!=(const self&) const
        {
            return true;
        }
        reference operator*() const
        {
            return static_object<T>::get();
        }
        self& operator++()
        {
            return *this;
        }
        self operator++(int)
        {
            return *this;
        }
    };
}

#endif
