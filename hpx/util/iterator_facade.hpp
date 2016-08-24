//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This code is based on boost::iterators::iterator_facade
//  (C) Copyright David Abrahams 2002.
//  (C) Copyright Jeremy Siek    2002.
//  (C) Copyright Thomas Witt    2002.
//  (C) copyright Jeffrey Lee Hellrung, Jr. 2012.

#if !defined(HPX_UTIL_ITERATOR_FACADE_HPP)
#define HPX_UTIL_ITERATOR_FACADE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>

#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    // Helper class to gain access to the implementation functions in the
    // derived (user-defined) iterator classes.
    class iterator_core_access
    {
    public:
        template <typename Iterator1, typename Iterator2>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static bool equal(Iterator1 const& lhs, Iterator2 const& rhs)
        {
            return lhs.equal(rhs);
        }

        template <typename Iterator>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void increment(Iterator& it)
        {
            it.increment();
        }

        template <typename Iterator>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void decrement(Iterator& it)
        {
            it.decrement();
        }

        template <typename Reference, typename Iterator>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static Reference dereference(Iterator const& it)
        {
            return it.dereference();
        }

        template <typename Iterator, typename Distance>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static void advance(Iterator& it, Distance n)
        {
            it.advance(n);
        }

        template <typename Iterator1, typename Iterator2>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        static typename std::iterator_traits<Iterator1>::difference_type
        distance_to(Iterator1 const& lhs, Iterator2 const& rhs)
        {
            return lhs.distance_to(rhs);
        }
    };

    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Reference>
        struct arrow_dispatch    // proxy references
        {
            struct proxy
            {
                HPX_HOST_DEVICE
                explicit proxy(Reference const& x)
                  : ref_(x)
                {
                }
                HPX_HOST_DEVICE HPX_FORCEINLINE Reference* operator->()
                {
                    return std::addressof(ref_);
                }
                Reference ref_;
            };

            typedef proxy type;

            HPX_HOST_DEVICE HPX_FORCEINLINE
            static type call(Reference const& x)
            {
                return type(x);
            }
        };

        template <typename T>
        struct arrow_dispatch<T&>    // "real" references
        {
            typedef T* type;

            HPX_HOST_DEVICE HPX_FORCEINLINE static type call(T& x)
            {
                return std::addressof(x);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // Implementation for input and forward iterators
        template <typename Derived,
            typename T,
            typename Category,
            typename Reference,
            typename Distance>
        class iterator_facade_base
        {
        public:
            typedef Category iterator_category;
            typedef typename std::remove_const<T>::type value_type;
            typedef Distance difference_type;
            typedef typename arrow_dispatch<Reference>::type pointer;
            typedef Reference reference;

            HPX_HOST_DEVICE iterator_facade_base()
            {
            }

        protected:
            HPX_HOST_DEVICE Derived& derived()
            {
                return *static_cast<Derived*>(this);
            }

            HPX_HOST_DEVICE Derived const& derived() const
            {
                return *static_cast<Derived const*>(this);
            }

        public:
            HPX_HOST_DEVICE reference operator*() const
            {
                return iterator_core_access::
                    template dereference<reference>(this->derived());
            }

            HPX_HOST_DEVICE pointer operator->() const
            {
                return arrow_dispatch<Reference>::call(*this->derived());
            }

            HPX_HOST_DEVICE Derived& operator++()
            {
                Derived& this_ = this->derived();
                iterator_core_access::increment(this_);
                return this_;
            }
        };

        // Implementation for bidirectional iterators
        template <typename Derived,
            typename T,
            typename Reference,
            typename Distance>
        class iterator_facade_base<Derived,
                T,
                std::bidirectional_iterator_tag,
                Reference,
                Distance>
          : public iterator_facade_base<Derived,
                    T,
                    std::forward_iterator_tag,
                    Reference,
                    Distance>
        {
            typedef iterator_facade_base<Derived,
                    T,
                    std::forward_iterator_tag,
                    Reference,
                    Distance
                > base_type;

        public:
            typedef std::bidirectional_iterator_tag iterator_category;
            typedef typename std::remove_const<T>::type value_type;
            typedef Distance difference_type;
            typedef typename arrow_dispatch<Reference>::type pointer;
            typedef Reference reference;

            HPX_HOST_DEVICE iterator_facade_base()
              : base_type()
            {
            }

            HPX_HOST_DEVICE Derived& operator--()
            {
                Derived& this_ = this->derived();
                iterator_core_access::decrement(this_);
                return this_;
            }

            HPX_HOST_DEVICE Derived operator--(int)
            {
                Derived result(this->derived());
                --*this;
                return result;
            }
        };

        // Implementation for random access iterators
        template <typename Derived,
            typename T,
            typename Reference,
            typename Distance>
        class iterator_facade_base<Derived,
                T,
                std::random_access_iterator_tag,
                Reference,
                Distance>
          : public iterator_facade_base<Derived,
                    T,
                    std::bidirectional_iterator_tag,
                    Reference,
                    Distance>
        {
            typedef iterator_facade_base<Derived,
                    T,
                    std::bidirectional_iterator_tag,
                    Reference,
                    Distance
                > base_type;

        public:
            typedef std::random_access_iterator_tag iterator_category;
            typedef typename std::remove_const<T>::type value_type;
            typedef Distance difference_type;
            typedef typename arrow_dispatch<Reference>::type pointer;
            typedef Reference reference;

            HPX_HOST_DEVICE iterator_facade_base()
              : base_type()
            {
            }

            HPX_HOST_DEVICE reference operator[](difference_type n) const
            {
                return *(this->derived() + n);
            }

            HPX_HOST_DEVICE Derived& operator+=(difference_type n)
            {
                Derived& this_ = this->derived();
                iterator_core_access::advance(this_, n);
                return this_;
            }

            HPX_HOST_DEVICE Derived operator+(difference_type n) const
            {
                Derived result(this->derived());
                return result += n;
            }

            HPX_HOST_DEVICE Derived& operator-=(difference_type n)
            {
                Derived& this_ = this->derived();
                iterator_core_access::advance(this_, -n);
                return this_;
            }

            HPX_HOST_DEVICE Derived operator-(difference_type n) const
            {
                Derived result(this->derived());
                return result -= n;
            }
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived,
        typename T,
        typename Category,
        typename Reference = T&,
        typename Distance = std::ptrdiff_t>
    struct iterator_facade
        : detail::
              iterator_facade_base<Derived, T, Category, Reference, Distance>
    {
    private:
        typedef detail::
            iterator_facade_base<Derived, T, Category, Reference, Distance>
                base_type;

    protected:
        // for convenience in derived classes
        typedef iterator_facade<
                Derived, T, Category, Reference, Distance
            > iterator_adaptor_;

    public:
        HPX_HOST_DEVICE iterator_facade()
          : base_type()
        {
        }
    };

    namespace detail
    {
        // Iterators whose dereference operators reference the same value for
        // all iterators into the same sequence (like many input iterators)
        // need help with their postfix ++: the referenced value must be read
        // and stored away before the increment occurs so that *a++ yields the
        // originally referenced element and not the next one.
        template <typename Iterator>
        class postfix_increment_proxy
        {
            typedef
                typename std::iterator_traits<Iterator>::value_type value_type;

        public:
            HPX_HOST_DEVICE explicit postfix_increment_proxy(Iterator const& x)
              : stored_value(*x)
            {
            }

            // Returning a mutable reference allows nonsense like (*r++).mutate(),
            // but it imposes fewer assumptions about the behavior of the
            // value_type. In particular, recall that (*r).mutate() is legal if
            // operator* returns by value.
            HPX_HOST_DEVICE value_type& operator*() const
            {
                return this->stored_value;
            }

        private:
            mutable typename std::remove_const<value_type>::type stored_value;
        };

        // In general, we can't determine that such an iterator isn't writable
        // -- we also need to store a copy of the old iterator so that it can
        // be written into.
        template <typename Iterator>
        class writable_postfix_increment_proxy
        {
            typedef
                typename std::iterator_traits<Iterator>::value_type value_type;

        public:
            HPX_HOST_DEVICE
            explicit writable_postfix_increment_proxy(Iterator const& x)
              : stored_value(*x)
              , stored_iterator(x)
            {
            }

            // Dereferencing must return a proxy so that both *r++ = o and
            // value_type(*r++) can work.  In this case, *r is the same as *r++,
            // and the conversion operator below is used to ensure readability.
            HPX_HOST_DEVICE
            writable_postfix_increment_proxy const& operator*() const
            {
                return *this;
            }

            // Provides readability of *r++
            HPX_HOST_DEVICE operator value_type&() const
            {
                return stored_value;
            }

            // Provides writability of *r++
            template <typename T>
            HPX_HOST_DEVICE T const& operator=(T const& x) const
            {
                *this->stored_iterator = x;
                return x;
            }

            // This overload just in case only non-const objects are writable
            template <typename T>
            HPX_HOST_DEVICE T& operator=(T& x) const
            {
                *this->stored_iterator = x;
                return x;
            }

            // Provides X(r++)
            HPX_HOST_DEVICE operator Iterator const&() const
            {
                return stored_iterator;
            }

        private:
            mutable typename std::remove_const<value_type>::type stored_value;
            Iterator stored_iterator;
        };

        template <typename Reference, typename Value>
        struct is_non_proxy_reference
          : std::is_convertible<
                typename std::remove_reference<Reference>::type const volatile*
              , Value const volatile*
            >
        {};

        // Because the C++98 input iterator requirements say that *r++ has
        // type T (value_type), implementations of some standard algorithms
        // like lexicographical_compare may use constructions like:
        //
        //          *r++ < *s++
        //
        // If *r++ returns a proxy (as required if r is writable but not
        // multi-pass), this sort of expression will fail unless the proxy
        // supports the operator<.  Since there are any number of such
        // operations, we're not going to try to support them.  Therefore,
        // even if r++ returns a proxy, *r++ will only return a proxy if *r
        // also returns a proxy.
        template <typename Iterator, typename Value, typename Reference,
            typename Enable = void>
        struct postfix_increment_result
        {
            typedef Iterator type;
        };

        template <typename Iterator, typename Value, typename Reference>
        struct postfix_increment_result<
            Iterator, Value, Reference,
            typename std::enable_if<
               !traits::is_forward_iterator<Iterator>::value &&
               !traits::is_output_iterator<Iterator>::value &&
                is_non_proxy_reference<Reference, Value>::value
            >::type>
        {
            typedef postfix_increment_proxy<Iterator> type;
        };

        template <typename Iterator, typename Value, typename Reference>
        struct postfix_increment_result<Iterator, Value, Reference,
            typename std::enable_if<
                !traits::is_forward_iterator<Iterator>::value &&
                !traits::is_output_iterator<Iterator>::value &&
                !is_non_proxy_reference<Reference, Value>::value
            >::type>
        {
            typedef writable_postfix_increment_proxy<Iterator> type;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived,
        typename T,
        typename Category,
        typename Reference,
        typename Distance>
    HPX_HOST_DEVICE inline
    typename detail::postfix_increment_result<
        Derived, T, Reference
    >::type
    operator++(iterator_facade<Derived, T, Category, Reference, Distance>& i, int)
    {
        typedef typename detail::postfix_increment_result<
                Derived, T, Reference
            >::type iterator_type;

        iterator_type tmp(*static_cast<Derived*>(&i));
        ++i;
        return tmp;
    }

    namespace detail
    {
        template <typename Facade1, typename Facade2, typename Return>
        struct enable_operator_interoperable
          : std::enable_if<
                std::is_convertible<Facade1, Facade2>::value ||
                    std::is_convertible<Facade2, Facade1>::value,
                Return>
        {};
    }

#define HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(prefix, op, result_type)        \
    template <                                                                \
        typename Derived1, typename T1, typename Category1,                   \
        typename Reference1, typename Distance1,                              \
        typename Derived2, typename T2, typename Category2,                   \
        typename Reference2, typename Distance2>                              \
    HPX_HOST_DEVICE prefix                                                    \
    typename hpx::util::detail::enable_operator_interoperable<                \
        Derived1, Derived2, result_type                                       \
    >::type                                                                   \
    operator op(                                                              \
        iterator_facade<Derived1, T1, Category1, Reference1, Distance1> const& lhs,\
        iterator_facade<Derived2, T2, Category2, Reference2, Distance2> const& rhs)\
/**/

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, ==, bool)
    {
        return iterator_core_access::equal(
            static_cast<Derived1 const&>(lhs),
            static_cast<Derived2 const&>(rhs));
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, !=, bool)
    {
        return !iterator_core_access::equal(
            static_cast<Derived1 const&>(lhs),
            static_cast<Derived2 const&>(rhs));
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, <, bool)
    {
        static_assert(
            hpx::traits::is_random_access_iterator<Derived1>::value,
            "Iterator needs to be random access");
        return 0 <
            iterator_core_access::distance_to(
                static_cast<Derived1 const&>(lhs),
                static_cast<Derived2 const&>(rhs));
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, >, bool)
    {
        static_assert(
            hpx::traits::is_random_access_iterator<Derived1>::value,
            "Iterator needs to be random access");
        return 0 >
            iterator_core_access::distance_to(
                static_cast<Derived1 const&>(lhs),
                static_cast<Derived2 const&>(rhs));
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, <=, bool)
    {
        static_assert(
            hpx::traits::is_random_access_iterator<Derived1>::value,
            "Iterator needs to be random access");
        return 0 <=
            iterator_core_access::distance_to(
                static_cast<Derived1 const&>(lhs),
                static_cast<Derived2 const&>(rhs));
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, >=, bool)
    {
        static_assert(
            hpx::traits::is_random_access_iterator<Derived1>::value,
            "Iterator needs to be random access");
        return 0 >=
            iterator_core_access::distance_to(
                static_cast<Derived1 const&>(lhs),
                static_cast<Derived2 const&>(rhs));
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, -,
        typename std::iterator_traits<Derived2>::difference_type)
    {
        static_assert(
            hpx::traits::is_random_access_iterator<Derived1>::value,
            "Iterator needs to be random access");
        return iterator_core_access::distance_to(
            static_cast<Derived1 const&>(rhs),
            static_cast<Derived2 const&>(lhs));
    }

#undef HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD

    template <typename Derived,
        typename T,
        typename Category,
        typename Reference,
        typename Distance>
    HPX_HOST_DEVICE inline Derived operator+(
        iterator_facade<Derived, T, Category, Reference, Distance> const& it,
        typename Derived::difference_type n)
    {
        Derived tmp(static_cast<Derived const&>(it));
        return tmp += n;
    }

    template <typename Derived,
        typename T,
        typename Category,
        typename Reference,
        typename Distance>
    HPX_HOST_DEVICE inline Derived operator+(
        typename Derived::difference_type n,
        iterator_facade<Derived, T, Category, Reference, Distance> const& it)
    {
        Derived tmp(static_cast<Derived const&>(it));
        return tmp += n;
    }
}}

#endif
