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

#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        template <typename Reference>
        struct arrow_dispatch    // proxy references
        {
            struct proxy
            {
                explicit proxy(Reference const& x)
                  : m_ref(x)
                {
                }
                Reference* operator->()
                {
                    return std::addressof(m_ref);
                }
                Reference m_ref;
            };

            typedef proxy type;

            static type call(Reference const& x)
            {
                return type(x);
            }
        };

        template <typename T>
        struct arrow_dispatch<T&>    // "real" references
        {
            typedef T* type;

            static type call(T& x)
            {
                return std::addressof(x);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Derived,
            typename T,
            typename Category,
            typename Reference,
            typename Distance>
        class iterator_facade_base;

        // Implementation for forward iterators
        template <typename Derived,
            typename T,
            typename Category,
            typename Reference,
            typename Distance>
        class iterator_facade_base
        {
        public:
            typedef Category iterator_category;
            typedef T value_type;
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
                return this->derived().dereference();
            }

            HPX_HOST_DEVICE pointer operator->() const
            {
                return arrow_dispatch<Reference>::call(*this->derived());
            }

            HPX_HOST_DEVICE Derived& operator++()
            {
                Derived& this_ = this->derived();
                this_.increment();
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
            typedef T value_type;
            typedef Distance difference_type;
            typedef value_type* pointer;
            typedef Reference reference;

            HPX_HOST_DEVICE iterator_facade_base()
              : base_type()
            {
            }

            HPX_HOST_DEVICE Derived& operator--()
            {
                Derived& this_ = this->derived();
                this_.decrement();
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
            typedef T value_type;
            typedef Distance difference_type;
            typedef value_type* pointer;
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
                this_.advance(n);
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
                this_.advance(-n);
                return this_;
            }

            HPX_HOST_DEVICE Derived operator-(difference_type n) const
            {
                Derived result(this->derived());
                return result -= n;
            }
        };
    }

    template <typename Derived,
        typename T,
        typename Category,
        typename Reference,
        typename Distance>
    struct iterator_core_access
        : detail::
              iterator_facade_base<Derived, T, Category, Reference, Distance>
    {
    private:
        typedef detail::
            iterator_facade_base<Derived, T, Category, Reference, Distance>
                base_type;

    public:
        HPX_HOST_DEVICE iterator_core_access()
          : base_type()
        {
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Derived,
        typename T,
        typename Category,
        typename Reference = T&,
        typename Distance = std::ptrdiff_t>
    struct iterator_facade
        : iterator_core_access<Derived, T, Category, Reference, Distance>
    {
    private:
        typedef iterator_core_access<Derived, T, Category, Reference, Distance>
            base_type;

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
            explicit postfix_increment_proxy(Iterator const& x)
              : stored_value(*x)
            {
            }

            // Returning a mutable reference allows nonsense like (*r++).mutate(),
            // but it imposes fewer assumptions about the behavior of the
            // value_type. In particular, recall that (*r).mutate() is legal if
            // operator* returns by value.
            value_type& operator*() const
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
            explicit writable_postfix_increment_proxy(Iterator const& x)
              : stored_value(*x)
              , stored_iterator(x)
            {
            }

            // Dereferencing must return a proxy so that both *r++ = o and
            // value_type(*r++) can work.  In this case, *r is the same as *r++,
            // and the conversion operator below is used to ensure readability.
            writable_postfix_increment_proxy const& operator*() const
            {
                return *this;
            }

            // Provides readability of *r++
            operator value_type&() const
            {
                return stored_value;
            }

            // Provides writability of *r++
            template <typename T>
            T const& operator=(T const& x) const
            {
                *this->stored_iterator = x;
                return x;
            }

            // This overload just in case only non-const objects are writable
            template <typename T>
            T& operator=(T& x) const
            {
                *this->stored_iterator = x;
                return x;
            }

            // Provides X(r++)
            operator Iterator const&() const
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
        // multipass), this sort of expression will fail unless the proxy
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
        struct postfix_increment_result<Iterator, Value, Reference,
                typename std::enable_if<
                    traits::is_input_iterator<Iterator>::value &&
                    is_non_proxy_reference<Reference, Value>::value
                >::type>
        {
            typedef postfix_increment_proxy<Iterator> type;
        };

        template <typename Iterator, typename Value, typename Reference>
        struct postfix_increment_result<Iterator, Value, Reference,
                typename std::enable_if<
                    traits::is_input_iterator<Iterator>::value &&
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

#define HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(prefix, op, result_type)         \
    template <typename Derived,                                                \
        typename Category,                                                     \
        typename T,                                                            \
        typename Distance,                                                     \
        typename Reference>                                                    \
    HPX_HOST_DEVICE prefix result_type operator op(                            \
        iterator_facade<Derived, Category, T, Reference, Distance> const& lhs, \
        iterator_facade<Derived, Category, T, Reference, Distance> const& rhs) \
/**/

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, ==, bool)
    {
        return lhs.equal(rhs);
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, !=, bool)
    {
        return !lhs.equal(rhs);
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, <, bool)
    {
        static_assert(hpx::traits::is_random_access_iterator<Derived>::value,
            "Iterator needs to be random access");
        return 0 > rhs.distance_to(lhs);
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, >, bool)
    {
        static_assert(hpx::traits::is_random_access_iterator<Derived>::value,
            "Iterator needs to be random access");
        return 0 < rhs.distance_to(lhs);
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, <=, bool)
    {
        static_assert(hpx::traits::is_random_access_iterator<Derived>::value,
            "Iterator needs to be random access");
        return 0 >= rhs.distance_to(lhs);
    }

    HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD(inline, >=, bool)
    {
        static_assert(hpx::traits::is_random_access_iterator<Derived>::value,
            "Iterator needs to be random access");
        return 0 <= rhs.distance_to(lhs);
    }

#undef HPX_UTIL_ITERATOR_FACADE_INTEROP_HEAD
}}

#endif
