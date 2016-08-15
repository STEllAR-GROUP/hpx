//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This code is based on boost::iterators::iterator_facade
// (C) Copyright David Abrahams 2002.
// (C) Copyright Jeremy Siek    2002.
// (C) Copyright Thomas Witt    2002.

#if !defined(HPX_UTIL_ITERATOR_ADAPTOR_HPP)
#define HPX_UTIL_ITERATOR_ADAPTOR_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/iterator_facade.hpp>

#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    // Default template argument handling for iterator_adaptor
    namespace detail
    {
        // A meta-function which computes an iterator_adaptor's base class,
        // a specialization of iterator_facade.
        template <typename Derived, typename Base, typename Value,
            typename Category, typename Reference, typename Difference>
        struct iterator_adaptor_base
        {
            typedef typename std::conditional<
                    std::is_void<Value>::value,
                    typename std::conditional<
                        std::is_void<Reference>::value,
                        typename std::iterator_traits<Base>::value_type,
                        typename std::remove_reference<Reference>::type
                    >::type,
                    Value
                >::type value_type;

            typedef typename std::conditional<
                    std::is_void<Reference>::value,
                    typename std::conditional<
                        std::is_void<Value>::value,
                        typename std::iterator_traits<Base>::reference,
                        typename std::add_lvalue_reference<Value>::type
                    >::type,
                    Reference
                >::type reference_type;

            typedef typename std::conditional<
                    std::is_void<Category>::value,
                    typename std::iterator_traits<Base>::iterator_category,
                    Category
                >::type iterator_category;

            typedef typename std::conditional<
                    std::is_void<Difference>::value,
                    typename std::iterator_traits<Base>::difference_type,
                    Difference
                >::type distance_type;

            typedef iterator_facade<
                    Derived, value_type, iterator_category, reference_type,
                    distance_type
                > type;
        };
    }

    // Iterator adaptor
    //
    // The idea is that when the user needs
    // to fiddle with the reference type it is highly likely that the
    // iterator category has to be adjusted as well.  Any of the
    // following four template arguments may be omitted or explicitly
    // replaced by void.
    //
    //   Value - if supplied, the value_type of the resulting iterator, unless
    //      const. If const, a conforming compiler strips const-ness for the
    //      value_type. If not supplied, iterator_traits<Base>::value_type is used
    //
    //   Category - the traversal category of the resulting iterator. If not
    //      supplied, iterator_traversal<Base>::type is used.
    //
    //   Reference - the reference type of the resulting iterator, and in
    //      particular, the result type of operator*(). If not supplied but
    //      Value is supplied, Value& is used. Otherwise
    //      iterator_traits<Base>::reference is used.
    //
    //   Difference - the difference_type of the resulting iterator. If not
    //      supplied, iterator_traits<Base>::difference_type is used.
    //
    template <
        typename Derived, typename Base, typename Value = void,
        typename Category = void, typename Reference = void,
        typename Difference = void>
    class iterator_adaptor
      : public hpx::util::detail::iterator_adaptor_base<
                Derived, Base, Value, Category, Reference, Difference
            >::type
    {
    protected:
        typedef typename hpx::util::detail::iterator_adaptor_base<
                Derived, Base, Value, Category, Reference, Difference
            >::type base_adaptor_type;

        friend class hpx::util::iterator_core_access;

    public:
        HPX_HOST_DEVICE iterator_adaptor()
        {
        }

        HPX_HOST_DEVICE explicit iterator_adaptor(Base const& iter)
          : iterator_(iter)
        {
        }

        typedef Base base_type;

        HPX_HOST_DEVICE HPX_FORCEINLINE
        Base const& base() const
        {
            return iterator_;
        }

    protected:
        // for convenience in derived classes
        typedef iterator_adaptor<
                Derived, Base, Value, Category, Reference, Difference
            > iterator_adaptor_;

        // lvalue access to the Base object for Derived
        HPX_HOST_DEVICE HPX_FORCEINLINE
        Base const& base_reference() const
        {
            return iterator_;
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE
        Base& base_reference()
        {
            return iterator_;
        }

    private:
        // Core iterator interface for iterator_facade.  This is private
        // to prevent temptation for Derived classes to use it, which
        // will often result in an error.  Derived classes should use
        // base_reference(), above, to get direct access to m_iterator.
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename base_adaptor_type::reference dereference() const
        {
            return *iterator_;
        }

        template <typename OtherDerived, typename OtherIterator, typename V,
            typename C, typename R, typename D>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        bool equal(
            iterator_adaptor<OtherDerived, OtherIterator, V, C, R, D> const& x)
            const
        {
            // Maybe re-add with same_distance
            //  static_assert(
            //      (detail::same_category_and_difference<Derived,OtherDerived>::value)
            //  );
            return iterator_ == x.base();
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE
        void advance(typename base_adaptor_type::difference_type n)
        {
            iterator_ += n;
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE
        void increment()
        {
            ++iterator_;
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE
        void decrement()
        {
            --iterator_;
        }

        template <typename OtherDerived, typename OtherIterator, typename V,
            typename C, typename R, typename D>
        HPX_HOST_DEVICE HPX_FORCEINLINE
        typename base_adaptor_type::difference_type
        distance_to(
            iterator_adaptor<
                OtherDerived, OtherIterator, V, C, R, D
            > const& y) const
        {
            // Maybe re-add with same_distance
            //  static_assert(
            //      (detail::same_category_and_difference<Derived,OtherDerived>::value)
            //  );
            return y.base() - iterator_;
        }

    private:    // data members
        Base iterator_;
    };
}}

#endif
