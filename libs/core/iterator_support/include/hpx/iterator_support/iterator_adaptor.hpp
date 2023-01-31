//  Copyright (c) 2016-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This code is based on boost::iterators::iterator_facade
// (C) Copyright David Abrahams 2002.
// (C) Copyright Jeremy Siek    2002.
// (C) Copyright Thomas Witt    2002.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/iterator_facade.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/identity.hpp>
#include <hpx/type_support/lazy_conditional.hpp>

#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx::util {

    // Default template argument handling for iterator_adaptor
    namespace detail {

        ///////////////////////////////////////////////////////////////////////
        template <typename Iterator>
        struct value_type_iterator_traits_helper
        {
            using type = typename std::iterator_traits<Iterator>::value_type;
        };

        template <typename Iterator>
        struct reference_iterator_traits_helper
        {
            using type = typename std::iterator_traits<Iterator>::reference;
        };

        template <typename Iterator>
        struct category_iterator_traits_helper
        {
            using type =
                typename std::iterator_traits<Iterator>::iterator_category;
        };

        template <typename Iterator>
        struct difference_type_iterator_traits_helper
        {
            using type =
                typename std::iterator_traits<Iterator>::difference_type;
        };

        // A meta-function which computes an iterator_adaptor's base class,
        // a specialization of iterator_facade.
        template <typename Derived, typename Base, typename Value,
            typename Category, typename Reference, typename Difference,
            typename Pointer>
        struct iterator_adaptor_base
        {
            // the following type calculations use lazy_conditional to avoid
            // premature instantiations
            using value_type = std::conditional_t<std::is_void_v<Value>,
                util::lazy_conditional_t<std::is_void_v<Reference>,
                    value_type_iterator_traits_helper<Base>,
                    std::remove_reference<Reference>>,
                Value>;

            using reference_type = std::conditional_t<std::is_void_v<Reference>,
                util::lazy_conditional_t<std::is_void_v<Value>,
                    reference_iterator_traits_helper<Base>,
                    std::add_lvalue_reference<Value>>,
                Reference>;

            using iterator_category =
                util::lazy_conditional_t<std::is_void_v<Category>,
                    category_iterator_traits_helper<Base>,
                    hpx::type_identity<Category>>;

            using difference_type =
                util::lazy_conditional_t<std::is_void_v<Difference>,
                    difference_type_iterator_traits_helper<Base>,
                    hpx::type_identity<Difference>>;

            using type = iterator_facade<Derived, value_type, iterator_category,
                reference_type, difference_type, Pointer>;
        };
    }    // namespace detail

    // Iterator adaptor
    //
    // The idea is that when the user needs to fiddle with the reference type it
    // is highly likely that the iterator category has to be adjusted as well.
    // Any of the following four template arguments may be omitted or explicitly
    // replaced by void.
    //
    //   Value - if supplied, the value_type of the resulting iterator, unless
    //      const. If const, a conforming compiler strips constness for the
    //      value_type. If not supplied, iterator_traits<Base>::value_type is
    //      used
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
    template <typename Derived, typename Base, typename Value = void,
        typename Category = void, typename Reference = void,
        typename Difference = void, typename Pointer = void>
    class iterator_adaptor
      : public hpx::util::detail::iterator_adaptor_base<Derived, Base, Value,
            Category, Reference, Difference, Pointer>::type
    {
    protected:
        using base_adaptor_type =
            typename hpx::util::detail::iterator_adaptor_base<Derived, Base,
                Value, Category, Reference, Difference, Pointer>::type;

        friend class hpx::util::iterator_core_access;

    public:
        HPX_HOST_DEVICE iterator_adaptor() = default;

        HPX_HOST_DEVICE explicit constexpr iterator_adaptor(
            Base const& iter) noexcept
          : iterator_(iter)
        {
        }

        using base_type = Base;

        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Base const& base()
            const noexcept
        {
            return iterator_;
        }

    protected:
        // for convenience in derived classes
        using iterator_adaptor_ = iterator_adaptor<Derived, Base, Value,
            Category, Reference, Difference, Pointer>;

        // lvalue access to the Base object for Derived
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr Base const& base_reference()
            const noexcept
        {
            return iterator_;
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE Base& base_reference() noexcept
        {
            return iterator_;
        }

    private:
        // Core iterator interface for iterator_facade.  This is private
        // to prevent temptation for Derived classes to use it, which
        // will often result in an error.  Derived classes should use
        // base_reference(), above, to get direct access to m_iterator.
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr
            typename base_adaptor_type::reference
            dereference() const noexcept(noexcept(*std::declval<Base>()))
        {
            return *iterator_;
        }

        template <typename OtherDerived, typename OtherIterator, typename V,
            typename C, typename R, typename D, typename P>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr bool equal(
            iterator_adaptor<OtherDerived, OtherIterator, V, C, R, D, P> const&
                x) const
            noexcept(
                noexcept(std::declval<Base>() == std::declval<OtherIterator>()))
        {
            // Maybe re-add with same_distance
            //  static_assert(
            //      (detail::same_category_and_difference<Derived,OtherDerived>::value)
            //  );
            return iterator_ == x.base();
        }

        // prevent this function from being instantiated if not needed
        template <typename DifferenceType>
        HPX_HOST_DEVICE HPX_FORCEINLINE void advance(DifferenceType n) noexcept(
            noexcept(std::advance(
                std::declval<Base&>(), std::declval<DifferenceType>())))
        {
            std::advance(iterator_, n);
        }

        HPX_HOST_DEVICE HPX_FORCEINLINE void increment() noexcept(
            noexcept(++std::declval<Base&>()))
        {
            ++iterator_;
        }

        // prevent this function from being instantiated if not needed
        template <typename Iterator = Base,
            typename Enable =
                std::enable_if_t<traits::is_bidirectional_iterator_v<Iterator>>>
        HPX_HOST_DEVICE HPX_FORCEINLINE void decrement() noexcept(
            noexcept(--std::declval<Base&>()))
        {
            --iterator_;
        }

        template <typename OtherDerived, typename OtherIterator, typename V,
            typename C, typename R, typename D, typename P>
        HPX_HOST_DEVICE HPX_FORCEINLINE constexpr
            typename base_adaptor_type::difference_type
            distance_to(iterator_adaptor<OtherDerived, OtherIterator, V, C, R,
                D, P> const& y) const
            noexcept(noexcept(std::distance(
                std::declval<Base>(), std::declval<OtherIterator>())))
        {
            // Maybe re-add with same_distance
            //  static_assert(
            //      (detail::same_category_and_difference<Derived,OtherDerived>::value)
            //  );
            return std::distance(iterator_, y.base());
        }

    private:    // data members
        Base iterator_;
    };
}    // namespace hpx::util
