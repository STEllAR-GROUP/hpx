//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  This code is based on boost::iterators::counting_iterator
//  Copyright David Abrahams 2003.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/iterator_adaptor.hpp>
#include <hpx/modules/type_support.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace hpx { namespace util {

    template <typename Incrementable, typename CategoryOrTraversal = void,
        typename Difference = void, typename Enable = void>
    class counting_iterator;

    namespace detail {

        template <typename Incrementable, typename CategoryOrTraversal,
            typename Difference>
        struct counting_iterator_base
        {
            // calculate category of the resulting iterator
            template <typename Iterator>
            struct iterator_category
            {
                using type =
                    typename std::iterator_traits<Iterator>::iterator_category;
            };

            using base_traversal =
                util::lazy_conditional<std::is_integral<Incrementable>::value,
                    util::identity<std::random_access_iterator_tag>,
                    iterator_category<Incrementable>>;

            using traversal = typename util::lazy_conditional<
                std::is_void<CategoryOrTraversal>::value, base_traversal,
                util::identity<CategoryOrTraversal>>::type;

            // calculate difference_type of the resulting iterator
            template <typename Integer>
            struct integer_difference_type
            {
                using type = typename std::conditional<
                    (sizeof(Integer) >= sizeof(std::intmax_t)),

                    std::conditional<(std::is_signed<Integer>::value), Integer,
                        std::intmax_t>,

                    std::conditional<(sizeof(Integer) < sizeof(std::ptrdiff_t)),
                        std::ptrdiff_t, std::intmax_t>>::type::type;
            };

            template <typename Iterator>
            struct iterator_difference_type
            {
                using type =
                    typename std::iterator_traits<Iterator>::difference_type;
            };

            using base_difference =
                util::lazy_conditional<std::is_integral<Incrementable>::value,
                    integer_difference_type<Incrementable>,
                    iterator_difference_type<Incrementable>>;

            using difference =
                typename util::lazy_conditional<std::is_void<Difference>::value,
                    base_difference, util::identity<Difference>>::type;

            using type = iterator_adaptor<counting_iterator<Incrementable,
                                              CategoryOrTraversal, Difference>,
                Incrementable, Incrementable, traversal, Incrementable const&,
                difference>;
        };
    }    // namespace detail

    ////////////////////////////////////////////////////////////////////////////
    // specialization for Iterators (non-integral types)
    template <typename Incrementable, typename CategoryOrTraversal,
        typename Difference, typename Enable>
    class counting_iterator
      : public detail::counting_iterator_base<Incrementable,
            CategoryOrTraversal, Difference>::type
    {
    private:
        using base_type = typename detail::counting_iterator_base<Incrementable,
            CategoryOrTraversal, Difference>::type;

        friend class iterator_core_access;

    public:
        counting_iterator() = default;
        counting_iterator(counting_iterator const& rhs) = default;

        explicit counting_iterator(Incrementable x)
          : base_type(x)
        {
        }

    private:
        typename base_type::reference dereference() const
        {
            return this->base_reference();
        }
    };

    template <typename Incrementable, typename CategoryOrTraversal,
        typename Difference>
    class counting_iterator<Incrementable, CategoryOrTraversal, Difference,
        typename std::enable_if<std::is_integral<Incrementable>::value>::type>
      : public detail::counting_iterator_base<Incrementable,
            CategoryOrTraversal, Difference>::type
    {
    private:
        using base_type = typename detail::counting_iterator_base<Incrementable,
            CategoryOrTraversal, Difference>::type;

        friend class iterator_core_access;

    public:
        counting_iterator() = default;
        counting_iterator(counting_iterator const& rhs) = default;

        explicit counting_iterator(Incrementable x)
          : base_type(x)
        {
        }

    private:
        template <typename Iterator>
        bool equal(Iterator const& rhs) const
        {
            return this->base() == rhs.base();
        }

        void increment()
        {
            ++this->base_reference();
        }

        void decrement()
        {
            --this->base_reference();
        }

        template <typename Distance>
        void advance(Distance n)
        {
            this->base_reference() +=
                static_cast<typename base_type::base_type>(n);
        }

        typename base_type::reference dereference() const
        {
            return this->base_reference();
        }

        template <typename OtherIncrementable>
        typename base_type::difference_type distance_to(
            counting_iterator<OtherIncrementable, CategoryOrTraversal,
                Difference> const& y) const
        {
            using difference_type = typename base_type::difference_type;
            return static_cast<difference_type>(y.base()) -
                static_cast<difference_type>(this->base());
        }
    };

    // Manufacture a counting iterator for an arbitrary incrementable type
    template <typename Incrementable>
    inline counting_iterator<Incrementable> make_counting_iterator(
        Incrementable x)
    {
        return counting_iterator<Incrementable>(x);
    }
}}    // namespace hpx::util
