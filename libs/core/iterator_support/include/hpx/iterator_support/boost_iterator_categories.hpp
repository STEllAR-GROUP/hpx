//  Copyright (c) 2021-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// (C) Copyright Jeremy Siek 2002.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/config/defines.hpp>

#if !defined(                                                                  \
    HPX_ITERATOR_SUPPORT_HAVE_BOOST_ITERATOR_TRAVERSAL_TAG_COMPATIBILITY)

#include <hpx/type_support/identity.hpp>
#include <hpx/type_support/lazy_conditional.hpp>

#include <iterator>
#include <type_traits>

namespace hpx::iterators {

    // Traversal Categories
    struct no_traversal_tag
    {
    };

    struct incrementable_traversal_tag : no_traversal_tag
    {
    };

    struct single_pass_traversal_tag : incrementable_traversal_tag
    {
    };

    struct forward_traversal_tag : single_pass_traversal_tag
    {
    };

    struct bidirectional_traversal_tag : forward_traversal_tag
    {
    };

    struct random_access_traversal_tag : bidirectional_traversal_tag
    {
    };

    namespace detail {

        // Convert a standard iterator category to a traversal tag.
        // This is broken out into a separate meta-function to reduce
        // the cost of instantiating iterator_category_to_traversal, below,
        // for new-style types.
        template <typename Cat>
        struct std_category_to_traversal
          : hpx::util::lazy_conditional<
                std::is_convertible_v<Cat, std::random_access_iterator_tag>,
                hpx::type_identity<random_access_traversal_tag>,
                hpx::util::lazy_conditional<
                    std::is_convertible_v<Cat, std::bidirectional_iterator_tag>,
                    hpx::type_identity<bidirectional_traversal_tag>,
                    hpx::util::lazy_conditional<
                        std::is_convertible_v<Cat, std::forward_iterator_tag>,
                        hpx::type_identity<forward_traversal_tag>,
                        hpx::util::lazy_conditional<
                            std::is_convertible_v<Cat, std::input_iterator_tag>,
                            hpx::type_identity<single_pass_traversal_tag>,
                            hpx::util::lazy_conditional<
                                std::is_convertible_v<Cat,
                                    std::output_iterator_tag>,
                                hpx::type_identity<incrementable_traversal_tag>,
                                hpx::type_identity<no_traversal_tag>>>>>>
        {
        };
    }    // namespace detail

    // Convert an iterator category into a traversal tag
    template <typename Cat>
    struct iterator_category_to_traversal
      : hpx::util::lazy_conditional<
            std::is_convertible_v<Cat, incrementable_traversal_tag>,
            hpx::type_identity<Cat>, detail::std_category_to_traversal<Cat>>
    {
    };

    // Trait to get an iterator's traversal category
    template <typename Iterator>
    struct iterator_traversal
      : iterator_category_to_traversal<
            typename std::iterator_traits<Iterator>::iterator_category>
    {
    };

    // Convert an iterator traversal to one of the traversal tags.
    template <typename Traversal>
    struct pure_traversal_tag
      : hpx::util::lazy_conditional<
            std::is_convertible_v<Traversal, random_access_traversal_tag>,
            hpx::type_identity<random_access_traversal_tag>,
            hpx::util::lazy_conditional<
                std::is_convertible_v<Traversal, bidirectional_traversal_tag>,
                hpx::type_identity<bidirectional_traversal_tag>,
                hpx::util::lazy_conditional<
                    std::is_convertible_v<Traversal, forward_traversal_tag>,
                    hpx::type_identity<forward_traversal_tag>,
                    hpx::util::lazy_conditional<std::is_convertible_v<Traversal,
                                                    single_pass_traversal_tag>,
                        hpx::type_identity<single_pass_traversal_tag>,
                        hpx::util::lazy_conditional<
                            std::is_convertible_v<Traversal,
                                incrementable_traversal_tag>,
                            hpx::type_identity<incrementable_traversal_tag>,
                            hpx::type_identity<no_traversal_tag>>>>>>
    {
    };

    // Trait to retrieve one of the iterator traversal tags from the
    // iterator category or traversal.
    template <typename Iterator>
    struct pure_iterator_traversal
      : pure_traversal_tag<typename iterator_traversal<Iterator>::type>
    {
    };
}    // namespace hpx::iterators

#define HPX_ITERATOR_TRAVERSAL_TAG_NS hpx

#else

#include <boost/iterator/iterator_categories.hpp>

#define HPX_ITERATOR_TRAVERSAL_TAG_NS boost

#endif    // HPX_ITERATOR_SUPPORT_HAVE_BOOST_ITERATOR_TRAVERSAL_TAG_COMPATIBILITY

namespace hpx {

    using HPX_ITERATOR_TRAVERSAL_TAG_NS::iterators::bidirectional_traversal_tag;
    using HPX_ITERATOR_TRAVERSAL_TAG_NS::iterators::forward_traversal_tag;
    using HPX_ITERATOR_TRAVERSAL_TAG_NS::iterators::incrementable_traversal_tag;
    using HPX_ITERATOR_TRAVERSAL_TAG_NS::iterators::no_traversal_tag;
    using HPX_ITERATOR_TRAVERSAL_TAG_NS::iterators::random_access_traversal_tag;
    using HPX_ITERATOR_TRAVERSAL_TAG_NS::iterators::single_pass_traversal_tag;

    namespace traits {

        ///////////////////////////////////////////////////////////////////////
        template <typename Traversal>
        using pure_traversal_tag =
            HPX_ITERATOR_TRAVERSAL_TAG_NS::iterators::pure_traversal_tag<
                Traversal>;

        template <typename Traversal>
        using pure_traversal_tag_t =
            typename pure_traversal_tag<Traversal>::type;

        template <typename Iterator>
        using pure_iterator_traversal =
            HPX_ITERATOR_TRAVERSAL_TAG_NS::iterators::pure_iterator_traversal<
                Iterator>;

        template <typename Iterator>
        using pure_iterator_traversal_t =
            typename pure_iterator_traversal<Iterator>::type;

        ///////////////////////////////////////////////////////////////////////
        template <typename Cat>
        using iterator_category_to_traversal = HPX_ITERATOR_TRAVERSAL_TAG_NS::
            iterators::iterator_category_to_traversal<Cat>;

        template <typename Cat>
        using iterator_category_to_traversal_t =
            typename iterator_category_to_traversal<Cat>::type;

        template <typename Iterator>
        using iterator_traversal =
            HPX_ITERATOR_TRAVERSAL_TAG_NS::iterators::iterator_traversal<
                Iterator>;

        template <typename Iterator>
        using iterator_traversal_t =
            typename iterator_traversal<Iterator>::type;
    }    // namespace traits
}    // namespace hpx

#undef HPX_ITERATOR_TRAVERSAL_TAG_NS
