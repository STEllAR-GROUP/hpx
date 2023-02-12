//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/iterator_support/traits/is_iterator.hpp>

#include <type_traits>
#include <utility>

namespace hpx::traits {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Enable = void>
    struct segmented_iterator_traits
    {
        using is_segmented_iterator = std::false_type;
    };

    template <typename Iterator, typename Enable>
    struct is_segmented_iterator
      : segmented_iterator_traits<Iterator>::is_segmented_iterator
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    // traits allowing to distinguish iterators which have a purely local
    // representation
    template <typename Iterator, typename Enable = void>
    struct segmented_local_iterator_traits
    {
        using is_segmented_local_iterator = std::false_type;

        using iterator = Iterator;
        using local_iterator = Iterator;
        using local_raw_iterator = Iterator;

        static local_raw_iterator const& local(
            local_iterator const& it) noexcept
        {
            return it;
        }

        static local_iterator const& remote(
            local_raw_iterator const& it) noexcept
        {
            return it;
        }

        static local_raw_iterator local(local_iterator&& it) noexcept
        {
            return HPX_MOVE(it);
        }

        static local_iterator remote(local_raw_iterator&& it) noexcept
        {
            return HPX_MOVE(it);
        }
    };

    // MSVC needs this for whatever reason
    template <>
    struct segmented_local_iterator_traits<void>
    {
        using is_segmented_local_iterator = std::false_type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Enable>
    struct is_segmented_local_iterator
      : segmented_local_iterator_traits<Iterator>::is_segmented_local_iterator
    {
    };
}    // namespace hpx::traits
