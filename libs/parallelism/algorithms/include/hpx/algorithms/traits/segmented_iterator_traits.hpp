//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/type_support/decay.hpp>

#include <type_traits>
#include <utility>

namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Enable = void>
    struct segmented_iterator_traits
    {
        typedef std::false_type is_segmented_iterator;
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
        typedef std::false_type is_segmented_local_iterator;

        typedef Iterator iterator;
        typedef Iterator local_iterator;
        typedef Iterator local_raw_iterator;

        static local_raw_iterator const& local(local_iterator const& it)
        {
            return it;
        }

        static local_iterator const& remote(local_raw_iterator const& it)
        {
            return it;
        }

        static local_raw_iterator local(local_iterator&& it)
        {
            return std::move(it);
        }

        static local_iterator remote(local_raw_iterator&& it)
        {
            return std::move(it);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename Iterator, typename Enable>
    struct is_segmented_local_iterator
      : segmented_local_iterator_traits<Iterator>::is_segmented_local_iterator
    {
    };
}}    // namespace hpx::traits
