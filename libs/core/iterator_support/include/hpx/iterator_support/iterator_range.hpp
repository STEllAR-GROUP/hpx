//  Copyright (c) 2017 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/iterator_support/range.hpp>
#include <hpx/iterator_support/traits/is_iterator.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    template <typename Iterator, typename Sentinel = Iterator>
    class iterator_range
    {
    public:
        iterator_range() = default;

        constexpr iterator_range(Iterator iterator, Sentinel sentinel) noexcept
          : _iterator(HPX_MOVE(iterator))
          , _sentinel(HPX_MOVE(sentinel))
        {
        }

        constexpr Iterator begin() const
        {
            return _iterator;
        }

        constexpr Iterator end() const
        {
            return _sentinel;
        }

        constexpr std::ptrdiff_t size() const
        {
            return std::distance(_iterator, _sentinel);
        }

        constexpr bool empty() const
        {
            return _iterator == _sentinel;
        }

    private:
        Iterator _iterator;
        Sentinel _sentinel;
    };

    template <typename Range,
        typename Iterator = traits::range_iterator_t<Range>,
        typename Sentinel = traits::range_iterator_t<Range>>
    constexpr std::enable_if_t<traits::is_range_v<Range>,
        iterator_range<Iterator, Sentinel>>
    make_iterator_range(Range& r)
    {
        return iterator_range<Iterator, Sentinel>(util::begin(r), util::end(r));
    }

    template <typename Range,
        typename Iterator = traits::range_iterator_t<Range const>,
        typename Sentinel = traits::range_iterator_t<Range const>>
    constexpr std::enable_if_t<traits::is_range_v<Range>,
        iterator_range<Iterator, Sentinel>>
    make_iterator_range(Range const& r)
    {
        return iterator_range<Iterator, Sentinel>(util::begin(r), util::end(r));
    }

    template <typename Iterator, typename Sentinel = Iterator>
    constexpr std::enable_if_t<traits::is_iterator_v<Iterator>,
        iterator_range<Iterator, Sentinel>>
    make_iterator_range(Iterator iterator, Sentinel sentinel) noexcept
    {
        return iterator_range<Iterator, Sentinel>(
            HPX_MOVE(iterator), HPX_MOVE(sentinel));
    }
}}    // namespace hpx::util

namespace hpx { namespace ranges {
    template <typename I, typename S = I>
    using subrange_t = hpx::util::iterator_range<I, S>;
}}    // namespace hpx::ranges
