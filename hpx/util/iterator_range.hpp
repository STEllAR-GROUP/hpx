//  Copyright (c) 2017 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_ITERATOR_RANGE_HPP
#define HPX_UTIL_ITERATOR_RANGE_HPP

#include <hpx/config.hpp>
#include <hpx/traits/is_range.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/range.hpp>

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    template <typename Iterator, typename Sentinel = Iterator>
    class iterator_range
    {
    public:
        iterator_range()
          : _iterator()
          , _sentinel()
        {}

        iterator_range(Iterator iterator, Sentinel sentinel)
          : _iterator(std::move(iterator))
          , _sentinel(std::move(sentinel))
        {}

        Iterator begin() const
        {
            return _iterator;
        }

        Iterator end() const
        {
            return _sentinel;
        }

        std::ptrdiff_t size() const
        {
            return std::distance(_iterator, _sentinel);
        }

        bool empty() const
        {
            return _iterator == _sentinel;
        }

    private:
        Iterator _iterator;
        Sentinel _sentinel;
    };

    template <
        typename Range,
        typename Iterator = typename traits::range_iterator<Range>::type,
        typename Sentinel = typename traits::range_iterator<Range>::type>
    typename std::enable_if<
        traits::is_range<Range>::value,
        iterator_range<Iterator, Sentinel>
    >::type make_iterator_range(Range& r)
    {
        return iterator_range<Iterator, Sentinel>(util::begin(r), util::end(r));
    }

    template <
        typename Range,
        typename Iterator = typename traits::range_iterator<Range const>::type,
        typename Sentinel = typename traits::range_iterator<Range const>::type>
    typename std::enable_if<
        traits::is_range<Range>::value,
        iterator_range<Iterator, Sentinel>
    >::type make_iterator_range(Range const& r)
    {
        return iterator_range<Iterator, Sentinel>(util::begin(r), util::end(r));
    }

    template <typename Iterator, typename Sentinel = Iterator>
    typename std::enable_if<
        traits::is_iterator<Iterator>::value,
        iterator_range<Iterator, Sentinel>
    >::type make_iterator_range(Iterator iterator, Sentinel sentinel)
    {
        return iterator_range<Iterator, Sentinel>(iterator, sentinel);
    }
}}

#endif /*HPX_UTIL_ITERATOR_RANGE_HPP*/
