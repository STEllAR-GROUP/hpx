//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_FUTURE_RANGE_HPP)
#define HPX_TRAITS_IS_FUTURE_RANGE_HPP

#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_range.hpp>
#include <hpx/util/decay.hpp>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/or.hpp>
#include <boost/range/iterator_range.hpp>

#include <vector>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Range, typename Enable = void>
    struct is_future_range
      : boost::mpl::false_
    {};

    template <typename Range>
    struct is_future_range<Range,
            typename boost::enable_if<is_range<Range> >::type>
      : is_future<typename util::decay<Range>::type::value_type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Range, typename Enable = void>
    struct future_range_traits;

    template <typename Range>
    struct future_range_traits<
            Range, typename boost::enable_if<is_future_range<Range> >::type
        >
    {
        typedef typename Range::value_type future_type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_future_or_future_range
      : boost::mpl::or_<is_future<T>, is_future_range<T> >
    {};
}}

#endif
