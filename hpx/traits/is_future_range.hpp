//  Copyright (c) 2007-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nodeprecatedinclude:boost/ref.hpp
// hpxinspect:nodeprecatedname:boost::reference_wrapper

#if !defined(HPX_TRAITS_IS_FUTURE_RANGE_HPP)
#define HPX_TRAITS_IS_FUTURE_RANGE_HPP

#include <hpx/traits/is_future.hpp>
#include <hpx/traits/is_range.hpp>
#include <hpx/util/decay.hpp>

#include <boost/ref.hpp>

#include <functional>
#include <type_traits>
#include <vector>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Range, typename Enable = void>
    struct is_future_range
      : std::false_type
    {};

    template <typename Range>
    struct is_future_range<Range,
            typename std::enable_if<is_range<Range>::value>::type>
      : is_future<typename util::decay<Range>::type::value_type>
    {};

    template <typename Range, typename Enable = void>
    struct is_ref_wrapped_future_range
      : std::false_type
    {};

    template <typename Range>
    struct is_ref_wrapped_future_range<boost::reference_wrapper<Range> >
      : is_future_range<Range>
    {};

    template <typename Range>
    struct is_ref_wrapped_future_range<std::reference_wrapper<Range> >
      : is_future_range<Range>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Range, typename Enable = void>
    struct future_range_traits;

    template <typename Range>
    struct future_range_traits<
            Range, typename std::enable_if<is_future_range<Range>::value >::type
        >
    {
        typedef typename Range::value_type future_type;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct is_future_or_future_range
      : std::integral_constant<bool,
            is_future<T>::value || is_future_range<T>::value>
    {};
}}

#endif
