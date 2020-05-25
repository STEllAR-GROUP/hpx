//  Copyright (c) 2007-2017 Hartmut Kaiser
//  Copyright (c) 2016 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/futures/traits/is_future.hpp>
#include <hpx/iterator_support/traits/is_range.hpp>

#include <functional>
#include <type_traits>

namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename Enable = void>
    struct is_future_range : std::false_type
    {
    };

    template <typename R>
    struct is_future_range<R, typename std::enable_if<is_range<R>::value>::type>
      : is_future<typename range_traits<R>::value_type>
    {
    };

    template <typename R, typename Enable = void>
    struct is_ref_wrapped_future_range : std::false_type
    {
    };

    template <typename R>
    struct is_ref_wrapped_future_range<::std::reference_wrapper<R>>
      : is_future_range<R>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, bool IsFutureRange = is_future_range<R>::value>
    struct future_range_traits
    {
    };

    template <typename R>
    struct future_range_traits<R, true>
    {
        typedef typename range_traits<R>::value_type future_type;
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename T>
        struct is_future_or_future_range
          : std::integral_constant<bool,
                is_future<T>::value || is_future_range<T>::value>
        {
        };
    }    // namespace detail
}}       // namespace hpx::traits
