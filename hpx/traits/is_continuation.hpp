//  Copyright (c) 2015 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/always_void.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename Continuation, typename Enable = void>
        struct is_continuation_impl
          : std::false_type
        {};

        template <typename Continuation>
        struct is_continuation_impl<Continuation,
            typename util::always_void<typename Continuation::continuation_tag>::type
        > : std::true_type
        {};
    }

    template <typename Continuation, typename Enable = void>
    struct is_continuation
      : detail::is_continuation_impl<typename std::decay<Continuation>::type>
    {};
}}


