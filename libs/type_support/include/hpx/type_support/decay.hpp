//  Copyright (c) 2012 Thomas Heller
//  Copyright (c) 2013-2015 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <functional>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    template <typename T>
    struct decay : std::decay<T>
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    namespace detail {
        template <typename TD>
        struct decay_unwrap_impl
        {
            typedef TD type;
        };

        template <typename X>
        struct decay_unwrap_impl<::std::reference_wrapper<X>>
        {
            typedef X& type;
        };
    }    // namespace detail

    template <typename T>
    struct decay_unwrap
      : detail::decay_unwrap_impl<typename std::decay<T>::type>
    {
    };
}}    // namespace hpx::util
