//  Copyright (c) 2017 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <functional>

namespace hpx { namespace util {

    template <typename T>
    struct unwrap_reference
    {
        using type = T;
    };

    template <typename T>
    struct unwrap_reference<std::reference_wrapper<T>>
    {
        using type = T;
    };

    template <typename T>
    struct unwrap_reference<std::reference_wrapper<T> const>
    {
        using type = T;
    };

    template <typename T>
    using unwrap_reference_t = typename unwrap_reference<T>::type;

    template <typename T>
    HPX_FORCEINLINE unwrap_reference_t<T>& unwrap_ref(T& t)
    {
        return t;
    }
}}    // namespace hpx::util
