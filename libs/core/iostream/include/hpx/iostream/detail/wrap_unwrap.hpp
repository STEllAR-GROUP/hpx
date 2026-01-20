//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iostream/traits.hpp>
#include <hpx/modules/type_support.hpp>

#include <functional>
#include <type_traits>

namespace hpx::iostream::detail {

    //------------------Definition of wrap/unwrap traits--------------------------//
    HPX_CXX_CORE_EXPORT template <typename T>
    struct wrapped_type
      : std::conditional<is_std_io_v<T>, std::reference_wrapper<T>, T>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    struct unwrapped_type
    {
        using type = util::unwrap_reference_t<T>;
    };

    HPX_CXX_CORE_EXPORT template <typename T>
    struct unwrap_ios
      : util::lazy_conditional<is_std_io_v<T>, util::unwrap_reference<T>,
            type_identity<T>>
    {
    };

    //------------------Definition of wrap----------------------------------------//
    HPX_CXX_CORE_EXPORT template <typename T>
    decltype(auto) wrap(T&& t)
    {
        if constexpr (is_std_io_v<std::decay_t<T>>)
        {
            return std::ref(HPX_FORWARD(T, t));
        }
        else
        {
            return HPX_FORWARD(T, t);
        }
    }
}    // namespace hpx::iostream::detail
