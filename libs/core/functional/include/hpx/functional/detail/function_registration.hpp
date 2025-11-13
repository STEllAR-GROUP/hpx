//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013-2025 Hartmut Kaiser
//  Copyright (c) 2014 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/debugging.hpp>
#include <hpx/modules/preprocessor.hpp>

#include <type_traits>

namespace hpx::util::detail {

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename VTable, typename T>
    struct get_function_name_declared : std::false_type
    {
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename VTable, typename F>
    struct get_function_name_impl
    {
        static char const* call()
#ifdef HPX_HAVE_AUTOMATIC_SERIALIZATION_REGISTRATION
        {
            return debug::type_id<F>();
        }
#else
            = delete;
#endif
    };

    ///////////////////////////////////////////////////////////////////////////
    HPX_CXX_EXPORT template <typename VTable, typename F>
    [[nodiscard]] char const* get_function_name()
    {
        return get_function_name_impl<VTable, F>::call();
    }
}    // namespace hpx::util::detail
