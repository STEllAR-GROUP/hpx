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

#include <type_traits>

namespace hpx::iostream::detail {

    HPX_CXX_CORE_EXPORT template <typename T, typename Tag1, typename Tag2,
        typename Tag3 = util::void_t, typename Tag4 = util::void_t,
        typename Tag5 = util::void_t, typename Tag6 = util::void_t,
        typename Category = category_of_t<T>>
    struct dispatch
      : util::select<std::is_convertible<Category, Tag1>, Tag1,
            std::is_convertible<Category, Tag2>, Tag2,
            std::is_convertible<Category, Tag3>, Tag3,
            std::is_convertible<Category, Tag4>, Tag4,
            std::is_convertible<Category, Tag5>, Tag5,
            std::is_convertible<Category, Tag6>, Tag6>
    {
    };

    HPX_CXX_CORE_EXPORT template <typename T, typename Tag1, typename Tag2,
        typename Tag3 = util::void_t, typename Tag4 = util::void_t,
        typename Tag5 = util::void_t, typename Tag6 = util::void_t,
        typename Category = category_of_t<T>>
    using dispatch_t =
        dispatch<T, Tag1, Tag2, Tag3, Tag4, Tag5, Tag6, Category>::type;
}    // namespace hpx::iostream::detail
