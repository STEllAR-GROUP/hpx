//  Copyright (c) 2017-2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_COMPILE_WITH_MODULES)
#include <hpx/functional/detail/invoke.hpp>
#endif

#define HPX_INVOKE(F, ...)                                                     \
    (::hpx::util::detail::invoke<decltype((F))>(F)(__VA_ARGS__))
