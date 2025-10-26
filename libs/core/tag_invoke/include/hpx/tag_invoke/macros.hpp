//  Copyright (c) 2017-2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if !defined(HPX_COMPILE_WITH_MODULES) ||                                      \
    (defined(HPX_COMPILE_BMI) && defined(HPX_COMPILE_TAG_INVOKE_WITH_MODULES))
#include <hpx/modules/type_support.hpp>
#endif

#define HPX_INVOKE(F, ...)                                                     \
    (::hpx::util::detail::invoke<decltype((F))>(F)(__VA_ARGS__))
#define HPX_INVOKE_R(R, F, ...)                                                \
    (::hpx::util::void_guard<R>(), HPX_INVOKE(F, __VA_ARGS__))
