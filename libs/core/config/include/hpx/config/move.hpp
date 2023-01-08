//  Copyright (c) 2021 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_BUILTIN_FORWARD_MOVE)
#include <utility>

#define HPX_MOVE(...) std::move(__VA_ARGS__)

#else
#include <type_traits>

#define HPX_MOVE(...)                                                          \
    static_cast<std::remove_reference_t<decltype(__VA_ARGS__)>&&>(__VA_ARGS__)

#endif
