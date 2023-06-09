//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if (defined(WIN32) || defined(_WIN32) || defined(__WIN32__)) &&               \
    !defined(HPX_MINGW)

namespace hpx::util {

    HPX_CORE_EXPORT void set_thread_name(char const* thread_name) noexcept;
}    // namespace hpx::util

#else

namespace hpx::util {

    constexpr void set_thread_name(char const*) noexcept {}
}    // namespace hpx::util

#endif
