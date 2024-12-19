//  Copyright (c) 2013-2016 Thomas Heller
//  Copyright (c) 2022 Christopher Taylor
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/config/compiler_specific.hpp>
#include <hpx/config/defines.hpp>

#if defined(HPX_WINDOWS)
#define HPX_HAVE_THREADS_GET_STACK_POINTER
#else
#if defined(HPX_HAVE_BUILTIN_FRAME_ADDRESS)
#define HPX_HAVE_THREADS_GET_STACK_POINTER
#else
#if defined(__x86_64__) || defined(__amd64) || defined(__i386__) ||            \
    defined(__i486__) || defined(__i586__) || defined(__i686__) ||             \
    defined(__powerpc__) || defined(__arm__) || defined(__riscv)
#define HPX_HAVE_THREADS_GET_STACK_POINTER
#endif
#endif

#include <cstddef>
#include <limits>

namespace hpx::threads::coroutines::detail {
    HPX_CORE_EXPORT std::size_t get_stack_ptr() noexcept;
}    // namespace hpx::threads::coroutines::detail
#endif
