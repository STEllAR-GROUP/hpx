//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_CXX20_COROUTINES)

#if defined(__has_include)
#if __has_include(<coroutine>)
#include <coroutine>

namespace hpx {

    using std::coroutine_handle;
    using std::noop_coroutine;
    using std::suspend_always;
    using std::suspend_never;
}    // namespace hpx
#define HPX_COROUTINE_NAMESPACE_STD std

#elif __has_include(<experimental/coroutine>)
#include <experimental/coroutine>

namespace hpx {

    using std::experimental::coroutine_handle;
    using std::experimental::noop_coroutine;
    using std::experimental::suspend_always;
    using std::experimental::suspend_never;
}    // namespace hpx
#define HPX_COROUTINE_NAMESPACE_STD std::experimental

#endif
#endif

#if !defined(HPX_COROUTINE_NAMESPACE_STD)
#error "Platform does not support C++20 coroutines"
#endif

#endif    // HPX_HAVE_CXX20_COROUTINES
