//  Copyright (c) 2022 Shreyas Atre
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_CXX20_COROUTINES)

#if defined(__has_include)
#if __has_include(<coroutine>)
#include <coroutine>

namespace hpx {

    HPX_CXX_EXPORT using std::coroutine_handle;
    HPX_CXX_EXPORT using std::coroutine_traits;
    HPX_CXX_EXPORT using std::noop_coroutine;
    HPX_CXX_EXPORT using std::suspend_always;
    HPX_CXX_EXPORT using std::suspend_never;
}    // namespace hpx

#elif __has_include(<experimental/coroutine>)
#include <experimental/coroutine>

namespace hpx {

    HPX_CXX_EXPORT using std::experimental::coroutine_handle;
    HPX_CXX_EXPORT using std::experimental::coroutine_traits;
    HPX_CXX_EXPORT using std::experimental::noop_coroutine;
    HPX_CXX_EXPORT using std::experimental::suspend_always;
    HPX_CXX_EXPORT using std::experimental::suspend_never;
}    // namespace hpx

#endif
#endif

#endif    // HPX_HAVE_CXX20_COROUTINES
