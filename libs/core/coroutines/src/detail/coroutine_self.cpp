//  Copyright (c) 2008-2012 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/coroutines/detail/coroutine_self.hpp>

#include <cstddef>

namespace hpx { namespace threads { namespace coroutines { namespace detail {
    coroutine_self*& coroutine_self::local_self()
    {
        static thread_local coroutine_self* local_self_ = nullptr;
        return local_self_;
    }
}}}}    // namespace hpx::threads::coroutines::detail
