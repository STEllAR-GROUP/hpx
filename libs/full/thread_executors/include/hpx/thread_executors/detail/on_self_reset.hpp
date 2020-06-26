//  Copyright (c) 2007-2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/threading_base.hpp>

namespace hpx { namespace threads { namespace executors { namespace detail {
    struct on_self_reset
    {
        on_self_reset(threads::thread_self* self)
        {
            threads::detail::set_self_ptr(self);
        }
        ~on_self_reset()
        {
            threads::detail::set_self_ptr(nullptr);
        }
    };
}}}}    // namespace hpx::threads::executors::detail
