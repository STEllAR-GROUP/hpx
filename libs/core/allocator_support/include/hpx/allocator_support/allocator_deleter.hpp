//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// This is partially taken from: http://www.garret.ru/threadalloc/readme.html

#pragma once

#include <hpx/config.hpp>

#include <memory>

namespace hpx::util {

    ///////////////////////////////////////////////////////////////////////////
    template <typename Allocator>
    struct allocator_deleter
    {
        template <typename SharedState>
        void operator()(SharedState* state) noexcept
        {
            using traits = std::allocator_traits<Allocator>;
            traits::deallocate(alloc_, state, 1);
        }

        Allocator alloc_;
    };
}    // namespace hpx::util
