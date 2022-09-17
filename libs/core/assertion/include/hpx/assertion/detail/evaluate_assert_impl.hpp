//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assertion/evaluate_assert.hpp>

#include <iostream>
#include <string>

#if !defined(HPX_ASSERTION_INLINE)
#define HPX_ASSERTION_INLINE inline
#endif

namespace hpx::assertion {

    namespace detail {
        HPX_ASSERTION_INLINE assertion_handler& get_handler()
        {
            static assertion_handler handler = nullptr;
            return handler;
        }

        HPX_ASSERTION_INLINE void handle_assert(hpx::source_location const& loc,
            const char* expr, std::string const& msg) noexcept
        {
            if (get_handler() == nullptr)
            {
                std::cerr << loc << ": Assertion '" << expr << "' failed";
                if (!msg.empty())
                {
                    std::cerr << " (" << msg << ")\n";
                }
                else
                {
                    std::cerr << '\n';
                }
                std::abort();
            }
            get_handler()(loc, expr, msg);
        }
    }    // namespace detail

    HPX_ASSERTION_INLINE void set_assertion_handler(
        detail::assertion_handler handler)
    {
        if (detail::get_handler() == nullptr)
        {
            detail::get_handler() = handler;
        }
    }
}    // namespace hpx::assertion

#undef HPX_ASSERTION_INLINE
