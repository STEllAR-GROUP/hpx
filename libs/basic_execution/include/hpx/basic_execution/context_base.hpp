//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_BASIC_EXECUTION_CONTEXT_BASE_HPP
#define HPX_BASIC_EXECUTION_CONTEXT_BASE_HPP

#include <hpx/basic_execution/resource_base.hpp>
#include <hpx/timing/steady_clock.hpp>
#include <hpx/functional/unique_function.hpp>

#include <cstdint>

namespace hpx { namespace basic_execution {

    struct context_base
    {
        virtual ~context_base() = default;

        virtual resource_base const& resource() const = 0;

        virtual void post(hpx::util::unique_function_nonser<void()> f) const = 0;
    };

    inline bool operator==(context_base const& lhs, context_base const& rhs)
    {
        // just do a pointer comparison for now...
        return &lhs == &rhs;
    }

    inline bool operator!=(context_base const& lhs, context_base const& rhs)
    {
        return !(lhs == rhs);
    }
}}    // namespace hpx::basic_execution

#endif
