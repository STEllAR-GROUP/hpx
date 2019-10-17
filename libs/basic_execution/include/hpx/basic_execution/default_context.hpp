//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_BASIC_EXECUTION_DEFAULT_CONTEXT_HPP
#define HPX_BASIC_EXECUTION_DEFAULT_CONTEXT_HPP

#include <hpx/basic_execution/context_base.hpp>
#include <hpx/basic_execution/resource_base.hpp>
#include <hpx/functional/unique_function.hpp>

namespace hpx { namespace basic_execution {
    struct default_context : basic_execution::context_base
    {
        resource_base const& resource() const override
        {
            return resource_;
        }

        void post(hpx::util::unique_function_nonser<void()> f) const override;

        resource_base resource_;
    };
}}

#endif
