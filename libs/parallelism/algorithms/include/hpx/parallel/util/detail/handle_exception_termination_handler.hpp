//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/modules/functional.hpp>

namespace hpx { namespace parallel { namespace util { namespace detail {
    using parallel_exception_termination_handler_type =
        hpx::util::function_nonser<void()>;
    HPX_PARALLELISM_EXPORT void set_parallel_exception_termination_handler(
        parallel_exception_termination_handler_type f);
    HPX_NORETURN HPX_PARALLELISM_EXPORT void
    parallel_exception_termination_handler();
}}}}    // namespace hpx::parallel::util::detail
