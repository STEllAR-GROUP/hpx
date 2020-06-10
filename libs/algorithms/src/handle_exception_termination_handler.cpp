//  Copyright (c) 2020      ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/parallel/util/detail/handle_exception_termination_handler.hpp>

#include <exception>

namespace hpx { namespace parallel { namespace util { namespace detail {
    parallel_exception_termination_handler_type&
    get_parallel_exception_termination_handler()
    {
        static parallel_exception_termination_handler_type f;
        return f;
    }

    void set_parallel_exception_termination_handler(
        parallel_exception_termination_handler_type f)
    {
        get_parallel_exception_termination_handler() = f;
    }

    HPX_NORETURN void parallel_exception_termination_handler()
    {
        if (get_parallel_exception_termination_handler())
        {
            get_parallel_exception_termination_handler()();
        }

        std::terminate();
    }
}}}}    // namespace hpx::parallel::util::detail
