//  Copyright (c) 2020      ETH Zurich
//                2007-2017 Hartmut Kaiser
//                2017-2018 Taeguk Kwon
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/local/config.hpp>
#include <hpx/executors/exception_list.hpp>

#include <exception>

namespace hpx { namespace parallel { inline namespace v1 { namespace detail {
    exception_list_termination_handler_type&
    get_exception_list_termination_handler()
    {
        static exception_list_termination_handler_type f;
        return f;
    }

    void set_exception_list_termination_handler(
        exception_list_termination_handler_type f)
    {
        get_exception_list_termination_handler() = f;
    }

    HPX_NORETURN void exception_list_termination_handler()
    {
        if (get_exception_list_termination_handler())
        {
            get_exception_list_termination_handler()();
        }

        std::terminate();
    }
}}}}    // namespace hpx::parallel::v1::detail
