//  Copyright (c) 2007-2016 Hartmut Kaiser
//  Copyright (c)      2011 Bryce Lelbach
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/errors/throw_exception.hpp>
#include <hpx/executors/current_executor.hpp>

namespace hpx { namespace threads {
    hpx::execution::parallel_executor get_executor(
        thread_id_type const& id, error_code& ec)
    {
        if (HPX_UNLIKELY(!id))
        {
            HPX_THROWS_IF(ec, hpx::error::null_thread_id,
                "hpx::threads::get_executor", "null thread id encountered");
            return hpx::execution::parallel_executor();
        }

        if (&ec != &throws)
            ec = make_success_code();

        return hpx::execution::parallel_executor(
            get_thread_id_data(id)->get_scheduler_base()->get_parent_pool());
    }
}}    // namespace hpx::threads

namespace hpx { namespace this_thread {
    hpx::execution::parallel_executor get_executor(error_code& ec)
    {
        return threads::get_executor(threads::get_self_id(), ec);
    }
}}    // namespace hpx::this_thread
