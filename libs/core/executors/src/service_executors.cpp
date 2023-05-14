//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>
#include <hpx/executors/service_executors.hpp>
#include <hpx/functional/function.hpp>
#include <hpx/io_service/io_service_pool.hpp>

#include <asio/io_context.hpp>

namespace hpx::parallel::execution::detail {

    void service_executor::post(
        hpx::util::io_service_pool* pool, hpx::function<void()>&& f)
    {
        pool->get_io_service().post(HPX_MOVE(f));
    }
}    // namespace hpx::parallel::execution::detail
