// Copyright (c) 2025 Sai Charan Arvapally
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/executors/parallel_scheduler.hpp>

std::shared_ptr<
    hpx::execution::experimental::detail::replaceability::parallel_scheduler>
hpx::execution::experimental::detail::replaceability::
    query_parallel_scheduler_backend()
{
    static std::shared_ptr<parallel_scheduler> backend =
        std::make_shared<default_parallel_scheduler>();
    return backend;
}