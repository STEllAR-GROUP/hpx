//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_PARALLEL_EXECUTORS_CURRENT_EXECUTOR_HPP
#define HPX_PARALLEL_EXECUTORS_CURRENT_EXECUTOR_HPP

#include <hpx/parallel/executors/thread_pool_executor.hpp>

namespace hpx { namespace parallel { namespace execution {
    using current_executor = parallel::execution::thread_pool_executor;
}}}    // namespace hpx::parallel::execution

#endif
