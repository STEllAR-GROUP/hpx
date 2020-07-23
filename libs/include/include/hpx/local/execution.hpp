//  Copyright (c) 2020 ETH Zurich
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/modules/execution.hpp>
#include <hpx/modules/executors.hpp>
#include <hpx/modules/thread_executors.hpp>

namespace hpx { namespace execution {
    using hpx::parallel::execution::par;
    using hpx::parallel::execution::par_unseq;
    using hpx::parallel::execution::parallel_executor;
    using hpx::parallel::execution::parallel_policy;
    using hpx::parallel::execution::parallel_unsequenced_policy;
    using hpx::parallel::execution::seq;
    using hpx::parallel::execution::sequenced_executor;
    using hpx::parallel::execution::sequenced_policy;
    using hpx::parallel::execution::task;

    using hpx::parallel::execution::auto_chunk_size;
    using hpx::parallel::execution::dynamic_chunk_size;
    using hpx::parallel::execution::guided_chunk_size;
    using hpx::parallel::execution::persistent_auto_chunk_size;
    using hpx::parallel::execution::static_chunk_size;
}}    // namespace hpx::execution
