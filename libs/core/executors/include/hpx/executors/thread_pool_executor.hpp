//  Copyright (c) 2019-2020 ETH Zurich
//  Copyright (c) 2007-2023 Hartmut Kaiser
//  Copyright (c) 2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/executors/parallel_executor.hpp>

namespace hpx::parallel::execution {

    using thread_pool_executor HPX_DEPRECATED_V(1, 9,
        "hpx::parallel::execution::thread_pool_executor is deprecated, use "
        "hpx::execution::parallel_executor instead") =
        hpx::execution::parallel_executor;
}    // namespace hpx::parallel::execution
