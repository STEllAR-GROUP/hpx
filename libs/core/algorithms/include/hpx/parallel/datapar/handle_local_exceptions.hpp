//  Copyright (c) 2007-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#if defined(HPX_HAVE_DATAPAR)
#include <hpx/execution/traits/is_execution_policy.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx::parallel::util::detail {

    // simd et.al. need to terminate on exceptions
    template <typename ExPolicy>
    struct handle_local_exceptions<ExPolicy,
        std::enable_if_t<hpx::is_vectorpack_execution_policy_v<ExPolicy>>>
      : terminate_on_local_exceptions
    {
    };
}    // namespace hpx::parallel::util::detail

#endif
