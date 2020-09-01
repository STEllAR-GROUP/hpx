//  Copyright (c) 2016 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

// TODO: Should this be experimental?
#if defined(HPX_HAVE_DATAPAR)
namespace hpx { namespace execution { inline namespace v1 {
    ///////////////////////////////////////////////////////////////////////////
    struct dataseq_policy;

    template <typename Executor, typename Parameters>
    struct dataseq_policy_shim;

    struct dataseq_task_policy;

    template <typename Executor, typename Parameters>
    struct dataseq_task_policy_shim;

    ///////////////////////////////////////////////////////////////////////////
    struct datapar_policy;

    template <typename Executor, typename Parameters>
    struct datapar_policy_shim;

    struct datapar_task_policy;

    template <typename Executor, typename Parameters>
    struct datapar_task_policy_shim;
}}}    // namespace hpx::execution::v1

#endif
