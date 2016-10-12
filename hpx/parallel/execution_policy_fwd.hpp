//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTION_POLICY_FWD_JUL_31_2015_0837PM)
#define HPX_PARALLEL_EXECUTION_POLICY_FWD_JUL_31_2015_0837PM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#if defined(HPX_HAVE_VC_DATAPAR)
#include <hpx/parallel/datapar/execution_policy_fwd.hpp>
#endif

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    // forward declarations, see execution_policy.hpp
    struct sequential_execution_policy;

    template <typename Executor, typename Parameters>
    struct sequential_execution_policy_shim;

    struct sequential_task_execution_policy;

    template <typename Executor, typename Parameters>
    struct sequential_task_execution_policy_shim;

    struct parallel_execution_policy;

    template <typename Executor, typename Parameters>
    struct parallel_execution_policy_shim;

    struct parallel_task_execution_policy;

    template <typename Executor, typename Parameters>
    struct parallel_task_execution_policy_shim;

    struct parallel_vector_execution_policy;
}}}

#endif
