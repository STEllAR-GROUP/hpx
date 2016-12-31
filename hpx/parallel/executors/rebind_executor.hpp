//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_REBIND_EXECUTOR_SEP_07_2016_0658AM)
#define HPX_PARALLEL_REBIND_EXECUTOR_SEP_07_2016_0658AM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/util/decay.hpp>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    /// \cond NOINTERNAL
    struct task_policy_tag
    {
        HPX_CONSTEXPR task_policy_tag() {}
    };

    /// The execution policy tag \a task can be used to create a execution
    /// policy which forces the given algorithm to be executed in an
    /// asynchronous way.
    static task_policy_tag HPX_CONSTEXPR_OR_CONST task;
    /// \endcond

    /// Rebind the type of executor used by an execution policy. The execution
    /// category of Executor shall not be weaker than that of ExecutionPolicy.
    template <typename ExecutionPolicy, typename Executor, typename Parameters>
    struct rebind_executor
    {
        /// \cond NOINTERNAL
        typedef typename hpx::util::decay<Executor>::type executor_type;
        typedef typename hpx::util::decay<Parameters>::type parameters_type;

        typedef typename ExecutionPolicy::execution_category category1;
        typedef typename executor_traits<executor_type>::execution_category
            category2;

        static_assert(
            (parallel::v3::detail::is_not_weaker<category2, category1>::value),
            "parallel::v3::detail::is_not_weaker<category2, category1>::value"
        );
        /// \endcond

        /// The type of the rebound execution policy
        typedef typename ExecutionPolicy::template rebind<
                executor_type, parameters_type
            >::type type;
    };
}}}

namespace hpx { namespace parallel { namespace executioon
{
    ///////////////////////////////////////////////////////////////////////////
    using task_execution_policy_tag = parallel::execution::task_policy_tag;

    static task_execution_policy_tag HPX_CONSTEXPR_OR_CONST task;
}}}

#endif
