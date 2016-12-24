//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_REBIND_EXECUTOR_SEP_07_2016_0658AM)
#define HPX_PARALLEL_REBIND_EXECUTOR_SEP_07_2016_0658AM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/util/decay.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    ///////////////////////////////////////////////////////////////////////////
    /// Function invocations executed by a group of sequential execution agents
    /// execute in sequential order.
    using sequential_execution_tag =
        parallel::execution::sequenced_execution_tag;

    /// Function invocations executed by a group of parallel execution agents
    /// execute in unordered fashion. Any such invocations executing in the
    /// same thread are indeterminately sequenced with respect to each other.
    ///
    /// \note \a parallel_execution_tag is weaker than
    ///       \a sequential_execution_tag.
    using parallel_execution_tag =
        parallel::execution::parallel_execution_tag;

    /// Function invocations executed by a group of vector execution agents are
    /// permitted to execute in unordered fashion when executed in different
    /// threads, and un-sequenced with respect to one another when executed in
    /// the same thread.
    ///
    /// \note \a vector_execution_tag is weaker than
    ///       \a parallel_execution_tag.
    using vector_execution_tag =
        parallel::execution::unsequenced_execution_tag;

    /// \cond NOINTERNAL
    struct task_execution_policy_tag
    {
        HPX_CONSTEXPR task_execution_policy_tag() {}
    };
    /// \endcond

    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Category1, typename Category2>
        struct is_not_weaker
          : std::false_type
        {};

        template <typename Category>
        struct is_not_weaker<Category, Category>
          : std::true_type
        {};

        template <>
        struct is_not_weaker<parallel_execution_tag, vector_execution_tag>
          : std::true_type
        {};

        template <>
        struct is_not_weaker<sequential_execution_tag, vector_execution_tag>
          : std::true_type
        {};

        template <>
        struct is_not_weaker<sequential_execution_tag, parallel_execution_tag>
          : std::true_type
        {};
        /// \endcond
    }

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
            (parallel::v1::detail::is_not_weaker<category2, category1>::value),
            "parallel::v1::detail::is_not_weaker<category2, category1>::value"
        );
        /// \endcond

        /// The type of the rebound execution policy
        typedef typename ExecutionPolicy::template rebind<
                executor_type, parameters_type
            >::type type;
    };
}}}

#endif
