//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_REBIND_EXECUTOR_SEP_07_2016_0658AM)
#define HPX_PARALLEL_REBIND_EXECUTOR_SEP_07_2016_0658AM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/traits/executor_traits.hpp>
#include <hpx/traits/is_launch_policy.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace parallel { namespace execution
{
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
        struct is_not_weaker<parallel_execution_tag, unsequenced_execution_tag>
          : std::true_type
        {};

        template <>
        struct is_not_weaker<sequenced_execution_tag, unsequenced_execution_tag>
          : std::true_type
        {};

        template <>
        struct is_not_weaker<sequenced_execution_tag, parallel_execution_tag>
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
        typedef typename std::decay<Executor>::type executor_type;
        typedef typename std::decay<Parameters>::type parameters_type;

        typedef typename ExecutionPolicy::execution_category category1;
        typedef typename hpx::traits::executor_execution_category<
                executor_type
            >::type category2;

        static_assert(
            detail::is_not_weaker<category2, category1>::value,
            "detail::is_not_weaker<category2, category1>::value"
        );
        /// \endcond

        /// The type of the rebound execution policy
        typedef typename ExecutionPolicy::template rebind<
                executor_type, parameters_type
            >::type type;
    };
}}}

#endif
