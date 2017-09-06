//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_LAUNCH_POLICY_APR_8_2014_0436PM)
#define HPX_TRAITS_IS_LAUNCH_POLICY_APR_8_2014_0436PM

#include <hpx/config.hpp>
#include <hpx/runtime/launch_policy.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/traits/executor_traits.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace threads
{
    class executor;
}}

namespace hpx { namespace traits
{
    namespace detail
    {
        template <typename Policy>
        struct is_launch_policy
          : std::is_base_of<hpx::detail::policy_holder_base, Policy>
        {};

        template <typename Policy>
        struct is_threads_executor
          : std::is_base_of<threads::executor, Policy>
        {};
    }

    template <typename Policy>
    struct is_launch_policy
      : detail::is_launch_policy<typename hpx::util::decay<Policy>::type>
    {};

    template <typename Policy>
    struct is_threads_executor
      : detail::is_threads_executor<typename hpx::util::decay<Policy>::type>
    {};

    template <typename Policy>
    struct is_launch_policy_or_executor
      : std::integral_constant<bool,
            is_launch_policy<Policy>::value ||
                is_threads_executor<Policy>::value>
    {};

    ///////////////////////////////////////////////////////////////////////////
    template <typename Executor>
    struct executor_execution_category<Executor,
        typename std::enable_if<
            is_threads_executor<Executor>::value
        >::type>
    {
        typedef parallel::execution::parallel_execution_tag type;
    };
}}

#endif

