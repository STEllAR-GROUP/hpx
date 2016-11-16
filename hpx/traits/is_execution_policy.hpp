//  Copyright (c) 2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_EXECUTION_POLICY_SEP_07_2016_0805AM)
#define HPX_TRAITS_IS_EXECUTION_POLICY_SEP_07_2016_0805AM

#include <hpx/config.hpp>
#include <hpx/config/inline_namespace.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v1)
{
    namespace detail
    {
        template <typename T>
        struct is_execution_policy
          : std::false_type
        {};

        template <typename T>
        struct is_parallel_execution_policy
          : std::false_type
        {};

        template <typename T>
        struct is_sequential_execution_policy
          : std::false_type
        {};

        template <typename T>
        struct is_async_execution_policy
          : std::false_type
        {};

        template <typename Executor>
        struct is_rebound_execution_policy
          : std::false_type
        {};

        template <typename Executor>
        struct is_vectorpack_execution_policy
          : std::false_type
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    /// 1. The type is_execution_policy can be used to detect execution
    ///    policies for the purpose of excluding function signatures
    ///    from otherwise ambiguous overload resolution participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_execution_policy<T> shall be publicly derived from
    ///    integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_execution_policy is undefined.
    ///
    template <typename T>
    struct is_execution_policy
      : detail::is_execution_policy<typename hpx::util::decay<T>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy enables parallelization
    ///
    /// 1. The type is_parallel_execution_policy can be used to detect parallel
    ///    execution policies for the purpose of excluding function signatures
    ///    from otherwise ambiguous overload resolution participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_parallel_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_parallel_execution_policy is undefined.
    ///
    template <typename T>
    struct is_parallel_execution_policy
      : detail::is_parallel_execution_policy<typename hpx::util::decay<T>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy does not enable
    ///            parallelization
    ///
    /// 1. The type is_sequential_execution_policy can be used to detect
    ///    non-parallel execution policies for the purpose of excluding
    ///    function signatures from otherwise ambiguous overload resolution
    ///    participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_sequential_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_sequential_execution_policy is undefined.
    ///
    // extension:
    template <typename T>
    struct is_sequential_execution_policy
      : detail::is_sequential_execution_policy<typename hpx::util::decay<T>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy makes algorithms
    ///            asynchronous
    ///
    /// 1. The type is_async_execution_policy can be used to detect
    ///    asynchronous execution policies for the purpose of excluding
    ///    function signatures from otherwise ambiguous overload resolution
    ///    participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_async_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_async_execution_policy is undefined.
    ///
    // extension:
    template <typename T>
    struct is_async_execution_policy
      : detail::is_async_execution_policy<typename hpx::util::decay<T>::type>
    {};

    template <typename T>
    struct is_rebound_execution_policy
      : detail::is_rebound_execution_policy<typename hpx::util::decay<T>::type>
    {};

    // extension:
    template <typename T>
    struct is_vectorpack_execution_policy
      : detail::is_vectorpack_execution_policy<typename hpx::util::decay<T>::type>
    {};
}}}

#endif
