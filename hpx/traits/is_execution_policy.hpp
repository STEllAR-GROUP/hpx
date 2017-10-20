//  Copyright (c) 2016-2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/traits/is_execution_policy.hpp

#if !defined(HPX_TRAITS_IS_EXECUTION_POLICY_SEP_07_2016_0805AM)
#define HPX_TRAITS_IS_EXECUTION_POLICY_SEP_07_2016_0805AM

#include <hpx/config.hpp>
#include <hpx/util/decay.hpp>

#include <type_traits>

namespace hpx { namespace parallel { namespace execution
{
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename T>
        struct is_execution_policy
          : std::false_type
        {};

        template <typename T>
        struct is_parallel_execution_policy
          : std::false_type
        {};

        template <typename T>
        struct is_sequenced_execution_policy
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
        /// \endcond
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
      : execution::detail::is_execution_policy<
            typename hpx::util::decay<T>::type>
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
      : execution::detail::is_parallel_execution_policy<
            typename hpx::util::decay<T>::type>
    {};

    ///////////////////////////////////////////////////////////////////////////
    /// Extension: Detect whether given execution policy does not enable
    ///            parallelization
    ///
    /// 1. The type is_sequenced_execution_policy can be used to detect
    ///    non-parallel execution policies for the purpose of excluding
    ///    function signatures from otherwise ambiguous overload resolution
    ///    participation.
    /// 2. If T is the type of a standard or implementation-defined execution
    ///    policy, is_sequenced_execution_policy<T> shall be publicly derived
    ///    from integral_constant<bool, true>, otherwise from
    ///    integral_constant<bool, false>.
    /// 3. The behavior of a program that adds specializations for
    ///    is_sequenced_execution_policy is undefined.
    ///
    // extension:
    template <typename T>
    struct is_sequenced_execution_policy
      : execution::detail::is_sequenced_execution_policy<
            typename hpx::util::decay<T>::type>
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
      : execution::detail::is_async_execution_policy<
            typename hpx::util::decay<T>::type>
    {};

    /// \cond NOINTERNAL
    template <typename T>
    struct is_rebound_execution_policy
      : execution::detail::is_rebound_execution_policy<
            typename hpx::util::decay<T>::type>
    {};

    // extension:
    template <typename T>
    struct is_vectorpack_execution_policy
      : execution::detail::is_vectorpack_execution_policy<
            typename hpx::util::decay<T>::type>
    {};
    /// \endcond
}}}

#if defined(HPX_HAVE_EXECUTION_POLICY_COMPATIBILITY)
///////////////////////////////////////////////////////////////////////////////
// Compatibility layer for changes introduced by C++17
namespace hpx { namespace parallel { inline namespace v1
{
    /// \cond NOINTERNAL
    template <typename T>
    using is_execution_policy =
        execution::is_execution_policy<T>;

    template <typename T>
    using is_parallel_execution_policy =
        execution::is_parallel_execution_policy<T>;

    template <typename T>
    using is_sequenced_execution_policy =
        execution::is_sequenced_execution_policy<T>;

    template <typename T>
    using is_async_execution_policy =
        execution::is_async_execution_policy<T>;

    template <typename T>
    using is_rebound_execution_policy =
        execution::is_rebound_execution_policy<T>;

    template <typename T>
    using is_vectorpack_execution_policy =
        execution::is_vectorpack_execution_policy<T>;
    /// \endcond
}}}
#endif

#endif
