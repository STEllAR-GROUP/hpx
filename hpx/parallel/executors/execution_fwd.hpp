//  Copyright (c) 2017 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_FWD_DEC_23_0712PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_FWD_DEC_23_0712PM

#include <hpx/config.hpp>
#include <hpx/traits/executor_traits.hpp>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    /// Function invocations executed by a group of sequential execution agents
    /// execute in sequential order.
    struct sequenced_execution_tag {};

    /// Function invocations executed by a group of parallel execution agents
    /// execute in unordered fashion. Any such invocations executing in the
    /// same thread are indeterminately sequenced with respect to each other.
    ///
    /// \note \a parallel_execution_tag is weaker than
    ///       \a sequenced_execution_tag.
    struct parallel_execution_tag {};

    /// Function invocations executed by a group of vector execution agents are
    /// permitted to execute in unordered fashion when executed in different
    /// threads, and un-sequenced with respect to one another when executed in
    /// the same thread.
    ///
    /// \note \a unsequenced_execution_tag is weaker than
    ///       \a parallel_execution_tag.
    struct unsequenced_execution_tag {};

    /// \cond NOINTERNAL
    struct task_policy_tag
    {
        HPX_CONSTEXPR task_policy_tag() {}
    };

    ///////////////////////////////////////////////////////////////////////////
    // Define infrastructure for customization points
    namespace detail
    {
        struct post_tag {};
        struct sync_execute_tag {};
        struct async_execute_tag {};
        struct then_execute_tag {};
        struct sync_bulk_execute_tag {};
        struct async_bulk_execute_tag {};
        struct then_bulk_execute_tag {};

        template <typename Executor, typename ... Ts>
        struct undefined_customization_point;

        template <typename Tag>
        struct customization_point
        {
            template <typename Executor, typename ... Ts>
            auto operator()(Executor && exec, Ts &&... ts) const
            ->  undefined_customization_point<Executor, Ts...>
            {
                return undefined_customization_point<Executor, Ts...>{};
            }
        };

#ifdef HPX_HAVE_CXX14_AUTO
        // forward declare customization point implementations
        template <>
        struct customization_point<post_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<sync_execute_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<async_execute_tag>
        {
            template <typename Executor, typename F, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Ts &&... ts) const;
        };

        template <>
        struct customization_point<then_execute_tag>
        {
            template <typename Executor, typename F, typename Future,
                typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Future& predecessor,
                Ts &&... ts) const;
        };

        template <>
        struct customization_point<sync_bulk_execute_tag>
        {
            template <typename Executor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Shape const& shape,
                Ts &&... ts) const;
        };

        template <>
        struct customization_point<async_bulk_execute_tag>
        {
            template <typename Executor, typename F, typename Shape,
                typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Shape const& shape,
                Ts &&... ts) const;
        };

        template <>
        struct customization_point<then_bulk_execute_tag>
        {
            template <typename Executor, typename F, typename Shape,
                typename Future, typename ... Ts>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, F && f, Shape const& shape,
                Future& predecessor, Ts &&... ts) const;
        };
#endif

        // Helper facility to avoid ODR violations
        template <typename T>
        struct static_const
        {
            static T const value;
        };

        template <typename T>
        T const static_const<T>::value = T{};
    }
    /// \endcond

}}}

#endif

