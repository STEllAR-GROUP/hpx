//  Copyright (c) 2017 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_INFORMATION_JAN_16_2017_0444PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_INFORMATION_JAN_16_2017_0444PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/executor_traits.hpp>
#include <hpx/traits/is_executor.hpp>

#include <hpx/parallel/executors/execution.hpp>
#include <hpx/parallel/executors/execution_information_fwd.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
// forward declaration only
namespace hpx { namespace threads
{
    HPX_API_EXPORT threads::mask_cref_type get_pu_mask(threads::topology& topo,
        std::size_t thread_num);
}}

namespace hpx { namespace parallel { inline namespace v3 { namespace detail
{
    /// \cond NOINTERNAL
    template <typename Parameters>
    std::size_t call_processing_units_parameter_count(Parameters && params);
    /// \endcond
}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace execution
{
    // customization point for interface processing_units_count()
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename Executor>
        struct processing_units_count_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value ||
                hpx::traits::is_two_way_executor<Executor>::value ||
                hpx::traits::is_never_blocking_one_way_executor<Executor>::value
            >::type>
        {
            template <typename AnyExecutor, typename Parameters>
            HPX_FORCEINLINE static auto
            call(hpx::traits::detail::wrap_int, AnyExecutor && exec,
                Parameters& params)
            -> decltype(parallel::v3::detail::
                    call_processing_units_parameter_count(params))
            {
                return parallel::v3::detail::
                    call_processing_units_parameter_count(params);
            }

            template <typename AnyExecutor, typename Parameters>
            HPX_FORCEINLINE static auto
            call(int, AnyExecutor && exec, Parameters&)
            -> decltype(exec.processing_units_count())
            {
                return exec.processing_units_count();
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface has_pending_closures()
        template <typename Executor>
        struct has_pending_closures_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value ||
                hpx::traits::is_two_way_executor<Executor>::value ||
                hpx::traits::is_never_blocking_one_way_executor<Executor>::value
            >::type>
        {
            template <typename AnyExecutor>
            HPX_FORCEINLINE static bool
            call(hpx::traits::detail::wrap_int, AnyExecutor && exec)
            {
                return false;   // assume stateless scheduling
            }

            template <typename AnyExecutor>
            HPX_FORCEINLINE static auto
            call(int, AnyExecutor && exec)
            -> decltype(exec.has_pending_closures())
            {
                return exec.has_pending_closures();
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface get_pu_mask()
        template <typename Executor>
        struct get_pu_mask_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value ||
                hpx::traits::is_two_way_executor<Executor>::value ||
                hpx::traits::is_never_blocking_one_way_executor<Executor>::value
            >::type>
        {
            template <typename AnyExecutor>
            HPX_FORCEINLINE static threads::mask_cref_type
            call(hpx::traits::detail::wrap_int,
                    AnyExecutor && exec, threads::topology& topo,
                    std::size_t thread_num)
            {
                return hpx::threads::get_pu_mask(topo, thread_num);
            }

            template <typename AnyExecutor>
            HPX_FORCEINLINE static auto
            call(int,
                    AnyExecutor && exec, threads::topology& topo,
                    std::size_t thread_num)
            -> decltype(exec.get_pu_mask(topo, thread_num))
            {
                return exec.get_pu_mask(topo, thread_num);
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface set_scheduler_mode()
        template <typename Executor>
        struct set_scheduler_mode_fn_helper<Executor,
            typename std::enable_if<
                hpx::traits::is_one_way_executor<Executor>::value ||
                hpx::traits::is_two_way_executor<Executor>::value ||
                hpx::traits::is_never_blocking_one_way_executor<Executor>::value
            >::type>
        {
            template <typename AnyExecutor, typename Mode>
            HPX_FORCEINLINE static void
            call(hpx::traits::detail::wrap_int,
                AnyExecutor && exec, Mode const& mode)
            {
            }

            template <typename AnyExecutor, typename Mode>
            HPX_FORCEINLINE static auto
            call(int, AnyExecutor && exec, Mode const& mode)
            -> decltype(exec.set_scheduler_mode(mode))
            {
                exec.set_scheduler_mode(mode);
            }
        };

        /// \endcond
    }
}}}

#endif

