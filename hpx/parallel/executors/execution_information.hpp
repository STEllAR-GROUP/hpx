//  Copyright (c) 2017 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_INFORMATION_JAN_16_2017_0444PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_INFORMATION_JAN_16_2017_0444PM

#include <hpx/config.hpp>
#include <hpx/runtime/threads/topology.hpp>
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
    template <typename Parameters, typename Executor>
    std::size_t call_processing_units_parameter_count(Parameters && params,
        Executor && exec);
    /// \endcond
}}}}

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace execution
{
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // customization point for interface processing_units_count()
        template <typename Parameters_>
        struct processing_units_count_parameter_helper
        {
            template <typename Parameters, typename Executor>
            static std::size_t call(hpx::traits::detail::wrap_int,
                Parameters && params, Executor && exec)
            {
                return hpx::get_os_thread_count();
            }

            template <typename Parameters, typename Executor>
            static auto call(int, Parameters && params, Executor && exec)
            ->  decltype(params.processing_units_count(
                    std::forward<Executor>(exec)))
            {
                return params.processing_units_count(
                    std::forward<Executor>(exec));
            }

            template <typename Executor>
            static std::size_t call(Parameters_& params, Executor && exec)
            {
                return call(0, params, std::forward<Executor>(exec));
            }

            template <typename Parameters, typename Executor>
            static std::size_t call(Parameters params, Executor && exec)
            {
                return call(static_cast<Parameters_&>(params),
                    std::forward<Executor>(exec));
            }
        };

        template <typename Parameters, typename Executor>
        std::size_t call_processing_units_parameter_count(Parameters && params,
            Executor && exec)
        {
            return processing_units_count_parameter_helper<
                    typename hpx::util::decay_unwrap<Parameters>::type
                >::call(std::forward<Parameters>(params),
                    std::forward<Executor>(exec));
        }

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
            ->  decltype(call_processing_units_parameter_count(params,
                    std::forward<AnyExecutor>(exec)))
            {
                return call_processing_units_parameter_count(
                    params, std::forward<AnyExecutor>(exec));
            }

            template <typename AnyExecutor, typename Parameters>
            HPX_FORCEINLINE static auto
            call(int, AnyExecutor && exec, Parameters&)
            ->  decltype(exec.processing_units_count())
            {
                return exec.processing_units_count();
            }

            template <typename AnyExecutor, typename Parameters>
            struct result
            {
                using type = decltype(call(0,
                    std::declval<AnyExecutor>(), std::declval<Parameters&>()
                ));
            };
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

            template <typename AnyExecutor>
            struct result
            {
                using type = decltype(call(0,
                    std::declval<AnyExecutor>()
                ));
            };
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

            template <typename AnyExecutor>
            struct result
            {
                using type = decltype(call(0,
                    std::declval<AnyExecutor>(),
                    std::declval<threads::topology&>(),
                    std::declval<std::size_t>()
                ));
            };
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

            template <typename AnyExecutor, typename Mode>
            struct result
            {
                using type = decltype(call(0,
                    std::declval<AnyExecutor>(), std::declval<Mode const&>()
                ));
            };
        };

        /// \endcond
    }
}}}

#endif

