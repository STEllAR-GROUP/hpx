//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/executor_information_traits.hpp

#if !defined(HPX_PARALLEL_EXECUTOR_INFORMATION_TRAITS_AUG_26_2015_1133AM)
#define HPX_PARALLEL_EXECUTOR_INFORMATION_TRAITS_AUG_26_2015_1133AM

#include <hpx/config.hpp>
#include <hpx/async.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/threads/policies/topology.hpp>
#include <hpx/traits/detail/wrap_int.hpp>
#include <hpx/traits/is_executor_v1.hpp>
#include <hpx/util/always_void.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/unwrapped.hpp>

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

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{
    ///////////////////////////////////////////////////////////////////////////
    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        template <typename Parameters>
        std::size_t call_processing_units_parameter_count(Parameters && params);

        struct processing_units_count_helper
        {
            template <typename Executor, typename Parameters>
            static std::size_t call(hpx::traits::detail::wrap_int,
                Executor& exec, Parameters& params)
            {
                return call_processing_units_parameter_count(params);
            }

            template <typename Executor, typename Parameters>
            static auto call(int, Executor& exec, Parameters&)
            ->  decltype(exec.processing_units_count())
            {
                return exec.processing_units_count();
            }
        };

        template <typename Executor, typename Parameters>
        std::size_t call_processing_units_count(Executor& exec,
            Parameters& params)
        {
            return processing_units_count_helper::call(0, exec, params);
        }

        ///////////////////////////////////////////////////////////////////////
        struct has_pending_closures_helper
        {
            template <typename Executor>
            static auto call(hpx::traits::detail::wrap_int,
                Executor const& exec) -> bool
            {
                return false;   // assume stateless scheduling
            }

            template <typename Executor>
            static auto call(int, Executor const& exec)
            ->  decltype(exec.has_pending_closures())
            {
                return exec.has_pending_closures();
            }
        };

        template <typename Executor>
        bool call_has_pending_closures(Executor const& exec)
        {
            return has_pending_closures_helper::call(0, exec);
        }

        ///////////////////////////////////////////////////////////////////////
        struct get_pu_mask_helper
        {
            template <typename Executor>
            static threads::mask_cref_type call(hpx::traits::detail::wrap_int,
                Executor const&, threads::topology& topo, std::size_t thread_num)
            {
                return hpx::threads::get_pu_mask(topo, thread_num);
            }

            template <typename Executor>
            static auto call(int, Executor const& exec,
                    threads::topology& topo, std::size_t thread_num)
            ->  decltype(exec.get_pu_mask(topo, thread_num))
            {
                return exec.get_pu_mask(topo, thread_num);
            }
        };

        template <typename Executor>
        threads::mask_cref_type call_get_pu_mask(Executor const& exec,
            threads::topology& topo, std::size_t thread_num)
        {
            return get_pu_mask_helper::call(0, exec, topo, thread_num);
        }

        ///////////////////////////////////////////////////////////////////////
        struct set_scheduler_mode_helper
        {
            template <typename Executor, typename Mode>
            static void call(hpx::traits::detail::wrap_int, Executor& exec,
                Mode const& mode)
            {
            }

            template <typename Executor, typename Mode>
            static auto call(int, Executor& exec, Mode const& mode)
            ->  decltype(exec.set_scheduler_mode(mode))
            {
                exec.set_scheduler_mode(mode);
            }
        };

        template <typename Executor, typename Mode>
        void call_set_scheduler_mode(Executor& exec, Mode const& mode)
        {
            set_scheduler_mode_helper::call(0, exec, mode);
        }
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    /// The executor_information_traits type is used to various pieces of
    /// information from an executor.
    ///
    template <typename Executor, typename Enable>
    struct executor_information_traits
    {
        /// The type of the executor associated with this instance of
        /// \a executor_traits
        typedef Executor executor_type;

        /// Retrieve the number of (kernel-)threads used by the associated
        /// executor.
        ///
        /// \param exec  [in] The executor object to use for scheduling of the
        ///              function \a f.
        ///
        /// \note This calls exec.os_thread_count() if it exists;
        ///       otherwise it executes hpx::get_os_thread_count().
        ///
        template <typename Parameters>
        static std::size_t processing_units_count(executor_type const& exec,
            Parameters& params)
        {
            return detail::call_processing_units_count(exec, params);
        }

        /// Retrieve whether this executor has operations pending or not.
        ///
        /// \param exec  [in] The executor object to use for scheduling of the
        ///              function \a f.
        ///
        /// \note If the executor does not expose this information, this call
        ///       will always return \a false
        ///
        static bool has_pending_closures(executor_type const& exec)
        {
            return detail::call_has_pending_closures(exec);
        }

        /// Retrieve the bitmask describing the processing units the given
        /// thread is allowed to run on
        /// All threads::executors invoke sched.get_pu_mask().
        ///
        /// \param exec  [in] The executor object to use for querying the
        ///               number of pending tasks.
        ///
        static threads::mask_cref_type get_pu_mask(executor_type const& exec,
            threads::topology& topo, std::size_t thread_num)
        {
            return detail::call_get_pu_mask(exec, topo, thread_num);
        }

        /// Set various modes of operation on the scheduler underneath the
        /// given executor.
        ///
        /// \param exec     [in] The executor object to use.
        /// \param mode     [in] The new mode for the scheduler to pick up
        ///
        /// \note This calls exec.set_scheduler_mode(mode) if it exists;
        ///       otherwise it does nothing.
        ///
        template <typename Mode>
        static void set_scheduler_mode(executor_type& exec, Mode const& mode)
        {
            detail::call_set_scheduler_mode(exec, mode);
        }
    };
}}}

#endif
