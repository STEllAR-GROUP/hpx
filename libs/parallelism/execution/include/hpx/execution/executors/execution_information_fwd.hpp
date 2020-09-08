//  Copyright (c) 2017-2020 Hartmut Kaiser
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/modules/topology.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution {
    ///////////////////////////////////////////////////////////////////////////
    // Define infrastructure for customization points
    namespace detail {
        /// \cond NOINTERNAL
        struct has_pending_closures_tag
        {
        };

        struct get_pu_mask_tag
        {
        };

        struct set_scheduler_mode_tag
        {
        };
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // Executor information customization points
    namespace detail {
        /// \cond NOINTERNAL
        template <typename Executor, typename Enable = void>
        struct has_pending_closures_fn_helper;

        template <typename Executor, typename Enable = void>
        struct get_pu_mask_fn_helper;

        template <typename Executor, typename Enable = void>
        struct set_scheduler_mode_fn_helper;
        /// \endcond
    }    // namespace detail

    namespace detail {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // has_pending_closures dispatch point
        template <typename Executor>
        HPX_FORCEINLINE auto has_pending_closures(Executor&& exec) ->
            typename has_pending_closures_fn_helper<typename std::decay<
                Executor>::type>::template result<Executor>::type
        {
            return has_pending_closures_fn_helper<
                typename std::decay<Executor>::type>::call(0,
                std::forward<Executor>(exec));
        }

        template <>
        struct customization_point<has_pending_closures_tag>
        {
        public:
            template <typename Executor>
            HPX_FORCEINLINE auto operator()(Executor&& exec) const
                -> decltype(has_pending_closures(std::forward<Executor>(exec)))
            {
                return has_pending_closures(std::forward<Executor>(exec));
            }
        };

        ///////////////////////////////////////////////////////////////////////
        // get_pu_mask dispatch point
        template <typename Executor>
        HPX_FORCEINLINE auto get_pu_mask(
            Executor&& exec, threads::topology& topo, std::size_t thread_num) ->
            typename get_pu_mask_fn_helper<typename std::decay<
                Executor>::type>::template result<Executor>::type
        {
            return get_pu_mask_fn_helper<
                typename std::decay<Executor>::type>::call(0,
                std::forward<Executor>(exec), topo, thread_num);
        }

        template <>
        struct customization_point<get_pu_mask_tag>
        {
        public:
            template <typename Executor>
            HPX_FORCEINLINE auto operator()(Executor&& exec,
                threads::topology& topo, std::size_t thread_num) const
                -> decltype(
                    get_pu_mask(std::forward<Executor>(exec), topo, thread_num))
            {
                return get_pu_mask(
                    std::forward<Executor>(exec), topo, thread_num);
            }
        };

        // set_scheduler_mode dispatch point
        template <typename Executor, typename Mode>
        HPX_FORCEINLINE auto set_scheduler_mode(
            Executor&& exec, Mode const& mode) ->
            typename set_scheduler_mode_fn_helper<typename std::decay<
                Executor>::type>::template result<Executor, Mode>::type
        {
            return set_scheduler_mode_fn_helper<
                typename std::decay<Executor>::type>::call(0,
                std::forward<Executor>(exec), mode);
        }

        template <>
        struct customization_point<set_scheduler_mode_tag>
        {
        public:
            template <typename Executor, typename Mode>
            HPX_FORCEINLINE auto operator()(
                Executor&& exec, Mode const& mode) const
                -> decltype(
                    set_scheduler_mode(std::forward<Executor>(exec), mode))
            {
                return set_scheduler_mode(std::forward<Executor>(exec), mode);
            }
        };
        /// \endcond
    }    // namespace detail

    // define customization points
    namespace {
        /// Retrieve whether this executor has operations pending or not.
        ///
        /// \param exec  [in] The executor object to use to extract the
        ///              requested information for.
        ///
        /// \note If the executor does not expose this information, this call
        ///       will always return \a false
        ///
        constexpr detail::customization_point<
            detail::has_pending_closures_tag> const& has_pending_closures =
            detail::static_const<detail::customization_point<
                detail::has_pending_closures_tag>>::value;

        /// Retrieve the bitmask describing the processing units the given
        /// thread is allowed to run on
        ///
        /// All threads::executors invoke sched.get_pu_mask().
        ///
        /// \param exec  [in] The executor object to use for querying the
        ///              number of pending tasks.
        /// \param topo  [in] The topology object to use to extract the
        ///              requested information.
        /// \param thream_num [in] The sequence number of the thread to
        ///              retrieve information for.
        ///
        /// \note If the executor does not support this operation, this call
        ///       will always invoke hpx::threads::get_pu_mask()
        ///
        constexpr detail::customization_point<detail::get_pu_mask_tag> const&
            get_pu_mask = detail::static_const<
                detail::customization_point<detail::get_pu_mask_tag>>::value;

        /// Set various modes of operation on the scheduler underneath the
        /// given executor.
        ///
        /// \param exec     [in] The executor object to use.
        /// \param mode     [in] The new mode for the scheduler to pick up
        ///
        /// \note This calls exec.set_scheduler_mode(mode) if it exists;
        ///       otherwise it does nothing.
        ///
        constexpr detail::customization_point<
            detail::set_scheduler_mode_tag> const& set_scheduler_mode =
            detail::static_const<detail::customization_point<
                detail::set_scheduler_mode_tag>>::value;
    }    // namespace
}}}      // namespace hpx::parallel::execution
