//  Copyright (c) 2017 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// hpxinspect:nounnamed

#if !defined(HPX_PARALLEL_EXECUTORS_EXECUTION_INFORMATION_FWD_JAN_16_2017_0350PM)
#define HPX_PARALLEL_EXECUTORS_EXECUTION_INFORMATION_FWD_JAN_16_2017_0350PM

#include <hpx/config.hpp>
#include <hpx/parallel/executors/execution_fwd.hpp>
#include <hpx/runtime/threads/thread_data_fwd.hpp>
#include <hpx/traits/executor_traits.hpp>

#include <cstddef>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    // Define infrastructure for customization points
    namespace detail
    {
        struct processing_units_count_tag {};
        struct has_pending_closures_tag {};
        struct get_pu_mask_tag {};
        struct set_scheduler_mode_tag {};

        // forward declare customization point implementations
        template <>
        struct customization_point<processing_units_count_tag>
        {
            template <typename Executor, typename Parameters>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, Parameters& params) const;
        };

        template <>
        struct customization_point<has_pending_closures_tag>
        {
            template <typename Executor>
            HPX_FORCEINLINE
            auto operator()(Executor && exec) const;
        };

        template <>
        struct customization_point<get_pu_mask_tag>
        {
            template <typename Executor>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, threads::topology& topo,
                std::size_t thread_num) const;
        };

        template <>
        struct customization_point<set_scheduler_mode_tag>
        {
            template <typename Executor, typename Mode>
            HPX_FORCEINLINE
            auto operator()(Executor && exec, Mode const& mode) const;
        };
    }
    /// \endcond

    // define customization points
    namespace
    {
        /// Retrieve the number of (kernel-)threads used by the associated
        /// executor.
        ///
        /// \param exec  [in] The executor object to use to extract the
        ///              requested information for.
        ///
        /// \note This calls exec.os_thread_count() if it exists;
        ///       otherwise it executes hpx::get_os_thread_count().
        ///
        constexpr detail::customization_point<
                detail::processing_units_count_tag
            > const& processing_units_count = detail::static_const<
                    detail::customization_point<detail::processing_units_count_tag>
                >::value;

        /// Retrieve whether this executor has operations pending or not.
        ///
        /// \param exec  [in] The executor object to use to extract the
        ///              requested information for.
        ///
        /// \note If the executor does not expose this information, this call
        ///       will always return \a false
        ///
        constexpr detail::customization_point<
                detail::has_pending_closures_tag
            > const& has_pending_closures = detail::static_const<
                    detail::customization_point<detail::has_pending_closures_tag>
                >::value;

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
                    detail::customization_point<detail::get_pu_mask_tag>
                >::value;

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
                detail::set_scheduler_mode_tag
            > const& set_scheduler_mode = detail::static_const<
                    detail::customization_point<detail::set_scheduler_mode_tag>
                >::value;
    }
}}}

#endif

