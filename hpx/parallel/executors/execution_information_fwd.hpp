//  Copyright (c) 2017 Hartmut Kaiser
//  Copyright (c) 2017 Google
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
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution
{
    ///////////////////////////////////////////////////////////////////////////
    // Define infrastructure for customization points
    namespace detail
    {
        /// \cond NOINTERNAL
        struct processing_units_count_tag {};
        struct has_pending_closures_tag {};
        struct get_pu_mask_tag {};
        struct set_scheduler_mode_tag {};

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
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
#endif
        /// \endcond
    }

    ///////////////////////////////////////////////////////////////////////////
    // Executor information customization points
    namespace detail
    {
        /// \cond NOINTERNAL
        template <typename Executor, typename Enable = void>
        struct processing_units_count_fn_helper;

        template <typename Executor, typename Enable = void>
        struct has_pending_closures_fn_helper;

        template <typename Executor, typename Enable = void>
        struct get_pu_mask_fn_helper;

        template <typename Executor, typename Enable = void>
        struct set_scheduler_mode_fn_helper;
        /// \endcond
    }

    namespace detail
    {
        /// \cond NOINTERNAL

        ///////////////////////////////////////////////////////////////////////
        // post_at dispatch point
        template <typename Executor, typename Parameters>
        HPX_FORCEINLINE auto
        processing_units_count(Executor && exec, Parameters& params)
        -> decltype(processing_units_count_fn_helper<
                    typename std::decay<Executor>::type
                >::call(0, std::forward<Executor>(exec), params)
            )
        {
            return processing_units_count_fn_helper<
                    typename std::decay<Executor>::type
                >::call(0, std::forward<Executor>(exec), params);
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Executor, typename Parameters>
        HPX_FORCEINLINE auto customization_point<processing_units_count_tag>::
        operator()(Executor&& exec, Parameters& params) const
        {
            return processing_units_count(std::forward<Executor>(exec), params);
        }
#else
        template <>
        struct customization_point<processing_units_count_tag>
        {
        public:
            template <typename Executor, typename Parameters>
            HPX_FORCEINLINE auto operator()(
                Executor&& exec, Parameters& params) const
                -> decltype(processing_units_count(std::forward<Executor>(exec),
                    params))
            {
                return processing_units_count(
                    std::forward<Executor>(exec), params);
            }
        };
#endif

        ///////////////////////////////////////////////////////////////////////
        // post_at dispatch point
        template <typename Executor>
        HPX_FORCEINLINE auto
        has_pending_closures(Executor && exec)
        -> decltype(has_pending_closures_fn_helper<
                    typename std::decay<Executor>::type
                >::call(0, std::forward<Executor>(exec))
            )
        {
            return has_pending_closures_fn_helper<
                    typename std::decay<Executor>::type
                >::call(0, std::forward<Executor>(exec));
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Executor>
        HPX_FORCEINLINE auto customization_point<has_pending_closures_tag>::
        operator()(Executor&& exec) const
        {
            return has_pending_closures(std::forward<Executor>(exec));
        }
#else
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
#endif

        ///////////////////////////////////////////////////////////////////////
        // post_at dispatch point
        template <typename Executor>
        HPX_FORCEINLINE auto
        get_pu_mask(Executor && exec, threads::topology& topo,
                std::size_t thread_num)
        -> decltype(get_pu_mask_fn_helper<
                    typename std::decay<Executor>::type
                >::call(0, std::forward<Executor>(exec), topo, thread_num)
            )
        {
            return has_pending_closures_fn_helper<
                    typename std::decay<Executor>::type
                >::call(0, std::forward<Executor>(exec), topo, thread_num);
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Executor>
        HPX_FORCEINLINE auto customization_point<get_pu_mask_tag>::operator()(
            Executor&& exec, threads::topology& topo,
            std::size_t thread_num) const
        {
            return get_pu_mask(std::forward<Executor>(exec), topo, thread_num);
        }
#else
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
#endif

        // post_at dispatch point
        template <typename Executor, typename Mode>
        HPX_FORCEINLINE auto
        set_scheduler_mode(Executor && exec, Mode const& mode)
        -> decltype(set_scheduler_mode_fn_helper<
                    typename std::decay<Executor>::type
                >::call(0, std::forward<Executor>(exec), mode)
            )
        {
            return set_scheduler_mode_fn_helper<
                    typename std::decay<Executor>::type
                >::call(0, std::forward<Executor>(exec), mode);
        }

#if defined(HPX_HAVE_CXX14_RETURN_TYPE_DEDUCTION)
        template <typename Executor, typename Mode>
        HPX_FORCEINLINE auto customization_point<set_scheduler_mode_tag>::
        operator()(Executor&& exec, Mode const& mode) const
        {
            return set_scheduler_mode(std::forward<Executor>(exec), mode);
        }
#else
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
#endif
        /// \endcond
    }

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

