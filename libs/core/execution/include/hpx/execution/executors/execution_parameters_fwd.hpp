//  Copyright (c) 2016-2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concepts/concepts.hpp>
#include <hpx/execution/traits/executor_traits.hpp>
#include <hpx/execution_base/execution.hpp>
#include <hpx/execution_base/traits/is_executor.hpp>
#include <hpx/execution_base/traits/is_executor_parameters.hpp>
#include <hpx/functional/tag_fallback_dispatch.hpp>
#include <hpx/type_support/decay.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace hpx { namespace parallel { namespace execution {

    ///////////////////////////////////////////////////////////////////////////
    // Executor information customization points
    namespace detail {
        /// \cond NOINTERNAL
        template <typename Parameters, typename Executor,
            typename Enable = void>
        struct get_chunk_size_fn_helper;

        template <typename Parameters, typename Executor,
            typename Enable = void>
        struct maximal_number_of_chunks_fn_helper;

        template <typename Parameters, typename Executor,
            typename Enable = void>
        struct reset_thread_distribution_fn_helper;

        template <typename Parameters, typename Executor,
            typename Enable = void>
        struct processing_units_count_fn_helper;

        template <typename Parameters, typename Executor,
            typename Enable = void>
        struct mark_begin_execution_fn_helper;

        template <typename Parameters, typename Executor,
            typename Enable = void>
        struct mark_end_of_scheduling_fn_helper;

        template <typename Parameters, typename Executor,
            typename Enable = void>
        struct mark_end_execution_fn_helper;
        /// \endcond
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    // define customization points

    /// Return the number of invocations of the given function \a f which
    /// should be combined into a single task
    ///
    /// \param params   [in] The executor parameters object to use for
    ///                 determining the chunk size for the given number of
    ///                 tasks \a num_tasks.
    /// \param exec     [in] The executor object which will be used
    ///                 for scheduling of the loop iterations.
    /// \param f        [in] The function which will be optionally scheduled
    ///                 using the given executor.
    /// \param cores    [in] The number of cores the number of chunks
    ///                 should be determined for.
    /// \param num_tasks [in] The number of tasks the chunk size should be
    ///                 determined for
    ///
    /// \note  The parameter \a f is expected to be a nullary function
    ///        returning a `std::size_t` representing the number of
    ///        iteration the function has already executed (i.e. which
    ///        don't have to be scheduled anymore).
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct get_chunk_size_t final
      : hpx::functional::tag_fallback<get_chunk_size_t>
    {
    private:
        // clang-format off
        template <typename Parameters, typename Executor, typename F,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_parameters<Parameters>::value &&
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            get_chunk_size_t, Parameters&& params, Executor&& exec, F&& f,
            std::size_t cores, std::size_t num_tasks)
        {
            return detail::get_chunk_size_fn_helper<
                hpx::util::decay_unwrap_t<Parameters>,
                std::decay_t<Executor>>::call(std::forward<Parameters>(params),
                std::forward<Executor>(exec), std::forward<F>(f), cores,
                num_tasks);
        }
    } get_chunk_size{};

    /// Return the largest reasonable number of chunks to create for a
    /// single algorithm invocation.
    ///
    /// \param params   [in] The executor parameters object to use for
    ///                 determining the number of chunks for the given
    ///                 number of \a cores.
    /// \param exec     [in] The executor object which will be used
    ///                 for scheduling of the loop iterations.
    /// \param cores    [in] The number of cores the number of chunks
    ///                 should be determined for.
    /// \param num_tasks [in] The number of tasks the chunk size should be
    ///                 determined for
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct maximal_number_of_chunks_t final
      : hpx::functional::tag_fallback<maximal_number_of_chunks_t>
    {
    private:
        // clang-format off
        template <typename Parameters, typename Executor,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_parameters<Parameters>::value &&
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            maximal_number_of_chunks_t, Parameters&& params, Executor&& exec,
            std::size_t cores, std::size_t num_tasks)
        {
            return detail::maximal_number_of_chunks_fn_helper<
                hpx::util::decay_unwrap_t<Parameters>,
                std::decay_t<Executor>>::call(std::forward<Parameters>(params),
                std::forward<Executor>(exec), cores, num_tasks);
        }
    } maximal_number_of_chunks{};

    /// Reset the internal round robin thread distribution scheme for the
    /// given executor.
    ///
    /// \param params   [in] The executor parameters object to use for
    ///                 resetting the thread distribution scheme.
    /// \param exec     [in] The executor object to use.
    ///
    /// \note This calls params.reset_thread_distribution(exec) if it exists;
    ///       otherwise it does nothing.
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct reset_thread_distribution_t final
      : hpx::functional::tag_fallback<reset_thread_distribution_t>
    {
    private:
        // clang-format off
        template <typename Parameters, typename Executor,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_parameters<Parameters>::value &&
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            reset_thread_distribution_t, Parameters&& params, Executor&& exec)
        {
            return detail::reset_thread_distribution_fn_helper<
                hpx::util::decay_unwrap_t<Parameters>,
                std::decay_t<Executor>>::call(std::forward<Parameters>(params),
                std::forward<Executor>(exec));
        }
    } reset_thread_distribution{};

    /// Retrieve the number of (kernel-)threads used by the associated
    /// executor.
    ///
    /// \param params [in] The executor parameters object to use as a
    ///              fallback if the executor does not expose
    ///
    /// \note This calls params.processing_units_count(Executor&&) if it
    ///       exists; otherwise it forwards the request to the executor
    ///       parameters object.
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct processing_units_count_t final
      : hpx::functional::tag_fallback<processing_units_count_t>
    {
    private:
        // clang-format off
        template <typename Parameters, typename Executor,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_parameters<Parameters>::value &&
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            processing_units_count_t, Parameters&& params, Executor&& exec)
        {
            return detail::processing_units_count_fn_helper<
                hpx::util::decay_unwrap_t<Parameters>,
                std::decay_t<Executor>>::call(std::forward<Parameters>(params),
                std::forward<Executor>(exec));
        }
    } processing_units_count{};

    /// Mark the begin of a parallel algorithm execution
    ///
    /// \param params [in] The executor parameters object to use as a
    ///              fallback if the executor does not expose
    ///
    /// \note This calls params.mark_begin_execution(exec) if it exists;
    ///       otherwise it does nothing.
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct mark_begin_execution_t final
      : hpx::functional::tag_fallback<mark_begin_execution_t>
    {
    private:
        // clang-format off
        template <typename Parameters, typename Executor,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_parameters<Parameters>::value &&
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            mark_begin_execution_t, Parameters&& params, Executor&& exec)
        {
            return detail::mark_begin_execution_fn_helper<
                hpx::util::decay_unwrap_t<Parameters>,
                std::decay_t<Executor>>::call(std::forward<Parameters>(params),
                std::forward<Executor>(exec));
        }
    } mark_begin_execution{};

    /// Mark the end of scheduling tasks during parallel algorithm execution
    ///
    /// \param params [in] The executor parameters object to use as a
    ///              fallback if the executor does not expose
    ///
    /// \note This calls params.mark_begin_execution(exec) if it exists;
    ///       otherwise it does nothing.
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct mark_end_of_scheduling_t final
      : hpx::functional::tag_fallback<mark_end_of_scheduling_t>
    {
    private:
        // clang-format off
        template <typename Parameters, typename Executor,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_parameters<Parameters>::value &&
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            mark_end_of_scheduling_t, Parameters&& params, Executor&& exec)
        {
            return detail::mark_end_of_scheduling_fn_helper<
                hpx::util::decay_unwrap_t<Parameters>,
                std::decay_t<Executor>>::call(std::forward<Parameters>(params),
                std::forward<Executor>(exec));
        }
    } mark_end_of_scheduling{};

    /// Mark the end of a parallel algorithm execution
    ///
    /// \param params [in] The executor parameters object to use as a
    ///              fallback if the executor does not expose
    ///
    /// \note This calls params.mark_end_execution(exec) if it exists;
    ///       otherwise it does nothing.
    ///
    HPX_INLINE_CONSTEXPR_VARIABLE struct mark_end_execution_t final
      : hpx::functional::tag_fallback<mark_end_execution_t>
    {
    private:
        // clang-format off
        template <typename Parameters, typename Executor,
            HPX_CONCEPT_REQUIRES_(
                hpx::traits::is_executor_parameters<Parameters>::value &&
                hpx::traits::is_executor_any<Executor>::value
            )>
        // clang-format on
        friend HPX_FORCEINLINE decltype(auto) tag_fallback_dispatch(
            mark_end_execution_t, Parameters&& params, Executor&& exec)
        {
            return detail::mark_end_execution_fn_helper<
                hpx::util::decay_unwrap_t<Parameters>,
                std::decay_t<Executor>>::call(std::forward<Parameters>(params),
                std::forward<Executor>(exec));
        }
    } mark_end_execution{};
}}}    // namespace hpx::parallel::execution
