//  Copyright (c) 2018-2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/resiliency/config.hpp>
#include <hpx/functional/tag_invoke.hpp>
#include <hpx/modules/async_local.hpp>

#include <utility>

namespace hpx { namespace resiliency { namespace experimental {

    ///////////////////////////////////////////////////////////////////////////
    // helper base class implementing the deferred tag_invoke logic for CPOs
    template <typename Tag, typename BaseTag>
    struct tag_deferred : hpx::functional::tag<Tag>
    {
        // force unwrapping of the inner future on return
        template <typename... Args>
        friend HPX_FORCEINLINE auto tag_invoke(Tag, Args&&... args) ->
            typename hpx::functional::tag_invoke_result<BaseTag,
                Args&&...>::type
        {
            return hpx::dataflow(BaseTag{}, std::forward<Args>(args)...);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    // Replay customization points

    /// Customization point for asynchronously launching the given function \a f.
    /// repeatedly. Verify the result of those invocations using the given
    /// predicate \a pred.
    /// Repeat launching on error exactly \a n times (except if
    /// abort_replay_exception is thrown).
    HPX_INLINE_CONSTEXPR_VARIABLE struct async_replay_validate_t final
      : hpx::functional::tag<async_replay_validate_t>
    {
    } async_replay_validate{};

    /// Customization point for asynchronously launching given function \a f
    /// repeatedly. Repeat launching on error exactly \a n times (except if
    /// abort_replay_exception is thrown).
    HPX_INLINE_CONSTEXPR_VARIABLE struct async_replay_t final
      : hpx::functional::tag<async_replay_t>
    {
    } async_replay{};

    /// Customization point for asynchronously launching the given function \a f.
    /// repeatedly. Verify the result of those invocations using the given
    /// predicate \a pred.
    /// Repeat launching on error exactly \a n times.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    HPX_INLINE_CONSTEXPR_VARIABLE struct dataflow_replay_validate_t final
      : tag_deferred<dataflow_replay_validate_t, async_replay_validate_t>
    {
    } dataflow_replay_validate{};

    /// Customization point for asynchronously launching the given function \a f.
    /// repeatedly.
    /// Repeat launching on error exactly \a n times.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    HPX_INLINE_CONSTEXPR_VARIABLE struct dataflow_replay_t final
      : tag_deferred<dataflow_replay_t, async_replay_t>
    {
    } dataflow_replay{};

    ///////////////////////////////////////////////////////////////////////////
    // Replicate customization points

    /// Customization point for asynchronously launching the given function \a f
    /// exactly \a n times concurrently. Verify the result of those invocations
    /// using the given predicate \a pred.
    /// Run all the valid results against a user provided voting function.
    /// Return the valid output.
    HPX_INLINE_CONSTEXPR_VARIABLE struct async_replicate_vote_validate_t final
      : hpx::functional::tag<async_replicate_vote_validate_t>
    {
    } async_replicate_vote_validate{};

    ///////////////////////////////////////////////////////////////////////////
    /// Customization point for asynchronously launching the given function \a f
    /// exactly \a n times concurrently. Verify the result of those invocations
    /// using the given predicate \a pred.
    /// Run all the valid results against a user provided voting function.
    /// Return the valid output.
    HPX_INLINE_CONSTEXPR_VARIABLE struct async_replicate_vote_t final
      : hpx::functional::tag<async_replicate_vote_t>
    {
    } async_replicate_vote{};

    ///////////////////////////////////////////////////////////////////////////
    /// Customization point for asynchronously launching the given function \a f
    /// exactly \a n times concurrently. Verify the result of those invocations
    /// using the given predicate \a pred.
    /// Return the first valid result.
    HPX_INLINE_CONSTEXPR_VARIABLE struct async_replicate_validate_t final
      : hpx::functional::tag<async_replicate_validate_t>
    {
    } async_replicate_validate{};

    ///////////////////////////////////////////////////////////////////////////
    /// Customization point for asynchronously launching the given function \a f
    /// exactly \a n times concurrently. Verify the result of those invocations
    /// by checking for exception.
    /// Return the first valid result.
    HPX_INLINE_CONSTEXPR_VARIABLE struct async_replicate_t final
      : hpx::functional::tag<async_replicate_t>
    {
    } async_replicate{};

    /// Customization point for asynchronously launching the given function \a f
    /// exactly \a n times concurrently. Run all the valid results against a
    /// user provided voting function.
    /// Return the valid output.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    HPX_INLINE_CONSTEXPR_VARIABLE struct dataflow_replicate_vote_validate_t
        final
      : tag_deferred<dataflow_replicate_vote_validate_t,
            async_replicate_vote_validate_t>
    {
    } dataflow_replicate_vote_validate{};

    /// Customization point for asynchronously launching the given function \a f
    /// exactly \a n times concurrently. Run all the valid results against a
    /// user provided voting function.
    /// Return the valid output.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    HPX_INLINE_CONSTEXPR_VARIABLE struct dataflow_replicate_vote_t final
      : tag_deferred<dataflow_replicate_vote_t, async_replicate_vote_t>
    {
    } dataflow_replicate_vote{};

    /// Customization point for asynchronously launching the given function \a f
    /// exactly \a n times concurrently. Verify the result of those invocations
    /// using the given predicate \a pred. Return the first valid result.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    HPX_INLINE_CONSTEXPR_VARIABLE struct dataflow_replicate_validate_t final
      : tag_deferred<dataflow_replicate_validate_t, async_replicate_validate_t>
    {
    } dataflow_replicate_validate{};

    /// Customization point for asynchronously launching the given function \a f
    /// exactly \a n times concurrently. Return the first valid result.
    ///
    /// Delay the invocation of \a f if any of the arguments to \a f are
    /// futures.
    HPX_INLINE_CONSTEXPR_VARIABLE struct dataflow_replicate_t final
      : tag_deferred<dataflow_replicate_t, async_replicate_t>
    {
    } dataflow_replicate{};
}}}    // namespace hpx::resiliency::experimental
