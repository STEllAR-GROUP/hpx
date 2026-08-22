//  Copyright (c) 2007-2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Unit tests for the ring-walk fallback logic used while forwarding the
// Dijkstra termination detection token (see
// hpx::components::server::detail::dijkstra_forward_token). These tests
// exercise the pure decision logic through a mocked send callback, without
// requiring AGAS, parcels, or a running distributed runtime.

#include <hpx/config.hpp>
#include <hpx/modules/runtime_distributed.hpp>
#include <hpx/modules/testing.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

using hpx::components::server::detail::dijkstra_forward_token;

namespace {

    // Records every locality id that the mocked send was invoked with, in
    // call order, and returns a caller-supplied result for each target.
    struct mock_send
    {
        explicit mock_send(std::vector<std::uint32_t> live_targets)
          : live_targets_(HPX_MOVE(live_targets))
        {
        }

        bool operator()(std::uint32_t const target_locality_id)
        {
            calls_.push_back(target_locality_id);
            for (auto const live : live_targets_)
            {
                if (live == target_locality_id)
                {
                    succeeded_calls_.push_back(target_locality_id);
                    return true;
                }
            }
            return false;
        }

        std::vector<std::uint32_t> live_targets_;
        std::vector<std::uint32_t> calls_;
        std::vector<std::uint32_t> succeeded_calls_;
    };
}    // namespace

// 1. Happy path -- immediate neighbor alive.
void test_immediate_neighbor_alive()
{
    std::uint32_t locality_id = 4;
    constexpr std::uint32_t initiating_locality_id = 0;

    mock_send send({3});
    bool const result =
        dijkstra_forward_token(locality_id, initiating_locality_id, send);

    HPX_TEST(result);
    HPX_TEST_EQ(send.calls_.size(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(send.calls_[0], static_cast<std::uint32_t>(3));
    HPX_TEST_EQ(send.succeeded_calls_.size(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(send.succeeded_calls_[0], static_cast<std::uint32_t>(3));
    HPX_TEST_EQ(locality_id, static_cast<std::uint32_t>(3));
}

// 2. One dead intermediate locality -- the walk should skip it rather than
//    looping forever or stopping early.
void test_one_dead_intermediate_locality()
{
    std::uint32_t locality_id = 4;
    constexpr std::uint32_t initiating_locality_id = 0;

    mock_send send({2});
    bool const result =
        dijkstra_forward_token(locality_id, initiating_locality_id, send);

    HPX_TEST(result);
    std::vector<std::uint32_t> const expected_calls{3, 2};
    HPX_TEST(send.calls_ == expected_calls);
    std::vector<std::uint32_t> const expected_succeeded_calls{2};
    HPX_TEST(send.succeeded_calls_ == expected_succeeded_calls);
    HPX_TEST_EQ(locality_id, static_cast<std::uint32_t>(2));
}

// 3. All intermediates dead down to the initiator -- the walk returns false
//    once it reaches the initiating locality, leaving the fallback send to the
//    initiator to the caller.
void test_all_intermediates_dead_reaches_initiator()
{
    std::uint32_t locality_id = 4;
    constexpr std::uint32_t initiating_locality_id = 0;

    mock_send send({});
    bool const result =
        dijkstra_forward_token(locality_id, initiating_locality_id, send);

    HPX_TEST(!result);
    std::vector<std::uint32_t> const expected_calls{3, 2, 1, 0};
    HPX_TEST(send.calls_ == expected_calls);
    HPX_TEST(send.succeeded_calls_.empty());
    HPX_TEST_EQ(locality_id, initiating_locality_id);

    // caller-side fallback: send directly to the initiator exactly once
    mock_send fallback_send({});
    bool const fallback_result = fallback_send(initiating_locality_id);
    HPX_TEST(!fallback_result);
    HPX_TEST_EQ(fallback_send.calls_.size(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(fallback_send.calls_[0], initiating_locality_id);
}

// 4. Fallback target itself unreachable -- characterizes the known gap that
//    runtime_support::dijkstra_termination discards the return value of its
//    final fallback send to the initiator, i.e. a total ring failure is never
//    surfaced anywhere. dijkstra_forward_token itself has already done its job
//    correctly (it truthfully reports "false: nobody along the ring accepted
//    it"); the gap lives in the caller ignoring that report for the fallback
//    call. This test pins down today's (buggy) behavior so that a future fix
//    that propagates/logs the failure needs to consciously update this test
//    rather than silently regressing.
void test_fallback_to_initiator_also_unreachable()
{
    std::uint32_t locality_id = 4;
    constexpr std::uint32_t initiating_locality_id = 0;

    mock_send send({});
    bool const result =
        dijkstra_forward_token(locality_id, initiating_locality_id, send);
    HPX_TEST(!result);

    // the fallback send to the initiator also fails
    mock_send fallback_send({});
    bool const fallback_result = fallback_send(initiating_locality_id);

    HPX_TEST_EQ(fallback_send.calls_.size(), static_cast<std::size_t>(1));
    HPX_TEST_EQ(fallback_send.calls_[0], initiating_locality_id);
    HPX_TEST(fallback_send.succeeded_calls_.empty());

    // Known gap: nothing currently consumes fallback_result on this path in
    // runtime_support::dijkstra_termination, so a total ring failure is
    // silently dropped instead of being propagated, logged, or retried.
    HPX_TEST(!fallback_result);
}

// 5. Wrap-around when locality_id == 0 is handled by the caller before
//    invoking dijkstra_forward_token; verify the first probed target is
//    num_localities - 1.
void test_wrap_around_before_forwarding()
{
    constexpr std::uint32_t num_localities = 5;
    std::uint32_t locality_id = 0;
    if (0 == locality_id)
        locality_id = num_localities;

    constexpr std::uint32_t initiating_locality_id = 0;

    mock_send send({});
    bool const result =
        dijkstra_forward_token(locality_id, initiating_locality_id, send);

    HPX_TEST(!result);
    HPX_TEST(!send.calls_.empty());
    HPX_TEST_EQ(send.calls_[0], num_localities - 1);
    HPX_TEST(send.succeeded_calls_.empty());
}

// 6. No forwarding when already at the initiator -- guards against an
//    accidental self-send loop.
void test_no_forwarding_when_already_at_initiator()
{
    std::uint32_t locality_id = 2;
    constexpr std::uint32_t initiating_locality_id = 2;

    mock_send send({});
    bool const result =
        dijkstra_forward_token(locality_id, initiating_locality_id, send);

    HPX_TEST(!result);
    HPX_TEST(send.calls_.empty());
    HPX_TEST(send.succeeded_calls_.empty());
}

int main()
{
    test_immediate_neighbor_alive();
    test_one_dead_intermediate_locality();
    test_all_intermediates_dead_reaches_initiator();
    test_fallback_to_initiator_also_unreachable();
    test_wrap_around_before_forwarding();
    test_no_forwarding_when_already_at_initiator();

    return hpx::util::report_errors();
}
