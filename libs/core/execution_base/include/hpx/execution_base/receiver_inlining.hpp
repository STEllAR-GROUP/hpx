//  Copyright (c) 2025 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/execution_base/receiver.hpp>

#include <type_traits>
#include <utility>

namespace hpx::execution::experimental {

    template <typename Rcvr, typename ChildOp>
    concept inlinable_receiver = receiver<Rcvr> && requires(ChildOp* child) {
        { Rcvr::make_receiver_for(child) } noexcept;
        requires std::is_same_v<decltype(Rcvr::make_receiver_for(child)), Rcvr>;
    };

    template <typename Derived, typename Receiver>
    class inlinable_operation_state
    {
    protected:
        explicit inlinable_operation_state(Receiver r) noexcept(
            std::is_nothrow_move_constructible_v<Receiver>)
          : rcvr_(HPX_MOVE(r))
        {
        }

        Receiver& get_receiver() noexcept
        {
            return rcvr_;
        }

        Receiver const& get_receiver() const noexcept
        {
            return rcvr_;
        }

    private:
        Receiver rcvr_;
    };

    template <typename Derived, typename Receiver>
        requires inlinable_receiver<Receiver, Derived>
    class inlinable_operation_state<Derived, Receiver>
    {
    protected:
        explicit inlinable_operation_state([[maybe_unused]] Receiver r) noexcept
        {
        }

        Receiver get_receiver() noexcept
        {
            return Receiver::make_receiver_for(static_cast<Derived*>(this));
        }

        Receiver get_receiver() const noexcept
        {
            return Receiver::make_receiver_for(
                static_cast<Derived const*>(this));
        }
    };

}    // namespace hpx::execution::experimental
