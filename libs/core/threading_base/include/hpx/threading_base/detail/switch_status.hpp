//  Copyright (c) 2007-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/threading_base/thread_data.hpp>
#include <hpx/threading_base/threading_base_fwd.hpp>

#include <atomic>

namespace hpx::threads::detail {

    ///////////////////////////////////////////////////////////////////////
    // helper class for switching thread state in and out during execution
    class switch_status
    {
    public:
        switch_status(
            thread_id_ref_type const& t, thread_state prev_state) noexcept
          : thread_(get_thread_id_data(t))
          , prev_state_(prev_state)
          , next_thread_id_(nullptr)
          , need_restore_state_(thread_->set_state_tagged(
                thread_schedule_state::active, prev_state_, orig_state_))
        {
        }

        ~switch_status()
        {
            if (need_restore_state_)
            {
                store_state(prev_state_);
            }
        }

        constexpr bool is_valid() const noexcept
        {
            return need_restore_state_;
        }

        // allow to change the state the thread will be switched to after
        // execution
        thread_state operator=(thread_result_type&& new_state) noexcept
        {
            prev_state_ = thread_state(
                new_state.first, prev_state_.state_ex(), prev_state_.tag() + 1);
            if (new_state.second != nullptr)
            {
                next_thread_id_ = HPX_MOVE(new_state.second);
            }
            return prev_state_;
        }

        // Get the state this thread was in before execution (usually pending),
        // this helps making sure no other worker-thread is started to execute
        // this HPX-thread in the meantime.
        thread_schedule_state get_previous() const noexcept
        {
            return prev_state_.state();
        }

        // This restores the previous state, while making sure that the original
        // state has not been changed since we started executing this thread.
        // The function returns true if the state has been set, false otherwise.
        bool store_state(thread_state& newstate) noexcept
        {
            disable_restore();

            if (thread_->restore_state(prev_state_, orig_state_,
                    std::memory_order_relaxed, std::memory_order_relaxed))
            {
                newstate = prev_state_;
                return true;
            }
            return false;
        }

        // disable default handling in destructor
        void disable_restore() noexcept
        {
            need_restore_state_ = false;
        }

        constexpr thread_id_ref_type const& get_next_thread() const noexcept
        {
            return next_thread_id_;
        }

        thread_id_ref_type move_next_thread() noexcept
        {
            return HPX_MOVE(next_thread_id_);
        }

    private:
        thread_data* thread_;
        thread_state prev_state_;
        thread_state orig_state_;
        thread_id_ref_type next_thread_id_;
        bool need_restore_state_;
    };
}    // namespace hpx::threads::detail
