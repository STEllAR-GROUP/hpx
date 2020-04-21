//  Copyright (c) 2020 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/coroutines/thread_enums.hpp>
#include <hpx/basic_execution/agent_ref.hpp>
#include <hpx/basic_execution/this_thread.hpp>
#include <hpx/synchronization/futex.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/timing/steady_clock.hpp>
#include <hpx/assertion.hpp>
#include <hpx/thread_support/unlock_guard.hpp>

#include <mutex>

namespace hpx { namespace synchronization {

    namespace detail {
        // thread_entry is a singly linked list to maintain the waiting
        // threads, there's no memory management what so ever, thread_entry
        // is created on the stack.
        //
        // TODO: replace with a fully lockfree version...
        struct thread_entry
        {
            thread_entry(futex& f, hpx::basic_execution::agent_ref ctx) noexcept
                : ctx_(ctx)
                , prev_(nullptr)
                , f_(f)
            {
                std::lock_guard<futex::mutex_type> l(f_.mtx_);

                epoch_ = f_.epoch_;
                ++f_.num_references_;

                next_ = f_.thread_head_;
                // if this is not the first entry...
                if (next_ != nullptr)
                {
                    next_->prev_ = this;
                }

                f_.thread_head_ = this;
            }

            // returns true if we are in the same epoch. Being in the same epoch
            // means that we have to delete the entry from the queue. Note that
            // this only happens when we are woken up by the timeout and not by
            // another thread notification
            bool release() noexcept
            {
                std::lock_guard<futex::mutex_type> l(f_.mtx_);

                --f_.num_references_;

                std::size_t epoch(-1);
                std::swap(epoch, epoch_);

                if (epoch < f_.epoch_)
                    return false;

                HPX_ASSERT(!(epoch > f_.epoch_));

                // Delete this entry from the queue.

                // If first node
                if (this == f_.thread_head_)
                {
                    f_.thread_head_ = next_;
                }

                // If not last node
                if (next_ != nullptr)
                {
                    next_->prev_ = prev_;
                }

                // if middle node
                if (prev_ != nullptr)
                {
                    prev_->next_ = next_;
                }

                return true;
            }

            ~thread_entry() noexcept
            {
                if (epoch_ != std::size_t(-1))
                {
                    std::lock_guard<futex::mutex_type> l(f_.mtx_);
                    --f_.num_references_;
                }
            }

            hpx::basic_execution::agent_ref ctx_;
            thread_entry* prev_;
            thread_entry* next_;
            futex& f_;
            std::size_t epoch_;
        };
    }

    futex::futex() noexcept
       : thread_head_(nullptr)
       , num_references_(0)
       , epoch_(0)
    {
    }

    futex::~futex() noexcept
    {
        // synchronize with calls to wait, wait_xxx, notify and abort
        hpx::util::yield_while([this]()
            {
                // If there are remaining reference, try to abort threads that
                // happen to be still queued up.
                abort_all();

                // if there are no remaining references, we can safely break the
                // yield loop.
                {
                    std::unique_lock<mutex_type> l(mtx_);
                    if (num_references_ == 0)
                    {
                        return false;
                    }
                }

                return true;
            }, "hpx::synchronization::futex::~futex"
        );
    }

    threads::thread_state_ex_enum futex::wait(char const* reason)
    {
        auto this_ctx = hpx::basic_execution::this_thread::agent();
        detail::thread_entry entry(*this, this_ctx);

        this_ctx.suspend();

        return entry.release() ? hpx::threads::wait_timeout : hpx::threads::wait_signaled;
    }

    threads::thread_state_ex_enum futex::wait_until(
        util::steady_time_point const& abs_time, char const* reason)
    {
        auto this_ctx = hpx::basic_execution::this_thread::agent();
        detail::thread_entry entry(*this, this_ctx);

        this_ctx.sleep_until(abs_time.value());

        return entry.release() ? hpx::threads::wait_timeout : hpx::threads::wait_signaled;
    }

    void futex::notify_all()
    {
        detail::thread_entry *entry = nullptr;
        {
            std::unique_lock<mutex_type> l(mtx_);
            std::swap(entry, thread_head_);
            ++num_references_;
            ++epoch_;
        }

        for(;entry != nullptr; entry = entry->next_)
        {
            entry->ctx_.resume();
        }

        {
            std::unique_lock<mutex_type> l(mtx_);
            --num_references_;
        }
    }

    bool futex::notify_one()
    {
        detail::thread_entry *entry = nullptr;
        bool non_empty = true;
        {
            std::unique_lock<mutex_type> l(mtx_);
            ++epoch_;

            if (thread_head_ == nullptr)
                return false;

            entry = thread_head_;
            thread_head_ = entry->next_;

            if (thread_head_ == nullptr)
                non_empty = false;
            else
                thread_head_->prev_ = nullptr;

            ++num_references_;
        }

        auto ctx = entry->ctx_;
        ctx.resume();

        {
            std::unique_lock<mutex_type> l(mtx_);
            --num_references_;
        }
        return non_empty;
    }

    void futex::abort_all()
    {
        detail::thread_entry *entry = nullptr;
        {
            std::unique_lock<mutex_type> l(mtx_);
            std::swap(entry, thread_head_);
            ++num_references_;
            ++epoch_;
        }

        for(;entry != nullptr; entry = entry->next_)
        {
            entry->ctx_.abort();
        }

        {
            std::unique_lock<mutex_type> l(mtx_);
            --num_references_;
        }
    }

    bool futex::abort_one()
    {
        detail::thread_entry *entry = nullptr;
        bool non_empty = true;
        {
            std::unique_lock<mutex_type> l(mtx_);
            ++epoch_;

            if (thread_head_ == nullptr)
                return false;

            entry = thread_head_;
            thread_head_ = entry->next_;

            if (thread_head_ == nullptr)
                non_empty = false;
            else
                thread_head_->prev_ = nullptr;

            ++num_references_;
        }

        auto ctx = entry->ctx_;
        ctx.abort();

        {
            std::unique_lock<mutex_type> l(mtx_);
            --num_references_;
        }
        return non_empty;
    }
}}
