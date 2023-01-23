//  Copyright (c) 2019-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is inspired by https://github.com/aprell/tasking-2.0

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/modules/concurrency.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/thread_support.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

namespace hpx::lcos::local {

    ////////////////////////////////////////////////////////////////////////////
    // A simple but very high performance implementation of the channel concept.
    // This channel is bounded to a size given at construction time and supports
    // multiple producers and multiple consumers. The data is stored in a
    // ring-buffer.
    template <typename T, typename Mutex = hpx::util::spinlock>
    class bounded_channel
    {
    private:
        using mutex_type = Mutex;

        constexpr bool is_full(std::size_t tail) const noexcept
        {
            std::size_t numitems = size_ + tail - head_.data_;
            if (numitems < size_)
            {
                return numitems == size_ - 1;
            }
            return numitems - size_ == size_ - 1;
        }

        constexpr bool is_empty(std::size_t head) const noexcept
        {
            return head == tail_.data_;
        }

    public:
        explicit bounded_channel(std::size_t size)
          : size_(size + 1)
          , buffer_(new T[size + 1])
          , closed_(false)
        {
            HPX_ASSERT(size != 0);

            head_.data_ = 0;
            tail_.data_ = 0;
        }

        bounded_channel(bounded_channel const& rhs) = delete;
        bounded_channel& operator=(bounded_channel const& rhs) = delete;

        bounded_channel(bounded_channel&& rhs) noexcept
          : head_(rhs.head_)
          , tail_(rhs.tail_)
          , size_(rhs.size_)
          , buffer_(HPX_MOVE(rhs.buffer_))
          , closed_(rhs.closed_)
        {
            rhs.size_ = 0;
            rhs.closed_ = true;
        }

        bounded_channel& operator=(bounded_channel&& rhs) noexcept
        {
            head_ = rhs.head_;
            tail_ = rhs.tail_;
            size_ = rhs.size_;
            buffer_ = HPX_MOVE(rhs.buffer_);
            closed_ = rhs.closed_;
            rhs.closed_ = true;
            return *this;
        }

        ~bounded_channel()
        {
            if (!closed_)
            {
                std::unique_lock<mutex_type> l(mtx_.data_);
                close(l);
            }
        }

        bool get(T* val = nullptr) const noexcept
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            if (closed_)
            {
                return false;
            }

            std::size_t head = head_.data_;

            if (is_empty(head))
            {
                return false;
            }

            if (val == nullptr)
            {
                return true;
            }

            *val = HPX_MOVE(buffer_[head]);
            if (++head >= size_)
            {
                head = 0;
            }
            head_.data_ = head;

            return true;
        }

        // clang-format off
        bool set(T&& t) noexcept(
            noexcept(std::declval<std::unique_lock<mutex_type>&>().lock()) &&
            noexcept(std::declval<std::unique_lock<mutex_type>&>().unlock()))
        // clang-format on
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            if (closed_)
            {
                return false;
            }

            std::size_t tail = tail_.data_;

            if (is_full(tail))
            {
                return false;
            }

            buffer_[tail] = HPX_MOVE(t);
            if (++tail >= size_)
            {
                tail = 0;
            }
            tail_.data_ = tail;

            return true;
        }

        std::size_t close()
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            return close(l);
        }

        constexpr std::size_t capacity() const noexcept
        {
            return size_ - 1;
        }

    protected:
        std::size_t close(std::unique_lock<mutex_type>& l)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            if (closed_)
            {
                l.unlock();
                HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                    "hpx::lcos::local::bounded_channel::close",
                    "attempting to close an already closed channel");
            }

            closed_ = true;
            return 0;
        }

    private:
        // keep the mutex, the head, and the tail pointer in separate cache
        // lines
        mutable hpx::util::cache_aligned_data<mutex_type> mtx_;
        mutable hpx::util::cache_aligned_data<std::size_t> head_;
        hpx::util::cache_aligned_data<std::size_t> tail_;

        // a channel of size n can buffer n-1 items
        std::size_t size_;

        // channel buffer
        std::unique_ptr<T[]> buffer_;

        // this channel was closed, i.e. no further operations are possible
        bool closed_;
    };

    ////////////////////////////////////////////////////////////////////////////
    // For use with HPX threads, the channel_mpmc defined here is the fastest
    // (even faster than the channel_spsc). Using hpx::util::spinlock as the
    // means of synchronization enables the use of this channel with non-HPX
    // threads, however the performance degrades by a factor of ten compared to
    // using hpx::spinlock.
    template <typename T>
    using channel_mpmc = bounded_channel<T, hpx::spinlock>;
}    // namespace hpx::lcos::local
