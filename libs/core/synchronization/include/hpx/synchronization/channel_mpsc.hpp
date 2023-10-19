//  Copyright (c) 2019-2023 Hartmut Kaiser
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
#include <hpx/synchronization/channel_spsc.hpp>
#include <hpx/synchronization/spinlock.hpp>
#include <hpx/type_support/construct_at.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

namespace hpx::lcos::local {

    ////////////////////////////////////////////////////////////////////////////
    // A simple but very high performance implementation of the channel concept.
    // This channel is bounded to a size given at construction time and supports
    // a multiple producers and a single consumer. The data is stored in a
    // ring-buffer.
    template <typename T, typename Mutex = hpx::util::spinlock,
        channel_mode = channel_mode::normal>
    class base_channel_mpsc
    {
    private:
        using mutex_type = Mutex;

        bool is_full(std::size_t tail) const noexcept
        {
            std::size_t const numitems =
                size_ + tail - head_.data_.load(std::memory_order_acquire);

            if (numitems < size_)
            {
                return numitems == size_ - 1;
            }
            return (numitems - size_ == size_ - 1);
        }

        bool is_empty(std::size_t head) const noexcept
        {
            return head == tail_.data_.tail_.load(std::memory_order_relaxed);
        }

    public:
        explicit base_channel_mpsc(std::size_t size)
          : size_(size + 1)
          , buffer_(new T[size + 1])
          , closed_(false)
        {
            HPX_ASSERT(size != 0);

            head_.data_.store(0, std::memory_order_relaxed);
            tail_.data_.tail_.store(0, std::memory_order_relaxed);
        }

        base_channel_mpsc(base_channel_mpsc const& rhs) = delete;
        base_channel_mpsc& operator=(base_channel_mpsc const& rhs) = delete;

        base_channel_mpsc(base_channel_mpsc&& rhs) noexcept
          : size_(rhs.size_)
          , buffer_(HPX_MOVE(rhs.buffer_))
        {
            head_.data_.store(rhs.head_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            tail_.data_.tail_.store(
                rhs.tail_.data_.tail_.load(std::memory_order_acquire),
                std::memory_order_relaxed);

            closed_.store(rhs.closed_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            rhs.closed_.store(true, std::memory_order_release);
        }

        base_channel_mpsc& operator=(base_channel_mpsc&& rhs) noexcept
        {
            head_.data_.store(rhs.head_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            tail_.data_.tail_.store(
                rhs.tail_.data_.tail_.load(std::memory_order_acquire),
                std::memory_order_relaxed);

            size_ = rhs.size_;
            buffer_ = HPX_MOVE(rhs.buffer_);

            closed_.store(rhs.closed_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            rhs.closed_.store(true, std::memory_order_release);

            return *this;
        }

        ~base_channel_mpsc()
        {
            if (!closed_.load(std::memory_order_relaxed))
            {
                close();
            }
        }

        bool get(T* val = nullptr) const noexcept
        {
            if (closed_.load(std::memory_order_relaxed))
            {
                return false;
            }

            std::size_t head = head_.data_.load(std::memory_order_relaxed);

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
            head_.data_.store(head, std::memory_order_release);

            return true;
        }

        // clang-format off
        bool set(T&& t) noexcept(
            noexcept(std::declval<std::unique_lock<mutex_type>&>().lock()) &&
            noexcept(std::declval<std::unique_lock<mutex_type>&>().unlock()))
        // clang-format on
        {
            if (closed_.load(std::memory_order_relaxed))
            {
                return false;
            }

            std::unique_lock<mutex_type> l(tail_.data_.mtx_);

            std::size_t tail =
                tail_.data_.tail_.load(std::memory_order_acquire);

            if (is_full(tail))
            {
                return false;
            }

            buffer_[tail] = HPX_MOVE(t);
            if (++tail >= size_)
            {
                tail = 0;
            }
            tail_.data_.tail_.store(tail, std::memory_order_relaxed);

            return true;
        }

        std::size_t close()
        {
            bool expected = false;
            if (!closed_.compare_exchange_weak(expected, true))
            {
                HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                    "hpx::lcos::local::base_channel_mpsc::close",
                    "attempting to close an already closed channel");
            }
            return 0;
        }

        constexpr std::size_t capacity() const noexcept
        {
            return size_ - 1;
        }

    private:
        // keep the mutex with the tail and the head pointer in separate cache
        // lines
        struct tail_data
        {
            mutex_type mtx_;
            std::atomic<std::size_t> tail_;
        };

        mutable hpx::util::cache_aligned_data<std::atomic<std::size_t>> head_;
        hpx::util::cache_aligned_data<tail_data> tail_;

        // a channel of size n can buffer n-1 items
        std::size_t size_;

        // channel buffer
        std::unique_ptr<T[]> buffer_;

        // this channel was closed, i.e. no further operations are possible
        std::atomic<bool> closed_;
    };

    template <typename T, typename Mutex>
    class base_channel_mpsc<T, Mutex, channel_mode::dont_support_close>
    {
    private:
        using mutex_type = Mutex;

        bool is_full(std::size_t tail) const noexcept
        {
            std::size_t const numitems =
                size_ + tail - head_.data_.load(std::memory_order_acquire);

            if (numitems < size_)
            {
                return numitems == size_ - 1;
            }
            return (numitems - size_ == size_ - 1);
        }

        bool is_empty(std::size_t head) const noexcept
        {
            return head == tail_.data_.tail_.load(std::memory_order_relaxed);
        }

    public:
        explicit base_channel_mpsc(std::size_t size)
          : size_(size + 1)
          , buffer_(new T[size + 1])
        {
            HPX_ASSERT(size != 0);

            head_.data_.store(0, std::memory_order_relaxed);
            tail_.data_.tail_.store(0, std::memory_order_relaxed);
        }

        base_channel_mpsc(base_channel_mpsc&& rhs) noexcept
          : size_(rhs.size_)
          , buffer_(HPX_MOVE(rhs.buffer_))
        {
            head_.data_.store(rhs.head_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            tail_.data_.tail_.store(
                rhs.tail_.data_.tail_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
        }

        base_channel_mpsc& operator=(base_channel_mpsc&& rhs) noexcept
        {
            head_.data_.store(rhs.head_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            tail_.data_.tail_.store(
                rhs.tail_.data_.tail_.load(std::memory_order_acquire),
                std::memory_order_relaxed);

            size_ = rhs.size_;
            buffer_ = HPX_MOVE(rhs.buffer_);

            return *this;
        }

        ~base_channel_mpsc() = default;

        bool get(T* val = nullptr) const noexcept
        {
            std::size_t head = head_.data_.load(std::memory_order_relaxed);

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
            head_.data_.store(head, std::memory_order_release);

            return true;
        }

        // clang-format off
        bool set(T&& t) noexcept(
            noexcept(std::declval<std::unique_lock<mutex_type>&>().lock()) &&
            noexcept(std::declval<std::unique_lock<mutex_type>&>().unlock()))
        // clang-format on
        {
            std::unique_lock<mutex_type> l(tail_.data_.mtx_);

            std::size_t tail =
                tail_.data_.tail_.load(std::memory_order_acquire);

            if (is_full(tail))
            {
                return false;
            }

            buffer_[tail] = HPX_MOVE(t);
            if (++tail >= size_)
            {
                tail = 0;
            }
            tail_.data_.tail_.store(tail, std::memory_order_relaxed);

            return true;
        }

        constexpr std::size_t capacity() const noexcept
        {
            return size_ - 1;
        }

    private:
        // keep the mutex with the tail and the head pointer in separate cache
        // lines
        struct tail_data
        {
            mutex_type mtx_;
            std::atomic<std::size_t> tail_;
        };

        mutable hpx::util::cache_aligned_data<std::atomic<std::size_t>> head_;
        hpx::util::cache_aligned_data<tail_data> tail_;

        // a channel of size n can buffer n-1 items
        std::size_t size_;

        // channel buffer
        std::unique_ptr<T[]> buffer_;
    };

    ////////////////////////////////////////////////////////////////////////////
    // Using hpx::util::spinlock as the means of synchronization enables the use
    // of this channel with non-HPX threads.
    template <typename T, channel_mode Mode = channel_mode::normal>
    using channel_mpsc = base_channel_mpsc<T, hpx::spinlock, Mode>;
}    // namespace hpx::lcos::local
