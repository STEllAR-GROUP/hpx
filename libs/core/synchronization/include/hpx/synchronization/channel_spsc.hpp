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
#include <hpx/type_support/construct_at.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace hpx::lcos::local {

    ////////////////////////////////////////////////////////////////////////////
    // A simple but very high performance implementation of the channel concept.
    // This channel is bounded to a size given at construction time and supports
    // a single producer and a single consumer. The data is stored in a
    // ring-buffer.
    enum class channel_mode : std::uint8_t
    {
        normal,
        dont_support_close
    };

    template <typename T, channel_mode = channel_mode::normal>
    class channel_spsc
    {
    private:
        HPX_FORCEINLINE bool is_full(std::size_t tail) const noexcept
        {
            std::size_t numitems =
                size_ + tail - head_.data_.load(std::memory_order_acquire);

            if (numitems < size_)
            {
                return numitems == size_ - 1;
            }
            return (numitems - size_ == size_ - 1);
        }

        HPX_FORCEINLINE bool is_empty(std::size_t head) const noexcept
        {
            return head == tail_.data_.load(std::memory_order_acquire);
        }

    public:
        explicit channel_spsc(std::size_t size)
          : size_(size + 1)
          , buffer_(new T[size + 1])
          , closed_(false)
        {
            HPX_ASSERT(size != 0);

            head_.data_.store(0, std::memory_order_relaxed);
            tail_.data_.store(0, std::memory_order_relaxed);
        }

        channel_spsc(channel_spsc&& rhs) noexcept
          : size_(rhs.size_)
          , buffer_(HPX_MOVE(rhs.buffer_))
        {
            head_.data_.store(rhs.head_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            tail_.data_.store(rhs.tail_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);

            closed_.store(rhs.closed_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            rhs.closed_.store(true, std::memory_order_release);
        }

        channel_spsc(channel_spsc const& rhs) = delete;
        channel_spsc& operator=(channel_spsc const& rhs) = delete;

        channel_spsc& operator=(channel_spsc&& rhs) noexcept
        {
            head_.data_.store(rhs.head_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            tail_.data_.store(rhs.tail_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);

            size_ = rhs.size_;
            buffer_ = HPX_MOVE(rhs.buffer_);

            closed_.store(rhs.closed_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            rhs.closed_.store(true, std::memory_order_release);

            return *this;
        }

        ~channel_spsc()
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

        bool set(T&& t) noexcept
        {
            if (closed_.load(std::memory_order_relaxed))
            {
                return false;
            }

            std::size_t tail = tail_.data_.load(std::memory_order_relaxed);

            if (is_full(tail))
            {
                return false;
            }

            buffer_[tail] = HPX_MOVE(t);
            if (++tail >= size_)
            {
                tail = 0;
            }
            tail_.data_.store(tail, std::memory_order_release);

            return true;
        }

        std::size_t close()
        {
            bool expected = false;
            if (!closed_.compare_exchange_weak(expected, true))
            {
                HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                    "hpx::lcos::local::channel_spsc::close",
                    "attempting to close an already closed channel");
            }
            return 0;
        }

        constexpr std::size_t capacity() const noexcept
        {
            return size_ - 1;
        }

    private:
        // keep the head and the tail pointer in separate cache lines
        mutable hpx::util::cache_aligned_data<std::atomic<std::size_t>> head_;
        hpx::util::cache_aligned_data<std::atomic<std::size_t>> tail_;

        // a channel of size n can buffer n-1 items
        std::size_t size_;

        // channel buffer
        std::unique_ptr<T[]> buffer_;

        // this channel was closed, i.e. no further operations are possible
        std::atomic<bool> closed_;
    };

    // Same as above, except that the channel doesn't support close(). This is
    // an optimization that can be applied in use cases where the surrounding
    // code ensures that the channel is always in proper state.
    template <typename T>
    class channel_spsc<T, channel_mode::dont_support_close>
    {
    private:
        HPX_FORCEINLINE bool is_full(std::size_t tail) const noexcept
        {
            std::size_t numitems =
                size_ + tail - head_.data_.load(std::memory_order_acquire);

            if (numitems < size_)
            {
                return numitems == size_ - 1;
            }
            return (numitems - size_ == size_ - 1);
        }

        HPX_FORCEINLINE bool is_empty(std::size_t head) const noexcept
        {
            return head == tail_.data_.load(std::memory_order_acquire);
        }

    public:
        explicit channel_spsc(std::size_t size)
          : size_(size + 1)
          , buffer_(new T[size + 1])
        {
            HPX_ASSERT(size != 0);

            head_.data_.store(0, std::memory_order_relaxed);
            tail_.data_.store(0, std::memory_order_relaxed);
        }

        channel_spsc(channel_spsc&& rhs) noexcept
          : size_(rhs.size_)
          , buffer_(HPX_MOVE(rhs.buffer_))
        {
            head_.data_.store(rhs.head_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            tail_.data_.store(rhs.tail_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
        }

        channel_spsc& operator=(channel_spsc&& rhs) noexcept
        {
            head_.data_.store(rhs.head_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            tail_.data_.store(rhs.tail_.data_.load(std::memory_order_acquire),
                std::memory_order_relaxed);

            size_ = rhs.size_;
            buffer_ = HPX_MOVE(rhs.buffer_);

            return *this;
        }

        ~channel_spsc() = default;

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

        bool set(T&& t) noexcept
        {
            std::size_t tail = tail_.data_.load(std::memory_order_relaxed);

            if (is_full(tail))
            {
                return false;
            }

            buffer_[tail] = HPX_MOVE(t);
            if (++tail >= size_)
            {
                tail = 0;
            }
            tail_.data_.store(tail, std::memory_order_release);

            return true;
        }

        constexpr std::size_t capacity() const noexcept
        {
            return size_ - 1;
        }

    private:
        // keep the head and the tail pointer in separate cache lines
        mutable hpx::util::cache_aligned_data<std::atomic<std::size_t>> head_;
        hpx::util::cache_aligned_data<std::atomic<std::size_t>> tail_;

        // a channel of size n can buffer n-1 items
        std::size_t size_;

        // channel buffer
        std::unique_ptr<T[]> buffer_;
    };
}    // namespace hpx::lcos::local
