//  Copyright (c) 2019-2024 Hartmut Kaiser
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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

namespace hpx::lcos::local {

    ////////////////////////////////////////////////////////////////////////////
    // A simple but very high performance implementation of the channel concept.
    // This channel is bounded to a size given at construction time and supports
    // a single producer and a single consumer. The data is stored in a
    // ring-buffer.
    HPX_CXX_CORE_EXPORT enum class channel_mode : std::uint8_t {
        normal,
        dont_support_close
    };

    HPX_CXX_CORE_EXPORT template <typename T,
        channel_mode = channel_mode::normal>
    class channel_spsc
    {
    private:
        [[nodiscard]] HPX_FORCEINLINE bool is_full(
            std::size_t tail) noexcept
        {
            std::size_t next_tail = tail + 1;
            if (next_tail >= size_)
                next_tail = 0;
            // Fast path: head_cached lives on the producer's own cache line,
            // so this comparison never touches the consumer's cache line.
            // head_cached is a conservative snapshot of consumer's head:
            // the consumer only moves head forward, so head_cached <= real_head
            // (in monotonic terms).  When next_tail != head_cached the buffer
            // cannot be full — the consumer has freed at least as many slots as
            // head_cached indicates, leaving room to write.
            if (next_tail != producer_.head_cached) [[likely]]
            {
                return false;
            }
            // Slow path: cached value says the buffer might be full; refresh
            // with an acquire load from the consumer's cache line and recheck.
            producer_.head_cached =
                consumer_.head.load(std::memory_order_acquire);
            return next_tail == producer_.head_cached;
        }

        [[nodiscard]] HPX_FORCEINLINE bool is_empty(
            std::size_t head) const noexcept
        {
            // Fast path: consumer_.tail_cached lives on the consumer's own
            // cache line, so this comparison never touches the producer's
            // cache line.  consumer_.tail_cached is a conservative lower-bound
            // on the real tail (producer only moves it forward), so when
            // head != tail_cached the buffer definitely has items and we can
            // return false without any cross-thread acquire.
            if (head != consumer_.tail_cached) [[likely]]
            {
                return false;
            }
            // Slow path: the cached value says the buffer might be empty;
            // refresh it with an acquire load from the producer's cache line
            // and recheck.
            consumer_.tail_cached =
                producer_.tail.load(std::memory_order_acquire);
            return head == consumer_.tail_cached;
        }

    public:
        explicit channel_spsc(std::size_t size)
          : size_(size + 1)
          , buffer_(new T[size + 1])
          , closed_(false)
        {
            HPX_ASSERT(size != 0);

            consumer_.head.store(0, std::memory_order_relaxed);
            producer_.tail.store(0, std::memory_order_relaxed);
        }

        channel_spsc(channel_spsc&& rhs) noexcept
          : size_(rhs.size_)
          , buffer_(HPX_MOVE(rhs.buffer_))
        {
            consumer_.head.store(
                rhs.consumer_.head.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            consumer_.tail_cached = rhs.consumer_.tail_cached;

            producer_.tail.store(
                rhs.producer_.tail.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            producer_.head_cached = rhs.producer_.head_cached;

            closed_.store(rhs.closed_.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            rhs.closed_.store(true, std::memory_order_release);
        }

        channel_spsc(channel_spsc const& rhs) = delete;
        channel_spsc& operator=(channel_spsc const& rhs) = delete;

        channel_spsc& operator=(channel_spsc&& rhs) noexcept
        {
            consumer_.head.store(
                rhs.consumer_.head.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            consumer_.tail_cached = rhs.consumer_.tail_cached;

            producer_.tail.store(
                rhs.producer_.tail.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            producer_.head_cached = rhs.producer_.head_cached;

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

        [[nodiscard]] bool is_empty() const noexcept
        {
            if (closed_.load(std::memory_order_relaxed))
            {
                return true;
            }
            return is_empty(consumer_.head.load(std::memory_order_relaxed));
        }

        bool get(T* val = nullptr) const noexcept
        {
            if (closed_.load(std::memory_order_relaxed))
            {
                return false;
            }

            std::size_t head = consumer_.head.load(std::memory_order_relaxed);

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
            consumer_.head.store(head, std::memory_order_release);

            return true;
        }

        bool set(T&& t) noexcept
        {
            if (closed_.load(std::memory_order_relaxed))
            {
                return false;
            }

            std::size_t tail = producer_.tail.load(std::memory_order_relaxed);

            if (is_full(tail))
            {
                return false;
            }

            buffer_[tail] = HPX_MOVE(t);
            if (++tail >= size_)
            {
                tail = 0;
            }
            producer_.tail.store(tail, std::memory_order_release);

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

        [[nodiscard]] constexpr std::size_t capacity() const noexcept
        {
            return size_ - 1;
        }

    private:
        // The consumer thread exclusively reads/writes consumer_:
        //   head       - advanced on every successful get(); the producer never
        //                writes this field.
        //   tail_cached - stale copy of the producer's tail; refreshed lazily
        //                 only when the buffer appears empty, so the consumer
        //                 avoids crossing to the producer's cache line on every
        //                 is_empty() check.
        // Both fields share one cache-line (alignas guarantees the struct's
        // size is padded to a full cache line), so all consumer hot-path
        // accesses stay local to a single cache line.
        // Reference: https://doi.org/10.1109/IPDPS.2010.5470368
        struct alignas(std::hardware_destructive_interference_size)
            consumer_state
        {
            std::atomic<std::size_t> head{0};
            // consumer's lazy copy of the producer's tail index;
            // only the consumer thread reads and writes this field
            std::size_t tail_cached{0};
        };
        static_assert(sizeof(consumer_state) <=
                std::hardware_destructive_interference_size,
            "consumer_state exceeds one cache line; false sharing possible");

        // The producer thread exclusively reads/writes producer_:
        //   tail       - advanced on every successful set(); the consumer never
        //                writes this field.
        //   head_cached - stale copy of the consumer's head; refreshed lazily
        //                 only when the buffer appears full, so the producer
        //                 avoids crossing to the consumer's cache line on every
        //                 is_full() check.
        // Both fields share one cache-line, so all producer hot-path accesses
        // stay local to a single cache line.
        // Reference: https://doi.org/10.1109/IPDPS.2010.5470368
        struct alignas(std::hardware_destructive_interference_size)
            producer_state
        {
            std::atomic<std::size_t> tail{0};
            // producer's lazy copy of the consumer's head index;
            // only the producer thread reads and writes this field
            std::size_t head_cached{0};
        };
        static_assert(sizeof(producer_state) <=
                std::hardware_destructive_interference_size,
            "producer_state exceeds one cache line; false sharing possible");

        mutable consumer_state consumer_;
        producer_state producer_;

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
    HPX_CXX_CORE_EXPORT template <typename T>
    class channel_spsc<T, channel_mode::dont_support_close>
    {
    private:
        [[nodiscard]] HPX_FORCEINLINE bool is_full(
            std::size_t tail) noexcept
        {
            std::size_t next_tail = tail + 1;
            if (next_tail >= size_)
                next_tail = 0;
            // Fast path: head_cached lives on the producer's own cache line,
            // so this comparison never touches the consumer's cache line.
            // head_cached is a conservative snapshot of consumer's head:
            // the consumer only moves head forward, so head_cached <= real_head
            // (in monotonic terms).  When next_tail != head_cached the buffer
            // cannot be full — the consumer has freed at least as many slots as
            // head_cached indicates, leaving room to write.
            if (next_tail != producer_.head_cached) [[likely]]
            {
                return false;
            }
            // Slow path: cached value says the buffer might be full; refresh
            // with an acquire load from the consumer's cache line and recheck.
            producer_.head_cached =
                consumer_.head.load(std::memory_order_acquire);
            return next_tail == producer_.head_cached;
        }

        [[nodiscard]] HPX_FORCEINLINE bool is_empty(
            std::size_t head) const noexcept
        {
            // Fast path: consumer_.tail_cached lives on the consumer's own
            // cache line, so this comparison never touches the producer's
            // cache line.  consumer_.tail_cached is a conservative lower-bound
            // on the real tail (producer only moves it forward), so when
            // head != tail_cached the buffer definitely has items and we can
            // return false without any cross-thread acquire.
            if (head != consumer_.tail_cached) [[likely]]
            {
                return false;
            }
            // Slow path: the cached value says the buffer might be empty;
            // refresh it with an acquire load from the producer's cache line
            // and recheck.
            consumer_.tail_cached =
                producer_.tail.load(std::memory_order_acquire);
            return head == consumer_.tail_cached;
        }

    public:
        explicit channel_spsc(std::size_t size)
          : size_(size + 1)
          , buffer_(new T[size + 1])
        {
            HPX_ASSERT(size != 0);

            consumer_.head.store(0, std::memory_order_relaxed);
            producer_.tail.store(0, std::memory_order_relaxed);
        }

        channel_spsc(channel_spsc const& rhs) = delete;
        channel_spsc& operator=(channel_spsc const& rhs) = delete;

        channel_spsc(channel_spsc&& rhs) noexcept
          : size_(rhs.size_)
          , buffer_(HPX_MOVE(rhs.buffer_))
        {
            consumer_.head.store(
                rhs.consumer_.head.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            consumer_.tail_cached = rhs.consumer_.tail_cached;

            producer_.tail.store(
                rhs.producer_.tail.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            producer_.head_cached = rhs.producer_.head_cached;
        }

        channel_spsc& operator=(channel_spsc&& rhs) noexcept
        {
            consumer_.head.store(
                rhs.consumer_.head.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            consumer_.tail_cached = rhs.consumer_.tail_cached;

            producer_.tail.store(
                rhs.producer_.tail.load(std::memory_order_acquire),
                std::memory_order_relaxed);
            producer_.head_cached = rhs.producer_.head_cached;

            size_ = rhs.size_;
            buffer_ = HPX_MOVE(rhs.buffer_);

            return *this;
        }

        ~channel_spsc() = default;

        [[nodiscard]] bool is_empty() const noexcept
        {
            return is_empty(consumer_.head.load(std::memory_order_relaxed));
        }

        bool get(T* val = nullptr) const noexcept
        {
            std::size_t head = consumer_.head.load(std::memory_order_relaxed);

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
            consumer_.head.store(head, std::memory_order_release);

            return true;
        }

        bool set(T&& t) noexcept
        {
            std::size_t tail = producer_.tail.load(std::memory_order_relaxed);

            if (is_full(tail))
            {
                return false;
            }

            buffer_[tail] = HPX_MOVE(t);
            if (++tail >= size_)
            {
                tail = 0;
            }
            producer_.tail.store(tail, std::memory_order_release);

            return true;
        }

        [[nodiscard]] constexpr std::size_t capacity() const noexcept
        {
            return size_ - 1;
        }

    private:
        // The consumer thread exclusively reads/writes consumer_:
        //   head       - advanced on every successful get(); the producer never
        //                writes this field.
        //   tail_cached - stale copy of the producer's tail; refreshed lazily
        //                 only when the buffer appears empty, so the consumer
        //                 avoids crossing to the producer's cache line on every
        //                 is_empty() check.
        // Both fields share one cache-line (alignas guarantees the struct's
        // size is padded to a full cache line), so all consumer hot-path
        // accesses stay local to a single cache line.
        // Reference: https://doi.org/10.1109/IPDPS.2010.5470368
        struct alignas(std::hardware_destructive_interference_size)
            consumer_state
        {
            std::atomic<std::size_t> head{0};
            // consumer's lazy copy of the producer's tail index;
            // only the consumer thread reads and writes this field
            std::size_t tail_cached{0};
        };
        static_assert(sizeof(consumer_state) <=
                std::hardware_destructive_interference_size,
            "consumer_state exceeds one cache line; false sharing possible");

        // The producer thread exclusively reads/writes producer_:
        //   tail       - advanced on every successful set(); the consumer never
        //                writes this field.
        //   head_cached - stale copy of the consumer's head; refreshed lazily
        //                 only when the buffer appears full, so the producer
        //                 avoids crossing to the consumer's cache line on every
        //                 is_full() check.
        // Both fields share one cache-line, so all producer hot-path accesses
        // stay local to a single cache line.
        // Reference: https://doi.org/10.1109/IPDPS.2010.5470368
        struct alignas(std::hardware_destructive_interference_size)
            producer_state
        {
            std::atomic<std::size_t> tail{0};
            // producer's lazy copy of the consumer's head index;
            // only the producer thread reads and writes this field
            std::size_t head_cached{0};
        };
        static_assert(sizeof(producer_state) <=
                std::hardware_destructive_interference_size,
            "producer_state exceeds one cache line; false sharing possible");

        mutable consumer_state consumer_;
        producer_state producer_;

        // a channel of size n can buffer n-1 items
        std::size_t size_;

        // channel buffer
        std::unique_ptr<T[]> buffer_;
    };
}    // namespace hpx::lcos::local
