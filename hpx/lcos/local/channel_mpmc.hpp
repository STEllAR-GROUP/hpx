//  Copyright (c) 2019 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//  This work is inspired by https://github.com/aprell/tasking-2.0

#if !defined(HPX_LCOS_LOCAL_CHANNEL_MPMC_NOV_24_2019_1141AM)
#define HPX_LCOS_LOCAL_CHANNEL_MPMC_NOV_24_2019_1141AM

#include <hpx/config.hpp>
#include <hpx/concurrency.hpp>
#include <hpx/errors.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/thread_support.hpp>

#include <cstddef>
#include <memory>
#include <mutex>

namespace hpx { namespace lcos { namespace local {

    ////////////////////////////////////////////////////////////////////////////
    // A simple but very high performance implementation of the channel concept.
    // This channel is bounded to a size given at construction time and supports
    // multiple producers and multiple consumers. The data is stored in a
    // ring-buffer.
    template <typename T>
    class channel_mpmc
    {
    private:
        using mutex_type = hpx::lcos::local::spinlock;

        std::size_t num_items() const noexcept
        {
            std::size_t numitems = size_ + tail_.data_ - head_.data_;
            if (numitems < size_)
            {
                return numitems;
            }
            return numitems - size_;
        }

        bool is_full() const noexcept
        {
            return num_items() == size_ - 1;
        }

        bool is_empty() const noexcept
        {
            return head_.data_ == tail_.data_;
        }

    public:
        explicit channel_mpmc(std::size_t size)
          : size_(size + 1)
          , buffer_(new T[size + 1])
          , closed_(false)
        {
            head_.data_ = 0;
            tail_.data_ = 0;
        }

        ~channel_mpmc()
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            if (!closed_)
            {
                close(l);
            }
        }

        T get() const
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            if (is_empty())
            {
                if (closed_)
                {
                    l.unlock();
                    HPX_THROW_EXCEPTION(hpx::invalid_status,
                        "hpx::lcos::local::channel_mpmc::get_sync",
                        "this channel is empty and was closed");
                }
                else
                {
                    l.unlock();
                    HPX_THROW_EXCEPTION(hpx::invalid_status,
                        "hpx::lcos::local::channel_mpmc::get_sync",
                        "this channel is empty");
                }
            }

            std::size_t head = head_.data_;
            T result = std::move(buffer_[head]);
            if (++head < size_)
            {
                head_.data_ = head;
            }
            else
            {
                head_.data_ = head - size_;
            }
            return result;
        }

        bool try_get(T* val = nullptr) const
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            if (is_empty())
            {
                return false;
            }

            if (val == nullptr)
            {
                return true;
            }

            std::size_t head = head_.data_;
            *val = std::move(buffer_[head]);
            if (++head < size_)
            {
                head_.data_ = head;
            }
            else
            {
                head_.data_ = head - size_;
            }
            return true;
        }

        bool set(T&& t)
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            if (closed_ || is_full())
            {
                return false;
            }

            std::size_t tail = tail_.data_;
            buffer_[tail] = std::move(t);
            if (++tail < size_)
            {
                tail_.data_ = tail;
            }
            else
            {
                tail_.data_ = tail - size_;
            }
            return true;
        }

        std::size_t close()
        {
            std::unique_lock<mutex_type> l(mtx_.data_);
            return close(l);
        }

    protected:
        std::size_t close(std::unique_lock<mutex_type>& l)
        {
            HPX_ASSERT_OWNS_LOCK(l);

            if (closed_)
            {
                l.unlock();
                HPX_THROW_EXCEPTION(hpx::invalid_status,
                    "hpx::lcos::local::channel_mpmc::close",
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
        std::size_t const size_;

        // channel buffer
        std::unique_ptr<T[]> const buffer_;

        bool closed_;
    };
}}}    // namespace hpx::lcos::local

#endif
