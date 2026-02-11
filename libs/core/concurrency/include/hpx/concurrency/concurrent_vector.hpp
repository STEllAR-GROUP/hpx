//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/spinlock.hpp>

#include <algorithm>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

namespace hpx::concurrent {

    template <typename T, typename Allocator = std::allocator<T>>
    class concurrent_vector
    {
    public:
        using value_type = T;
        using allocator_type = Allocator;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = T&;
        using const_reference = T const&;
        using pointer = typename std::allocator_traits<Allocator>::pointer;
        using const_pointer =
            typename std::allocator_traits<Allocator>::const_pointer;

        // Note: Iterators are invalidated by push_back/resize operations on
        // std::deque. Only references/pointers to elements are stable.
        using iterator = typename std::deque<T, Allocator>::iterator;
        using const_iterator =
            typename std::deque<T, Allocator>::const_iterator;
        using reverse_iterator =
            typename std::deque<T, Allocator>::reverse_iterator;
        using const_reverse_iterator =
            typename std::deque<T, Allocator>::const_reverse_iterator;

    private:
        mutable hpx::util::spinlock mutex_;
        std::deque<T, Allocator> data_;

    public:
        concurrent_vector() = default;

        explicit concurrent_vector(Allocator const& alloc)
          : data_(alloc)
        {
        }

        explicit concurrent_vector(size_type count, T const& value = T(),
            Allocator const& alloc = Allocator())
          : data_(count, value, alloc)
        {
        }

        concurrent_vector(concurrent_vector const& other)
          : data_(other.data_)
        {
        }

        concurrent_vector(concurrent_vector&& other) noexcept
          : data_(std::move(other.data_))
        {
        }

        concurrent_vector& operator=(concurrent_vector const& other)
        {
            if (this != &other)
            {
                std::lock(mutex_, other.mutex_);
                std::lock_guard<hpx::util::spinlock> lock(
                    mutex_, std::adopt_lock);
                std::lock_guard<hpx::util::spinlock> other_lock(
                    other.mutex_, std::adopt_lock);
                data_ = other.data_;
            }
            return *this;
        }

        concurrent_vector& operator=(concurrent_vector&& other) noexcept
        {
            if (this != &other)
            {
                std::lock_guard<hpx::util::spinlock> lock(mutex_);
                data_ = std::move(other.data_);
            }
            return *this;
        }

        // Element access
        reference operator[](size_type pos)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_[pos];
        }

        const_reference operator[](size_type pos) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_[pos];
        }

        reference at(size_type pos)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_.at(pos);
        }

        const_reference at(size_type pos) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_.at(pos);
        }

        reference front()
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_.front();
        }

        const_reference front() const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_.front();
        }

        reference back()
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_.back();
        }

        const_reference back() const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_.back();
        }

        // Iterators
        // Note: Iterators are not thread-safe if modification happens.
        iterator begin() noexcept
        {
            return data_.begin();
        }
        const_iterator begin() const noexcept
        {
            return data_.begin();
        }
        const_iterator cbegin() const noexcept
        {
            return data_.cbegin();
        }
        iterator end() noexcept
        {
            return data_.end();
        }
        const_iterator end() const noexcept
        {
            return data_.end();
        }
        const_iterator cend() const noexcept
        {
            return data_.cend();
        }

        // Capacity
        bool empty() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_.empty();
        }

        size_type size() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return data_.size();
        }

        size_type max_size() const noexcept
        {
            return data_.max_size();
        }

        void reserve(size_type /*new_cap*/)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            // std::deque doesn't have reserve(), but we provide API compatibility
            // It might shrink_to_fit or do nothing.
            // data_.reserve(new_cap); // deque has no reserve
        }

        size_type capacity() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            // deque has no capacity, size is approx capacity
            return data_.size();
        }

        void shrink_to_fit()
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            data_.shrink_to_fit();
        }

        // Modifiers
        void clear() noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            data_.clear();
        }

        void push_back(T const& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            data_.push_back(value);
        }

        void push_back(T&& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            data_.push_back(std::move(value));
        }

        iterator grow_by(size_type n)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            auto old_size = data_.size();
            data_.resize(old_size + n);
            return data_.begin() + old_size;
        }

        iterator grow_by(size_type n, T const& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            auto old_size = data_.size();
            data_.resize(old_size + n, value);
            return data_.begin() + old_size;
        }

        // Additional TBB-like methods can be added as needed.
    };

}    // namespace hpx::concurrent
