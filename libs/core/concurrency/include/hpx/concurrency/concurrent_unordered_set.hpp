//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/spinlock.hpp>

#include <algorithm>
#include <functional>
#include <mutex>
#include <unordered_set>
#include <utility>

namespace hpx::concurrent {

    template <typename Key, typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key>,
        typename Allocator = std::allocator<Key>>
    class concurrent_unordered_set
    {
    public:
        using key_type = Key;
        using value_type = Key;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using hasher = Hash;
        using key_equal = KeyEqual;
        using allocator_type = Allocator;
        using reference = value_type&;
        using const_reference = value_type const&;
        using pointer = typename std::allocator_traits<Allocator>::pointer;
        using const_pointer =
            typename std::allocator_traits<Allocator>::const_pointer;

        using iterator = typename std::unordered_set<Key, Hash, KeyEqual,
            Allocator>::iterator;
        using const_iterator = typename std::unordered_set<Key, Hash, KeyEqual,
            Allocator>::const_iterator;
        using local_iterator = typename std::unordered_set<Key, Hash, KeyEqual,
            Allocator>::local_iterator;
        using const_local_iterator = typename std::unordered_set<Key, Hash,
            KeyEqual, Allocator>::const_local_iterator;

    private:
        mutable hpx::util::spinlock mutex_;
        std::unordered_set<Key, Hash, KeyEqual, Allocator> set_;

    public:
        concurrent_unordered_set() = default;

        explicit concurrent_unordered_set(size_type bucket_count,
            Hash const& hash = Hash(), KeyEqual const& equal = KeyEqual(),
            Allocator const& alloc = Allocator())
          : set_(bucket_count, hash, equal, alloc)
        {
        }

        explicit concurrent_unordered_set(Allocator const& alloc)
          : set_(alloc)
        {
        }

        concurrent_unordered_set(concurrent_unordered_set const& other)
          : set_(other.set_)
        {
        }

        concurrent_unordered_set(concurrent_unordered_set&& other) noexcept
          : set_(std::move(other.set_))
        {
        }

        concurrent_unordered_set& operator=(
            concurrent_unordered_set const& other)
        {
            if (this != &other)
            {
                std::lock(mutex_, other.mutex_);
                std::lock_guard<hpx::util::spinlock> lock(
                    mutex_, std::adopt_lock);
                std::lock_guard<hpx::util::spinlock> other_lock(
                    other.mutex_, std::adopt_lock);
                set_ = other.set_;
            }
            return *this;
        }

        concurrent_unordered_set& operator=(
            concurrent_unordered_set&& other) noexcept
        {
            if (this != &other)
            {
                std::lock_guard<hpx::util::spinlock> lock(mutex_);
                set_ = std::move(other.set_);
            }
            return *this;
        }

        // Capacity
        bool empty() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.empty();
        }

        size_type size() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.size();
        }

        size_type max_size() const noexcept
        {
            return set_.max_size();
        }

        // Iterators
        iterator begin() noexcept
        {
            return set_.begin();
        }
        const_iterator begin() const noexcept
        {
            return set_.begin();
        }
        const_iterator cbegin() const noexcept
        {
            return set_.cbegin();
        }
        iterator end() noexcept
        {
            return set_.end();
        }
        const_iterator end() const noexcept
        {
            return set_.end();
        }
        const_iterator cend() const noexcept
        {
            return set_.cend();
        }

        // Modifiers
        void clear() noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            set_.clear();
        }

        std::pair<iterator, bool> insert(value_type const& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.insert(value);
        }

        std::pair<iterator, bool> insert(value_type&& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.insert(std::move(value));
        }

        iterator erase(const_iterator pos)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.erase(pos);
        }

        size_type erase(Key const& key)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.erase(key);
        }

        void swap(concurrent_unordered_set& other) noexcept
        {
            if (this != &other)
            {
                std::lock(mutex_, other.mutex_);
                std::lock_guard<hpx::util::spinlock> lock(
                    mutex_, std::adopt_lock);
                std::lock_guard<hpx::util::spinlock> other_lock(
                    other.mutex_, std::adopt_lock);
                set_.swap(other.set_);
            }
        }

        // Lookup
        size_type count(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.count(key);
        }

        iterator find(Key const& key)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.find(key);
        }

        const_iterator find(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.find(key);
        }

        bool contains(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.find(key) != set_.end();
        }

        // Bucket interface
        size_type bucket_count() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.bucket_count();
        }

        size_type max_bucket_count() const noexcept
        {
            return set_.max_bucket_count();
        }

        size_type bucket_size(size_type n) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.bucket_size(n);
        }

        size_type bucket(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.bucket(key);
        }

        // Hash policy
        float load_factor() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.load_factor();
        }

        float max_load_factor() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return set_.max_load_factor();
        }

        void max_load_factor(float ml)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            set_.max_load_factor(ml);
        }

        void rehash(size_type count)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            set_.rehash(count);
        }

        void reserve(size_type count)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            set_.reserve(count);
        }
    };

}    // namespace hpx::concurrent
