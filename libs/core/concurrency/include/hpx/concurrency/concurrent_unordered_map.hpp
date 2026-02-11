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
#include <unordered_map>
#include <utility>

namespace hpx::concurrent {

    template <typename Key, typename T, typename Hash = std::hash<Key>,
        typename KeyEqual = std::equal_to<Key>,
        typename Allocator = std::allocator<std::pair<Key const, T>>>
    class concurrent_unordered_map
    {
    public:
        using key_type = Key;
        using mapped_type = T;
        using value_type = std::pair<Key const, T>;
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

        // Note: Iterators are not thread-safe if modifications happen concurrently.
        using iterator = typename std::unordered_map<Key, T, Hash, KeyEqual,
            Allocator>::iterator;
        using const_iterator = typename std::unordered_map<Key, T, Hash,
            KeyEqual, Allocator>::const_iterator;
        using local_iterator = typename std::unordered_map<Key, T, Hash,
            KeyEqual, Allocator>::local_iterator;
        using const_local_iterator = typename std::unordered_map<Key, T, Hash,
            KeyEqual, Allocator>::const_local_iterator;

    private:
        mutable hpx::util::spinlock mutex_;
        std::unordered_map<Key, T, Hash, KeyEqual, Allocator> map_;

    public:
        concurrent_unordered_map() = default;

        explicit concurrent_unordered_map(size_type bucket_count,
            Hash const& hash = Hash(), KeyEqual const& equal = KeyEqual(),
            Allocator const& alloc = Allocator())
          : map_(bucket_count, hash, equal, alloc)
        {
        }

        explicit concurrent_unordered_map(Allocator const& alloc)
          : map_(alloc)
        {
        }

        concurrent_unordered_map(concurrent_unordered_map const& other)
          : map_(other.map_)
        {
        }

        concurrent_unordered_map(concurrent_unordered_map&& other) noexcept
          : map_(std::move(other.map_))
        {
        }

        concurrent_unordered_map& operator=(
            concurrent_unordered_map const& other)
        {
            if (this != &other)
            {
                std::lock(mutex_, other.mutex_);
                std::lock_guard<hpx::util::spinlock> lock(
                    mutex_, std::adopt_lock);
                std::lock_guard<hpx::util::spinlock> other_lock(
                    other.mutex_, std::adopt_lock);
                map_ = other.map_;
            }
            return *this;
        }

        concurrent_unordered_map& operator=(
            concurrent_unordered_map&& other) noexcept
        {
            if (this != &other)
            {
                std::lock_guard<hpx::util::spinlock> lock(mutex_);
                map_ = std::move(other.map_);
            }
            return *this;
        }

        // Capacity
        bool empty() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.empty();
        }

        size_type size() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.size();
        }

        size_type max_size() const noexcept
        {
            return map_.max_size();
        }

        // Iterators
        iterator begin() noexcept
        {
            return map_.begin();
        }
        const_iterator begin() const noexcept
        {
            return map_.begin();
        }
        const_iterator cbegin() const noexcept
        {
            return map_.cbegin();
        }
        iterator end() noexcept
        {
            return map_.end();
        }
        const_iterator end() const noexcept
        {
            return map_.end();
        }
        const_iterator cend() const noexcept
        {
            return map_.cend();
        }

        // Modifiers
        void clear() noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            map_.clear();
        }

        std::pair<iterator, bool> insert(value_type const& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.insert(value);
        }

        std::pair<iterator, bool> insert(value_type&& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.insert(std::move(value));
        }

        template <typename P>
        std::pair<iterator, bool> insert(P&& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.insert(std::forward<P>(value));
        }

        iterator erase(const_iterator pos)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.erase(pos);
        }

        iterator erase(iterator pos)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.erase(pos);
        }

        size_type erase(Key const& key)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.erase(key);
        }

        void swap(concurrent_unordered_map& other) noexcept
        {
            if (this != &other)
            {
                std::lock(mutex_, other.mutex_);
                std::lock_guard<hpx::util::spinlock> lock(
                    mutex_, std::adopt_lock);
                std::lock_guard<hpx::util::spinlock> other_lock(
                    other.mutex_, std::adopt_lock);
                map_.swap(other.map_);
            }
        }

        // Lookup
        T& at(Key const& key)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.at(key);
        }

        T const& at(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.at(key);
        }

        T& operator[](Key const& key)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_[key];
        }

        T& operator[](Key&& key)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_[std::move(key)];
        }

        size_type count(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.count(key);
        }

        iterator find(Key const& key)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.find(key);
        }

        const_iterator find(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.find(key);
        }

        bool contains(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.find(key) != map_.end();
        }

        // Bucket interface
        size_type bucket_count() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.bucket_count();
        }

        size_type max_bucket_count() const noexcept
        {
            return map_.max_bucket_count();
        }

        size_type bucket_size(size_type n) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.bucket_size(n);
        }

        size_type bucket(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.bucket(key);
        }

        // Hash policy
        float load_factor() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.load_factor();
        }

        float max_load_factor() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.max_load_factor();
        }

        void max_load_factor(float ml)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            map_.max_load_factor(ml);
        }

        void rehash(size_type count)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            map_.rehash(count);
        }

        void reserve(size_type count)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            map_.reserve(count);
        }
    };

}    // namespace hpx::concurrent
