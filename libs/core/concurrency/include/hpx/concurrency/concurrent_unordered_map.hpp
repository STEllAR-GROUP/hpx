//  Copyright (c) 2026 Pratyksh Gupta
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/concurrency/spinlock.hpp>
#include <hpx/modules/errors.hpp>
#include <hpx/modules/type_support.hpp>

#include <functional>
#include <mutex>

#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace hpx::concurrent {

    HPX_CXX_CORE_EXPORT template <typename Key, typename T,
        typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>,
        typename Allocator = std::allocator<std::pair<Key const, T>>>
    class concurrent_unordered_map
    {
    private:
        mutable hpx::util::spinlock mutex_;
        std::unordered_map<Key, T, Hash, KeyEqual, Allocator> map_;

    public:
        using key_type = Key;
        using mapped_type = T;
        using value_type = std::pair<Key const, T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using hasher = Hash;
        using key_equal = KeyEqual;
        using allocator_type = Allocator;

        class accessor;
        class const_accessor;

        using reference = accessor;
        using const_reference = const_accessor;

        // Accessors
        class accessor
        {
            friend class concurrent_unordered_map;
            std::unique_lock<hpx::util::spinlock> lock_;
            T* value_ = nullptr;

            accessor(std::unique_lock<hpx::util::spinlock>&& l, T& v)
              : lock_(HPX_MOVE(l))
              , value_(&v)
            {
                HPX_ASSERT_OWNS_LOCK(lock_);
            }

            void validate() const
            {
                if (!value_)
                {
                    HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                        "concurrent_unordered_map::accessor",
                        "Empty accessor dereference");
                }
            }

        public:
            accessor() = default;

            bool empty() const
            {
                return value_ == nullptr;
            }

            explicit operator bool() const noexcept
                requires(!std::same_as<T, bool>)
            {
                return !empty();
            }

            operator T&() const
            {
                return get();
            }

            T& get() const
            {
                validate();
                return *value_;
            }

            accessor& operator=(T const& v)
            {
                validate();
                *value_ = v;
                return *this;
            }
        };

        class const_accessor
        {
            friend class concurrent_unordered_map;
            std::unique_lock<hpx::util::spinlock> lock_;
            T const* value_ = nullptr;

            const_accessor(
                std::unique_lock<hpx::util::spinlock>&& l, T const& v)
              : lock_(HPX_MOVE(l))
              , value_(&v)
            {
                HPX_ASSERT_OWNS_LOCK(lock_);
            }

            void validate() const
            {
                if (!value_)
                {
                    HPX_THROW_EXCEPTION(hpx::error::invalid_status,
                        "concurrent_unordered_map::const_accessor",
                        "Empty accessor dereference");
                }
            }

        public:
            const_accessor() = default;

            bool empty() const
            {
                return value_ == nullptr;
            }

            explicit operator bool() const noexcept
                requires(!std::same_as<T, bool>)
            {
                return !empty();
            }

            operator T const&() const
            {
                return get();
            }

            T const& get() const
            {
                validate();
                return *value_;
            }
        };

        concurrent_unordered_map() = default;

        explicit concurrent_unordered_map(size_type bucket_count,
            Hash const& hash = Hash(), KeyEqual const& equal = KeyEqual(),
            Allocator const& alloc = Allocator())
          : map_(bucket_count, hash, equal, alloc)
        {
        }

        concurrent_unordered_map(size_type bucket_count, Allocator const& alloc)
          : map_(bucket_count, alloc)
        {
        }

        concurrent_unordered_map(
            size_type bucket_count, Hash const& hash, Allocator const& alloc)
          : map_(bucket_count, hash, alloc)
        {
        }

        explicit concurrent_unordered_map(Allocator const& alloc)
          : map_(alloc)
        {
        }

        concurrent_unordered_map(concurrent_unordered_map const& other)
        {
            std::lock_guard<hpx::util::spinlock> lock(other.mutex_);
            map_ = other.map_;
        }

        concurrent_unordered_map(concurrent_unordered_map&& other) noexcept
          : map_(HPX_MOVE(other.map_))
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
                map_ = HPX_MOVE(other.map_);
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
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.max_size();
        }

        // Modifiers
        void clear() noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            map_.clear();
        }

        bool insert(value_type const& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.insert(value).second;
        }

        bool insert(value_type&& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.insert(HPX_MOVE(value)).second;
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
        accessor operator[](Key const& key)
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return accessor(HPX_MOVE(lock), map_[key]);
        }

        accessor operator[](Key&& key)
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return accessor(HPX_MOVE(lock), map_[HPX_MOVE(key)]);
        }

        accessor at(Key const& key)
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return accessor(HPX_MOVE(lock), map_.at(key));
        }

        const_accessor at(Key const& key) const
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return const_accessor(HPX_MOVE(lock), map_.at(key));
        }

        size_type count(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.count(key);
        }

        accessor find(Key const& key)
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            auto it = map_.find(key);
            if (it != map_.end())
            {
                return accessor(HPX_MOVE(lock), it->second);
            }
            return accessor();
        }

        const_accessor find(Key const& key) const
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            auto it = map_.find(key);
            if (it != map_.end())
            {
                return const_accessor(HPX_MOVE(lock), it->second);
            }
            return const_accessor();
        }

        bool contains(Key const& key) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.contains(key);
        }

        // Thread-safe iteration
        template <typename F>
        void for_each(F&& f)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            for (auto& kv : map_)
            {
                if constexpr (std::is_void_v<
                                  std::invoke_result_t<F, decltype(kv)>>)
                {
                    f(kv);
                }
                else
                {
                    if (!f(kv))
                        break;
                }
            }
        }

        template <typename F>
        void for_each(F&& f) const
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            for (auto const& kv : map_)
            {
                if constexpr (std::is_void_v<
                                  std::invoke_result_t<F, decltype(kv)>>)
                {
                    f(kv);
                }
                else
                {
                    if (!f(kv))
                        break;
                }
            }
        }

        // Bucket interface
        size_type bucket_count() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return map_.bucket_count();
        }

        size_type max_bucket_count() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
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
