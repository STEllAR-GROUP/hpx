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
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace hpx::concurrent {

    template <typename T, typename Allocator = std::allocator<T>>
    class concurrent_vector
    {
    private:
        mutable hpx::util::spinlock mutex_;
        std::deque<T, Allocator> data_;

    public:
        using value_type = T;
        using allocator_type = Allocator;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;

        // Accessor classes to ensure thread-safe element access
        class accessor;
        class const_accessor;

        using reference = accessor;
        using const_reference = const_accessor;

        class accessor
        {
            friend class concurrent_vector;
            std::unique_lock<hpx::util::spinlock> lock_;
            T* value_;

            accessor(std::unique_lock<hpx::util::spinlock>&& l, T* v)
              : lock_(std::move(l))
              , value_(v)
            {
            }

        public:
            accessor() = default;

            operator T&() const
            {
                if (!value_)
                    throw std::runtime_error("Empty accessor dereference");
                return *value_;
            }
            T& get() const
            {
                if (!value_)
                    throw std::runtime_error("Empty accessor dereference");
                return *value_;
            }
            accessor& operator=(T const& v)
            {
                if (!value_)
                    throw std::runtime_error("Empty accessor dereference");
                *value_ = v;
                return *this;
            }
        };

        class const_accessor
        {
            friend class concurrent_vector;
            std::unique_lock<hpx::util::spinlock> lock_;
            T const* value_;

            const_accessor(
                std::unique_lock<hpx::util::spinlock>&& l, T const* v)
              : lock_(std::move(l))
              , value_(v)
            {
            }

        public:
            const_accessor() = default;

            operator T const&() const
            {
                if (!value_)
                    throw std::runtime_error("Empty accessor dereference");
                return *value_;
            }
            T const& get() const
            {
                if (!value_)
                    throw std::runtime_error("Empty accessor dereference");
                return *value_;
            }
        };

        // Custom Index-based Iterator (Stable against growth)
        template <bool IsConst>
        class iterator_impl
        {
            friend class concurrent_vector;
            using VectorT = std::conditional_t<IsConst, concurrent_vector const,
                concurrent_vector>;
            using AccessorT =
                std::conditional_t<IsConst, const_accessor, accessor>;

            VectorT* vec_;
            size_type index_;

        public:
            using iterator_category = std::random_access_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = T;
            using pointer = T*;
            using reference = AccessorT;

            iterator_impl()
              : vec_(nullptr)
              , index_(0)
            {
            }
            iterator_impl(VectorT* vec, size_type index)
              : vec_(vec)
              , index_(index)
            {
            }

            reference operator*() const
            {
                // Accessing element locks the container
                return (*vec_)[index_];
            }

            iterator_impl& operator++()
            {
                ++index_;
                return *this;
            }
            iterator_impl operator++(int)
            {
                iterator_impl tmp = *this;
                ++index_;
                return tmp;
            }
            iterator_impl& operator--()
            {
                --index_;
                return *this;
            }
            iterator_impl operator--(int)
            {
                iterator_impl tmp = *this;
                --index_;
                return tmp;
            }
            iterator_impl& operator+=(difference_type n)
            {
                index_ += n;
                return *this;
            }
            iterator_impl& operator-=(difference_type n)
            {
                index_ -= n;
                return *this;
            }
            iterator_impl operator+(difference_type n) const
            {
                return iterator_impl(vec_, index_ + n);
            }
            iterator_impl operator-(difference_type n) const
            {
                return iterator_impl(vec_, index_ - n);
            }
            difference_type operator-(iterator_impl const& other) const
            {
                return index_ - other.index_;
            }
            bool operator==(iterator_impl const& other) const
            {
                return vec_ == other.vec_ && index_ == other.index_;
            }
            bool operator!=(iterator_impl const& other) const
            {
                return !(*this == other);
            }
            bool operator<(iterator_impl const& other) const
            {
                return index_ < other.index_;
            }
        };

        using iterator = iterator_impl<false>;
        using const_iterator = iterator_impl<true>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

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
        {
            std::lock_guard<hpx::util::spinlock> lock(other.mutex_);
            data_ = other.data_;
        }

        concurrent_vector(concurrent_vector&& other) noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(other.mutex_);
            data_ = std::move(other.data_);
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
                std::lock_guard<hpx::util::spinlock> other_lock(other.mutex_);
                data_ = std::move(other.data_);
            }
            return *this;
        }

        // Element access
        accessor operator[](size_type pos)
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return accessor(std::move(lock), &data_[pos]);
        }

        const_accessor operator[](size_type pos) const
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return const_accessor(std::move(lock), &data_[pos]);
        }

        accessor at(size_type pos)
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return accessor(std::move(lock), &data_.at(pos));
        }

        const_accessor at(size_type pos) const
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return const_accessor(std::move(lock), &data_.at(pos));
        }

        accessor front()
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return accessor(std::move(lock), &data_.front());
        }

        const_accessor front() const
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return const_accessor(std::move(lock), &data_.front());
        }

        accessor back()
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return accessor(std::move(lock), &data_.back());
        }

        const_accessor back() const
        {
            std::unique_lock<hpx::util::spinlock> lock(mutex_);
            return const_accessor(std::move(lock), &data_.back());
        }

        // Iterators
        iterator begin() noexcept
        {
            return iterator(this, 0);
        }
        const_iterator begin() const noexcept
        {
            return const_iterator(this, 0);
        }
        const_iterator cbegin() const noexcept
        {
            return const_iterator(this, 0);
        }
        iterator end() noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return iterator(this, data_.size());
        }
        const_iterator end() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return const_iterator(this, data_.size());
        }
        const_iterator cend() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            return const_iterator(this, data_.size());
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

        void reserve(size_type /*new_cap*/) {}

        size_type capacity() const noexcept
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
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
            return iterator(this, old_size);
        }

        iterator grow_by(size_type n, T const& value)
        {
            std::lock_guard<hpx::util::spinlock> lock(mutex_);
            auto old_size = data_.size();
            data_.resize(old_size + n, value);
            return iterator(this, old_size);
        }
    };

}    // namespace hpx::concurrent
