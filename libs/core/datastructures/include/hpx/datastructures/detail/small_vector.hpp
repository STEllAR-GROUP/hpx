//  Copyright (c) 2021 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
#include <memory>

#if !defined(HPX_HAVE_CXX17_MEMORY_RESOURCE)

// fall back to Boost if memory_resource is not supported
#include <boost/container/small_vector.hpp>

namespace hpx::detail {

    template <typename T, std::size_t Size,
        typename Allocator = std::allocator<T>>
    using small_vector = boost::container::small_vector<T, Size, Allocator>;
}

#else

#include <initializer_list>
#include <memory_resource>
#include <utility>
#include <vector>

namespace hpx::detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Allocator>
    class allocator_memory_resource final : public std::pmr::memory_resource
    {
    public:
        explicit allocator_memory_resource(Allocator const& alloc) noexcept
          : allocator_(alloc)
        {
        }

    private:
        bool do_is_equal(memory_resource const& rhs) const noexcept override
        {
            return this == &rhs;
        }

        // Note: the PMR infrastructure calls this with the number of bytes to
        // allocate but the underlying allocator expects the number of elements
        void* do_allocate(std::size_t count, std::size_t) override
        {
            return allocator_.allocate((count + sizeof(T) - 1) / sizeof(T));
        }

        void do_deallocate(
            void* ptr, std::size_t count, std::size_t) noexcept override
        {
            using value_type =
                typename std::allocator_traits<Allocator>::value_type;

            return allocator_.deallocate(static_cast<value_type*>(ptr),
                (count + sizeof(T) - 1) / sizeof(T));
        }

    public:
        HPX_NO_UNIQUE_ADDRESS Allocator allocator_;
    };

    template <typename T, std::size_t Size, typename Allocator>
    struct memory_storage final
    {
        static_assert(Size > 0, "memory_storage::Size must be non-zero");

        static constexpr std::size_t preallocated_size = Size * sizeof(T);

        using buffer_resource_type = std::pmr::monotonic_buffer_resource;
        using allocator_type = std::pmr::polymorphic_allocator<T>;

        explicit memory_storage(Allocator const& alloc = Allocator()) noexcept(
            noexcept(Allocator()))
          : resource_(alloc)
          , pool_(std::data(memory_), preallocated_size, &resource_)
          , allocator_(&pool_)
        {
        }

        memory_storage(memory_storage const& rhs)
          : resource_(rhs.resource_)
          , pool_(std::data(memory_), preallocated_size, &resource_)
          , allocator_(&pool_)
        {
        }
        memory_storage(memory_storage&& rhs) noexcept
          : resource_(HPX_MOVE(rhs.resource_))
          , pool_(std::data(memory_), preallocated_size, &resource_)
          , allocator_(&pool_)
        {
        }

        // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
        memory_storage& operator=(memory_storage const& rhs)
        {
            // release all memory owned by the old instance
            allocator_.allocator_type::~allocator_type();
            pool_.buffer_resource_type::~buffer_resource_type();

            // no need to invoke: memory_ = rhs.memory_; as the data will
            // be provided by the small_vector::operator= below

            // copy allocator
            resource_ = rhs.resource_;

            // reconstruct the memory management infrastructure
            new (&pool_) buffer_resource_type(
                std::data(memory_), preallocated_size, &resource_);
            new (&allocator_) allocator_type(&pool_);

            return *this;
        }

        // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
        memory_storage& operator=(memory_storage&& rhs) noexcept
        {
            // release all memory owned by the old instance
            allocator_.allocator_type::~allocator_type();
            pool_.buffer_resource_type::~buffer_resource_type();

            // no need to invoke: memory_ = rhs.memory_; as the data will
            // be provided by the small_vector::operator= below

            // move allocator
            resource_ = HPX_MOVE(rhs.resource_);

            // reconstruct the memory management infrastructure
            new (&pool_) buffer_resource_type(
                std::data(memory_), preallocated_size, &resource_);
            new (&allocator_) allocator_type(&pool_);

            return *this;
        }

        std::aligned_storage_t<sizeof(T), alignof(T)> memory_[Size];
        HPX_NO_UNIQUE_ADDRESS allocator_memory_resource<T, Allocator> resource_;
        buffer_resource_type pool_;
        allocator_type allocator_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T, std::size_t Size,
        typename Allocator = std::allocator<T>>
    class small_vector
    {
    private:
        using other_allocator =
            typename std::allocator_traits<Allocator>::template rebind_alloc<T>;

        using storage_type = memory_storage<T, Size, other_allocator>;
        using data_type = std::pmr::vector<T>;

    public:
        using value_type = typename data_type::value_type;
        using allocator_type = other_allocator;
        using size_type = typename data_type::size_type;
        using difference_type = typename data_type::difference_type;
        using reference = typename data_type::reference;
        using const_reference = typename data_type::const_reference;
        using pointer = typename data_type::pointer;
        using const_pointer = typename data_type::const_pointer;
        using iterator = typename data_type::iterator;
        using const_iterator = typename data_type::const_iterator;
        using reverse_iterator = typename data_type::reverse_iterator;
        using const_reverse_iterator =
            typename data_type::const_reverse_iterator;

        static constexpr std::size_t static_capacity = Size;

        small_vector() noexcept(noexcept(allocator_type()))
          : storage_()
          , data_(storage_.allocator_)
        {
            data_.reserve(Size);
        }

        explicit small_vector(allocator_type const& alloc) noexcept
          : storage_(alloc)
          , data_(storage_.allocator_)
        {
            data_.reserve(Size);
        }

        explicit small_vector(
            size_type count, allocator_type const& alloc = allocator_type())
          : storage_(alloc)
          , data_(count, storage_.allocator_)
        {
        }

        small_vector(size_type count, value_type const& value,
            allocator_type const& alloc = allocator_type())
          : storage_(alloc)
          , data_(count, value, storage_.allocator_)
        {
        }

        small_vector(std::initializer_list<T> init_list,
            allocator_type const& alloc = allocator_type())
          : storage_(alloc)
          , data_(std::cbegin(init_list), std::cend(init_list),
                storage_.allocator_)
        {
        }

        template <typename Iterator>
        small_vector(Iterator first, Iterator last,
            allocator_type const& alloc = allocator_type())
          : storage_(alloc)
          , data_(first, last, storage_.allocator_)
        {
        }

        small_vector(small_vector const& rhs)
          : storage_(rhs.storage_)
          , data_(rhs.data_, storage_.allocator_)
        {
        }

        small_vector(small_vector&& rhs) noexcept
          : storage_(rhs.storage_)
          , data_(HPX_MOVE(rhs.data_), storage_.allocator_)
        {
        }

        small_vector& operator=(small_vector const& rhs)
        {
            if (this != &rhs)
            {
                // free all data owned by the old instance
                data_.data_type::~data_type();

                // reconstruct the memory management infrastructure for
                // the new instance
                storage_ = rhs.storage_;

                // fill the new instance with a copy of the rhs data
                new (&data_) data_type(rhs.data_, storage_.allocator_);
            }
            return *this;
        }
        small_vector& operator=(small_vector&& rhs) noexcept
        {
            if (this != &rhs)
            {
                // free all data owned by the old instance
                data_.data_type::~data_type();

                // reconstruct the memory management infrastructure for
                // the new instance
                storage_ = HPX_MOVE(rhs.storage_);

                // fill the new instance with the moved rhs data
                new (&data_)
                    data_type(HPX_MOVE(rhs.data_), storage_.allocator_);
            }
            return *this;
        }
        small_vector& operator=(std::initializer_list<T> init_list)
        {
            data_ = init_list;
            return *this;
        }

        allocator_type get_allocator() const noexcept
        {
            return storage_.resource_.allocator_;
        }

        void assign(size_type count, value_type const& value)
        {
            data_.assign(count, value);
        }
        template <typename Iterator>
        void assign(Iterator first, Iterator last)
        {
            data_.assign(first, last);
        }
        void assign(std::initializer_list<T> init_list)
        {
            data_.assign(init_list);
        }

        reference operator[](size_type pos)
        {
            return data_[pos];
        }
        const_reference operator[](size_type pos) const
        {
            return data_[pos];
        }

        reference at(size_type pos)
        {
            return data_.at(pos);
        }
        const_reference at(size_type pos) const
        {
            return data_.at(pos);
        }

        reference front()
        {
            return data_.front();
        }
        const_reference front() const
        {
            return data_.front();
        }

        reference back()
        {
            return data_.back();
        }
        const_reference back() const
        {
            return data_.back();
        }

        T* data() noexcept
        {
            return data_.data();
        }
        T const* data() const noexcept
        {
            return data_.data();
        }

        iterator begin()
        {
            return data_.begin();
        }
        const_iterator begin() const
        {
            return data_.begin();
        }

        const_iterator cbegin() const
        {
            return data_.cbegin();
        }

        reverse_iterator rbegin()
        {
            return data_.rbegin();
        }
        const_reverse_iterator rbegin() const
        {
            return data_.rbegin();
        }

        const_reverse_iterator crbegin() const
        {
            return data_.crbegin();
        }

        iterator end()
        {
            return data_.end();
        }
        const_iterator end() const
        {
            return data_.end();
        }

        const_iterator cend() const
        {
            return data_.cend();
        }

        reverse_iterator rend()
        {
            return data_.rend();
        }
        const_reverse_iterator rend() const
        {
            return data_.rend();
        }

        const_reverse_iterator crend() const
        {
            return data_.crend();
        }

        bool empty() const
        {
            return data_.empty();
        }

        size_type size() const
        {
            return data_.size();
        }

        size_type max_size() const
        {
            return data_.max_size();
        }

        void reserve(std::size_t count)
        {
            data_.reserve(count);
        }

        size_type capacity() const
        {
            return data_.capacity();
        }

        void shrink_to_fit()
        {
            data_.shrink_to_fit();
        }

        void clear() noexcept
        {
            data_.clear();
        }

        iterator insert(const_iterator pos, value_type const& value)
        {
            return data_.insert(pos, value);
        }
        iterator insert(const_iterator pos, value_type&& value)
        {
            return data_.insert(pos, HPX_MOVE(value));
        }
        iterator insert(
            const_iterator pos, size_type count, value_type const& value)
        {
            return data_.insert(pos, count, value);
        }
        template <typename Iterator>
        iterator insert(Iterator first, Iterator last)
        {
            return data_.insert(first, last);
        }
        template <typename Iterator>
        iterator insert(const_iterator pos, Iterator first, Iterator last)
        {
            return data_.insert(pos, first, last);
        }
        iterator insert(const_iterator pos, std::initializer_list<T> init_list)
        {
            return data_.insert(pos, init_list);
        }

        template <typename... Ts>
        iterator emplace(const_iterator pos, Ts&&... ts)
        {
            return data_.emplace(pos, HPX_FORWARD(Ts, ts)...);
        }

        iterator erase(const_iterator pos)
        {
            return data_.erase(pos);
        }

        void push_back(value_type const& value)
        {
            data_.push_back(value);
        }
        void push_back(value_type&& value)
        {
            data_.push_back(HPX_MOVE(value));
        }

        template <typename... Ts>
        reference emplace_back(Ts&&... ts)
        {
            return data_.emplace_back(HPX_FORWARD(Ts, ts)...);
        }

        void pop_back()
        {
            data_.pop_back();
        }

        void resize(size_type count)
        {
            data_.resize(count);
        }
        void resize(size_type count, value_type const& value)
        {
            data_.resize(count, value);
        }

        void swap(small_vector& other) noexcept
        {
            // data.swap(other.data_);
            // explicitly move the objects as the MSVC standard library has a
            // bug preventing to swap two std::pmr::vectors
            small_vector tmp = HPX_MOVE(*this);
            *this = HPX_MOVE(other);
            other = HPX_MOVE(tmp);
        }

        friend bool operator==(small_vector const& lhs, small_vector const& rhs)
        {
            return lhs.data_ == rhs.data_;
        }
        friend bool operator<(small_vector const& lhs, small_vector const& rhs)
        {
            return lhs.data_ < rhs.data_;
        }
        friend bool operator>(small_vector const& lhs, small_vector const& rhs)
        {
            return lhs.data_ > rhs.data_;
        }
        friend bool operator<=(small_vector const& lhs, small_vector const& rhs)
        {
            return lhs.data_ <= rhs.data_;
        }
        friend bool operator>=(small_vector const& lhs, small_vector const& rhs)
        {
            return lhs.data_ >= rhs.data_;
        }
        friend bool operator!=(small_vector const& lhs, small_vector const& rhs)
        {
            return lhs.data_ != rhs.data_;
        }

    private:
        storage_type storage_;
        data_type data_;
    };
}    // namespace hpx::detail

namespace std {

    template <typename T, std::size_t Size, typename Allocator>
    void swap(hpx::detail::small_vector<T, Size, Allocator>& lhs,
        hpx::detail::small_vector<T, Size, Allocator>& rhs)
    {
        lhs.swap(rhs);
    }
}    // namespace std

#endif
