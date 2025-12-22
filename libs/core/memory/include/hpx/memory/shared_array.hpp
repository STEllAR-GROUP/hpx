//  Copyright (c) 2025 Hartmut Kaiser
//  Copyright (c) 2025 Mamidi Surya Teja
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/memory/config/defines.hpp>

#include <atomic>
#include <cstddef>
#include <utility>

namespace hpx::memory {
    namespace detail {
        template <typename T>
        struct shared_array_control_block_base
        {
            std::atomic<long> count;

            explicit shared_array_control_block_base(long initial_count = 1)
              : count{initial_count}
            {
            }

            virtual ~shared_array_control_block_base() = default;

            virtual void destroy(T* ptr) noexcept = 0;

            void increment() noexcept
            {
                count.fetch_add(1, std::memory_order_relaxed);
            }

            long decrement() noexcept
            {
                return count.fetch_sub(1, std::memory_order_acq_rel) - 1;
            }

            long get_count() const noexcept
            {
                return count.load(std::memory_order_relaxed);
            }
        };

        template <typename T>
        struct shared_array_control_block_default
          : shared_array_control_block_base<T>
        {
            shared_array_control_block_default()
              : shared_array_control_block_base<T>(1)
            {
            }

            void destroy(T* ptr) noexcept override
            {
                delete[] ptr;
            }
        };

        template <typename T, typename Deleter>
        struct shared_array_control_block_deleter
          : shared_array_control_block_base<T>
        {
            Deleter deleter;

            explicit shared_array_control_block_deleter(Deleter d)
              : shared_array_control_block_base<T>(1)
              , deleter(HPX_MOVE(d))
            {
            }

            void destroy(T* ptr) noexcept override
            {
                deleter(ptr);
            }
        };
    }    // namespace detail

    HPX_CXX_EXPORT template <typename T>
    class shared_array
    {
    public:
        using element_type = T;
        using this_type = shared_array;

    private:
        using control_block_base = detail::shared_array_control_block_base<T>;

    public:
        constexpr shared_array() noexcept
          : ptr_{nullptr}
          , control_{nullptr}
        {
        }

        constexpr shared_array(std::nullptr_t) noexcept
          : ptr_{nullptr}
          , control_{nullptr}
        {
        }

        explicit shared_array(element_type* ptr)
          : ptr_{ptr}
          , control_{ptr ? new detail::shared_array_control_block_default<T>() :
                           nullptr}
        {
        }

        template <typename Deleter>
        shared_array(element_type* ptr, Deleter deleter)
          : ptr_{ptr}
          , control_{ptr ?
                    new detail::shared_array_control_block_deleter<T, Deleter>(
                        HPX_MOVE(deleter)) :
                    nullptr}
        {
        }

        shared_array(shared_array const& other) noexcept
          : ptr_{other.ptr_}
          , control_{other.control_}
        {
            if (control_ != nullptr)
            {
                control_->increment();
            }
        }

        shared_array(shared_array&& other) noexcept
          : ptr_{other.ptr_}
          , control_{other.control_}
        {
            other.ptr_ = nullptr;
            other.control_ = nullptr;
        }

        ~shared_array()
        {
            release();
        }

        shared_array& operator=(shared_array const& other) noexcept
        {
            if (this != &other)
            {
                release();
                ptr_ = other.ptr_;
                control_ = other.control_;
                if (control_ != nullptr)
                {
                    control_->increment();
                }
            }
            return *this;
        }

        shared_array& operator=(shared_array&& other) noexcept
        {
            if (this != &other)
            {
                release();
                ptr_ = other.ptr_;
                control_ = other.control_;
                other.ptr_ = nullptr;
                other.control_ = nullptr;
            }
            return *this;
        }

        void reset() noexcept
        {
            release();
            ptr_ = nullptr;
            control_ = nullptr;
        }

        void reset(element_type* ptr)
        {
            release();
            ptr_ = ptr;
            control_ = ptr ?
                new detail::shared_array_control_block_default<T>() :
                nullptr;
        }

        template <typename Deleter>
        void reset(element_type* ptr, Deleter deleter)
        {
            release();
            ptr_ = ptr;
            control_ = ptr ?
                new detail::shared_array_control_block_deleter<T, Deleter>(
                    HPX_MOVE(deleter)) :
                nullptr;
        }

        element_type& operator[](std::ptrdiff_t idx) const noexcept
        {
            return ptr_[idx];
        }

        element_type* get() const noexcept
        {
            return ptr_;
        }

        explicit operator bool() const noexcept
        {
            return ptr_ != nullptr;
        }

        long use_count() const noexcept
        {
            if (control_ == nullptr)
                return 0;
            return control_->get_count();
        }

        bool unique() const noexcept
        {
            return use_count() == 1;
        }

        void swap(shared_array& other) noexcept
        {
            std::swap(ptr_, other.ptr_);
            std::swap(control_, other.control_);
        }

        friend bool operator==(
            shared_array const& lhs, shared_array const& rhs) noexcept
        {
            return lhs.ptr_ == rhs.ptr_;
        }

        friend bool operator!=(
            shared_array const& lhs, shared_array const& rhs) noexcept
        {
            return lhs.ptr_ != rhs.ptr_;
        }

        friend bool operator<(
            shared_array const& lhs, shared_array const& rhs) noexcept
        {
            return lhs.ptr_ < rhs.ptr_;
        }

        friend bool operator==(shared_array const& lhs, std::nullptr_t) noexcept
        {
            return lhs.ptr_ == nullptr;
        }

        friend bool operator==(std::nullptr_t, shared_array const& rhs) noexcept
        {
            return rhs.ptr_ == nullptr;
        }

        friend bool operator!=(shared_array const& lhs, std::nullptr_t) noexcept
        {
            return lhs.ptr_ != nullptr;
        }

        friend bool operator!=(std::nullptr_t, shared_array const& rhs) noexcept
        {
            return rhs.ptr_ != nullptr;
        }

    private:
        void release() noexcept
        {
            if (control_ != nullptr)
            {
                if (control_->decrement() == 0)
                {
                    control_->destroy(ptr_);
                    delete control_;
                }
            }
            ptr_ = nullptr;
            control_ = nullptr;
        }

        element_type* ptr_ = nullptr;
        control_block_base* control_ = nullptr;
    };

    template <typename T>
    inline void swap(shared_array<T>& lhs, shared_array<T>& rhs) noexcept
    {
        lhs.swap(rhs);
    }

}    // namespace hpx::memory
