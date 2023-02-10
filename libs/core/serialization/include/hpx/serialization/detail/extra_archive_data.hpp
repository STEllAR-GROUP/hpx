//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2019-2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::serialization::detail {

    using extra_archive_data_id_type = void*;

    template <typename T>
    struct extra_archive_data_helper
    {
        // this is intentionally left unimplemented, will lead to linker errors
        // if used with unknown data type
        static extra_archive_data_id_type id() noexcept = delete;

        // this is a function that should be implemented in order to reset the
        // extra archive data item
        static void reset(T* data) noexcept = delete;
    };

    template <typename T>
    extra_archive_data_id_type extra_archive_data_id() noexcept
    {
        return extra_archive_data_helper<T>::id();
    }

    template <typename T>
    constexpr void reset_extra_archive_data(T* data) noexcept(
        noexcept(extra_archive_data_helper<T>::reset(data)))
    {
        extra_archive_data_helper<T>::reset(data);
    }

    struct extra_archive_data_member_base;
    template <typename T>
    struct extra_archive_data_member;

    struct extra_archive_data_node
    {
        constexpr extra_archive_data_node() noexcept = default;

        template <typename T>
        extra_archive_data_node(T* t, extra_archive_data_node&& next);

        extra_archive_data_node(extra_archive_data_node const&) = delete;
        extra_archive_data_node(extra_archive_data_node&&) = default;
        extra_archive_data_node& operator=(
            extra_archive_data_node const&) = delete;
        extra_archive_data_node& operator=(extra_archive_data_node&&) = default;

        template <typename T>
        [[nodiscard]] T* get() const noexcept;

        std::unique_ptr<extra_archive_data_member_base> ptr_;
        extra_archive_data_id_type id_ = nullptr;
    };

    struct extra_archive_data_member_base
    {
        explicit extra_archive_data_member_base(
            extra_archive_data_node&& next) noexcept
          : next_(HPX_MOVE(next))
        {
        }

        virtual ~extra_archive_data_member_base() = default;
        virtual void reset() = 0;

        extra_archive_data_node next_;
    };

    template <typename T>
    struct extra_archive_data_member : extra_archive_data_member_base
    {
        explicit constexpr extra_archive_data_member(
            extra_archive_data_node&& next) noexcept
          : extra_archive_data_member_base(HPX_MOVE(next))
        {
        }

        extra_archive_data_member(extra_archive_data_member const&) = delete;
        extra_archive_data_member& operator=(
            extra_archive_data_member const&) = delete;

        [[nodiscard]] constexpr T* value() const noexcept
        {
            return std::addressof(const_cast<T&>(t_));
        }

        void reset() override
        {
            reset_extra_archive_data(value());
        }

        T t_;
    };

    template <typename T>
    extra_archive_data_node::extra_archive_data_node(
        T*, extra_archive_data_node&& next)
      : ptr_(new extra_archive_data_member<T>(HPX_MOVE(next)))
      , id_(extra_archive_data_id<T>())
    {
    }

    template <typename T>
    T* extra_archive_data_node::get() const noexcept
    {
        auto id = extra_archive_data_id<T>();
        if (id_ == nullptr)
        {
            HPX_ASSERT(!ptr_);
            return nullptr;
        }

        HPX_ASSERT(ptr_);
        if (id_ == id)
        {
            return static_cast<extra_archive_data_member<T>*>(ptr_.get())
                ->value();
        }

        return ptr_->next_.get<T>();
    }

    struct extra_archive_data
    {
        extra_archive_data() noexcept = default;

        template <typename T>
        [[nodiscard]] T& get()
        {
            if (T* t = try_get<T>())
            {
                return *t;
            }

            head_ = extra_archive_data_node(
                static_cast<T*>(nullptr), HPX_MOVE(head_));
            return *try_get<T>();
        }

        template <typename T>
        [[nodiscard]] T* try_get() const noexcept
        {
            return head_.get<T>();
        }

        // reset all extra archive data
        void reset() const
        {
            auto* ptr = head_.ptr_.get();
            while (ptr != nullptr)
            {
                ptr->reset();
                ptr = ptr->next_.ptr_.get();
            }
        }

        extra_archive_data_node head_;
    };
}    // namespace hpx::serialization::detail
