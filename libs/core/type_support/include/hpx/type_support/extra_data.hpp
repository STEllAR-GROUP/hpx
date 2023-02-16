//  Copyright (c) 2019 Thomas Heller
//  Copyright (c) 2019-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/assert.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace hpx::util {

    using extra_data_id_type = void*;

    template <typename T>
    struct extra_data_helper
    {
        // this is intentionally left unimplemented, will lead to linker errors
        // if used with unknown data type
        static extra_data_id_type id() noexcept;

        // this is a function that should be implemented in order to reset the
        // extra archive data item
        static void reset(T* data) noexcept;
    };

    template <typename T>
    [[nodiscard]] extra_data_id_type extra_data_id() noexcept
    {
        return extra_data_helper<T>::id();
    }

    template <typename T>
    constexpr void reset_extra_data(T* data) noexcept(
        noexcept(extra_data_helper<T>::reset(data)))
    {
        extra_data_helper<T>::reset(data);
    }

    struct extra_data_member_base;
    template <typename T>
    struct extra_data_member;

    struct extra_data_node
    {
        constexpr extra_data_node() noexcept = default;

        template <typename T>
        extra_data_node(T* t, extra_data_node&& next);

        extra_data_node(extra_data_node const&) = delete;
        extra_data_node(extra_data_node&&) noexcept = default;
        extra_data_node& operator=(extra_data_node const&) = delete;
        extra_data_node& operator=(extra_data_node&&) noexcept = default;

        ~extra_data_node() = default;

        template <typename T>
        [[nodiscard]] T* get() const noexcept;

        std::unique_ptr<extra_data_member_base> ptr_;
        extra_data_id_type id_ = nullptr;
    };

    struct extra_data_member_base
    {
        explicit extra_data_member_base(extra_data_node&& next) noexcept
          : next_(HPX_MOVE(next))
        {
        }

        extra_data_member_base(extra_data_member_base const&) = delete;
        extra_data_member_base(extra_data_member_base&&) noexcept = default;
        extra_data_member_base& operator=(
            extra_data_member_base const&) = delete;
        extra_data_member_base& operator=(
            extra_data_member_base&&) noexcept = default;

        virtual ~extra_data_member_base() = default;

        virtual void reset() = 0;

        extra_data_node next_;
    };

    template <typename T>
    struct extra_data_member : extra_data_member_base
    {
        explicit constexpr extra_data_member(extra_data_node&& next) noexcept
          : extra_data_member_base(HPX_MOVE(next))
        {
        }

        extra_data_member(extra_data_member const&) = delete;
        extra_data_member(extra_data_member&&) = default;
        extra_data_member& operator=(extra_data_member const&) = delete;
        extra_data_member& operator=(extra_data_member&&) = default;

        ~extra_data_member() override = default;

        [[nodiscard]] constexpr T* value() const noexcept
        {
            return std::addressof(const_cast<T&>(t_));
        }

        void reset() override
        {
            reset_extra_data(value());
        }

        T t_;
    };

    template <typename T>
    extra_data_node::extra_data_node(T*, extra_data_node&& next)
      : ptr_(std::make_unique<extra_data_member<T>>(HPX_MOVE(next)))
      , id_(extra_data_id<T>())
    {
    }

    template <typename T>
    T* extra_data_node::get() const noexcept
    {
        auto id = extra_data_id<T>();
        if (id_ == nullptr)
        {
            HPX_ASSERT(!ptr_);
            return nullptr;
        }

        HPX_ASSERT(ptr_);
        if (id_ == id)
        {
            return static_cast<extra_data_member<T>*>(ptr_.get())->value();
        }

        return ptr_->next_.get<T>();
    }

    struct extra_data
    {
        extra_data() noexcept = default;

        template <typename T>
        T& get()
        {
            if (T* t = try_get<T>())
            {
                return *t;
            }

            head_ = extra_data_node(static_cast<T*>(nullptr), HPX_MOVE(head_));
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

        extra_data_node head_;
    };
}    // namespace hpx::util
