//  Copyright (c) 2019 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_SERIALIZATION_DETAIL_EXTRA_ARCHIVE_DATA_HPP
#define HPX_SERIALIZATION_DETAIL_EXTRA_ARCHIVE_DATA_HPP

#include <hpx/assertion.hpp>

#include <memory>
#include <type_traits>
#include <utility>

namespace hpx { namespace serialization { namespace detail {

    struct extra_archive_data_member_base;
    template <typename T>
    struct extra_archive_data_member;

    struct extra_archive_data_node
    {
        extra_archive_data_node() noexcept
          : ptr_()
          , id_(nullptr)
        {
        }

        template <typename T>
        extra_archive_data_node(T* t, extra_archive_data_node&& next);

        extra_archive_data_node(extra_archive_data_node&&) noexcept = default;
        extra_archive_data_node& operator=(
            extra_archive_data_node&&) noexcept = default;

        template <typename T>
        inline T* get() noexcept;

        std::unique_ptr<extra_archive_data_member_base> ptr_;
        void (*id_)();
    };

    struct extra_archive_data_member_base
    {
        extra_archive_data_member_base(extra_archive_data_node&& next) noexcept
          : next_(std::move(next))
        {
        }

        virtual ~extra_archive_data_member_base() = default;

        extra_archive_data_node next_;
    };

    template <typename T>
    struct extra_archive_data_member : extra_archive_data_member_base
    {
        extra_archive_data_member(extra_archive_data_node&& next)
          : extra_archive_data_member_base(std::move(next))
        {
        }

        extra_archive_data_member(
            extra_archive_data_member const&) noexcept = delete;
        extra_archive_data_member& operator=(
            extra_archive_data_member const&) noexcept = delete;

        T* value() noexcept
        {
            return std::addressof(t_);
        }

        static void id() noexcept {}

        T t_;
    };

    template <typename T>
    extra_archive_data_node::extra_archive_data_node(
        T*, extra_archive_data_node&& next)
      : ptr_(new extra_archive_data_member<T>(std::move(next)))
      , id_(&extra_archive_data_member<T>::id)
    {
    }

    template <typename T>
    T* extra_archive_data_node::get() noexcept
    {
        constexpr auto id = &extra_archive_data_member<T>::id;
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
        T& get() noexcept
        {
            if (T* t = try_get<T>())
            {
                return *t;
            }

            head_ = extra_archive_data_node(
                static_cast<T*>(nullptr), std::move(head_));
            return *try_get<T>();
        }

        template <typename T>
        T* try_get() noexcept
        {
            return head_.get<T>();
        }

        extra_archive_data_node head_;
    };

}}}    // namespace hpx::serialization::detail

#endif
