//  Copyright (c) 2018 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file hpx/runtime/threads/thread_id_type.hpp

#ifndef HPX_THREADS_THREAD_ID_TYPE_HPP
#define HPX_THREADS_THREAD_ID_TYPE_HPP

#include <hpx/config/constexpr.hpp>
#include <hpx/config/export_definitions.hpp>

#include <cstddef>
#include <functional>
#include <iosfwd>

namespace hpx {
namespace threads {
    class HPX_EXPORT thread_data;

    struct thread_id_type
    {
        constexpr thread_id_type()
          : thrd_(nullptr)
        {
        }
        explicit constexpr thread_id_type(thread_data* thrd)
          : thrd_(thrd)
        {
        }

        thread_id_type(thread_id_type const&) = default;
        thread_id_type& operator=(thread_id_type const&) = default;

        constexpr thread_data* operator->() const
        {
            return thrd_;
        }

        constexpr thread_data& operator*() const
        {
            return *thrd_;
        }

        explicit constexpr operator bool() const
        {
            return nullptr != thrd_;
        }

        constexpr thread_data* get() const
        {
            return thrd_;
        }

        HPX_CXX14_CONSTEXPR void reset()
        {
            thrd_ = nullptr;
        }

        friend constexpr bool operator==(
            std::nullptr_t, thread_id_type const& rhs)
        {
            return nullptr == rhs.thrd_;
        }

        friend constexpr bool operator!=(
            std::nullptr_t, thread_id_type const& rhs)
        {
            return nullptr != rhs.thrd_;
        }

        friend constexpr bool operator==(
            thread_id_type const& lhs, std::nullptr_t)
        {
            return nullptr == lhs.thrd_;
        }

        friend constexpr bool operator!=(
            thread_id_type const& lhs, std::nullptr_t)
        {
            return nullptr != lhs.thrd_;
        }

        friend constexpr bool operator==(
            thread_id_type const& lhs, thread_id_type const& rhs)
        {
            return lhs.thrd_ == rhs.thrd_;
        }

        friend constexpr bool operator!=(
            thread_id_type const& lhs, thread_id_type const& rhs)
        {
            return lhs.thrd_ != rhs.thrd_;
        }

        friend HPX_CXX14_CONSTEXPR bool operator<(
            thread_id_type const& lhs, thread_id_type const& rhs)
        {
            return std::less<void const*>{}(lhs.thrd_, rhs.thrd_);
        }

        friend HPX_CXX14_CONSTEXPR bool operator>(
            thread_id_type const& lhs, thread_id_type const& rhs)
        {
            return std::less<void const*>{}(rhs.thrd_, lhs.thrd_);
        }

        friend HPX_CXX14_CONSTEXPR bool operator<=(
            thread_id_type const& lhs, thread_id_type const& rhs)
        {
            return !(rhs > lhs);
        }

        friend HPX_CXX14_CONSTEXPR bool operator>=(
            thread_id_type const& lhs, thread_id_type const& rhs)
        {
            return !(rhs < lhs);
        }

        template <typename Char, typename Traits>
        friend std::basic_ostream<Char, Traits>& operator<<(
            std::basic_ostream<Char, Traits>& os, thread_id_type const& id)
        {
            os << id.get();
            return os;
        }

    private:
        thread_data* thrd_;
    };

    HPX_CONSTEXPR_OR_CONST thread_id_type invalid_thread_id;
}
}

namespace std {
template <>
struct hash<::hpx::threads::thread_id_type>
{
    typedef ::hpx::threads::thread_id_type argument_type;
    typedef std::size_t result_type;

    std::size_t operator()(::hpx::threads::thread_id_type const& v) const
        noexcept
    {
        std::hash<const ::hpx::threads::thread_data*> hasher_;
        return hasher_(v.get());
    }
};
}

#endif
