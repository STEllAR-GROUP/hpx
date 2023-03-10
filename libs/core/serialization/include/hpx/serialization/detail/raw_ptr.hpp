//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2022 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/serialization/detail/pointer.hpp>
#include <hpx/serialization/serialization_fwd.hpp>

namespace hpx::serialization::detail {

    template <typename T>
    struct raw_ptr_type
    {
        using element_type = T;

        explicit constexpr raw_ptr_type(T* t) noexcept
          : t(t)
        {
        }

        [[nodiscard]] constexpr T* get() const noexcept
        {
            return t;
        }

        [[nodiscard]] constexpr T& operator*() const noexcept
        {
            return *t;
        }

        [[nodiscard]] explicit constexpr operator bool() const noexcept
        {
            return t != nullptr;
        }

    private:
        T* t;
    };

    template <typename T>
    struct raw_ptr_proxy
    {
        explicit constexpr raw_ptr_proxy(T*& t) noexcept
          : t(t)
        {
        }

        explicit constexpr raw_ptr_proxy(T* const& t) noexcept    //-V835
          : t(const_cast<T*&>(t))
        {
        }

        void serialize(output_archive& ar) const
        {
            serialize_pointer_untracked(ar, raw_ptr_type<T>(t));
        }

        void serialize(input_archive& ar)
        {
            raw_ptr_type<T> ptr(t);
            serialize_pointer_untracked(ar, ptr);
            t = ptr.get();
        }

        T*& t;
    };

    template <typename T>
    HPX_FORCEINLINE constexpr raw_ptr_proxy<T> raw_ptr(T*& t) noexcept
    {
        return raw_ptr_proxy<T>(t);
    }

    template <typename T>
    HPX_FORCEINLINE constexpr raw_ptr_proxy<T> raw_ptr(
        T* const& t) noexcept    //-V835
    {
        return raw_ptr_proxy<T>(t);
    }

    // allow raw_ptr_type to be serialized as prvalue
    template <typename T>
    HPX_FORCEINLINE output_archive& operator<<(
        output_archive& ar, raw_ptr_proxy<T> t)
    {
        t.serialize(ar);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE input_archive& operator>>(
        input_archive& ar, raw_ptr_proxy<T> t)
    {
        t.serialize(ar);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE output_archive& operator&(    //-V524
        output_archive& ar, raw_ptr_proxy<T> t)
    {
        t.serialize(ar);
        return ar;
    }

    template <typename T>
    HPX_FORCEINLINE input_archive& operator&(    //-V524
        input_archive& ar, raw_ptr_proxy<T> t)
    {
        t.serialize(ar);
        return ar;
    }
}    // namespace hpx::serialization::detail
