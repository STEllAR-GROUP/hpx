//  Copyright (c) 2011 Thomas Heller
//  Copyright (c) 2013 Hartmut Kaiser
//  Copyright (c) 2014-2019 Agustin Berge
//  Copyright (c) 2017 Google
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_DETAIL_BASIC_FUNCTION_HPP
#define HPX_UTIL_DETAIL_BASIC_FUNCTION_HPP

#include <hpx/config.hpp>
#include <hpx/assertion.hpp>
#include <hpx/functional/detail/empty_function.hpp>
#include <hpx/functional/detail/vtable/function_vtable.hpp>
#include <hpx/functional/detail/vtable/vtable.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/functional/traits/is_callable.hpp>

#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace util { namespace detail {
    static const std::size_t function_storage_size = 3 * sizeof(void*);

    ///////////////////////////////////////////////////////////////////////////
    class HPX_EXPORT function_base
    {
        using vtable = function_base_vtable;

    public:
        HPX_CONSTEXPR explicit function_base(
            function_base_vtable const* empty_vptr) noexcept
          : vptr(empty_vptr)
          , object(nullptr)
          , storage_init()
        {
        }

        function_base(function_base const& other, vtable const* empty_vtable);
        function_base(function_base&& other, vtable const* empty_vptr) noexcept;
        ~function_base();

        void op_assign(function_base const& other, vtable const* empty_vtable);
        void op_assign(
            function_base&& other, vtable const* empty_vtable) noexcept;

        void destroy() noexcept;
        void reset(vtable const* empty_vptr) noexcept;
        void swap(function_base& f) noexcept;

        bool empty() const noexcept
        {
            return object == nullptr;
        }

        explicit operator bool() const noexcept
        {
            return !empty();
        }

        std::size_t get_function_address() const;
        char const* get_function_annotation() const;
        util::itt::string_handle get_function_annotation_itt() const;

    protected:
        vtable const* vptr;
        void* object;
        union
        {
            char storage_init;
            mutable unsigned char storage[function_storage_size];
        };
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename F>
    HPX_CONSTEXPR bool is_empty_function(F* fp) noexcept
    {
        return fp == nullptr;
    }

    template <typename T, typename C>
    HPX_CONSTEXPR bool is_empty_function(T C::*mp) noexcept
    {
        return mp == nullptr;
    }

    inline bool is_empty_function_impl(function_base const* f) noexcept
    {
        return f->empty();
    }

    inline HPX_CONSTEXPR bool is_empty_function_impl(...) noexcept
    {
        return false;
    }

    template <typename F>
    HPX_CONSTEXPR bool is_empty_function(F const& f) noexcept
    {
        return detail::is_empty_function_impl(&f);
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig, bool Copyable, bool Serializable>
    class basic_function;

    template <bool Copyable, typename R, typename... Ts>
    class basic_function<R(Ts...), Copyable, /*Serializable*/ false>
      : public function_base
    {
        using base_type = function_base;
        using vtable = function_vtable<R(Ts...), Copyable>;

    public:
        HPX_CONSTEXPR basic_function() noexcept
          : base_type(get_empty_vtable())
        {
        }

        basic_function(basic_function const& other)
          : base_type(other, get_empty_vtable())
        {
        }

        basic_function(basic_function&& other) noexcept
          : base_type(std::move(other), get_empty_vtable())
        {
        }

        basic_function& operator=(basic_function const& other)
        {
            base_type::op_assign(other, get_empty_vtable());
            return *this;
        }

        basic_function& operator=(basic_function&& other) noexcept
        {
            base_type::op_assign(std::move(other), get_empty_vtable());
            return *this;
        }

        void assign(std::nullptr_t) noexcept
        {
            base_type::reset(get_empty_vtable());
        }

        template <typename F>
        void assign(F&& f)
        {
            using T = typename std::decay<F>::type;
            static_assert(
                !Copyable || std::is_constructible<T, T const&>::value,
                "F shall be CopyConstructible");

            if (!detail::is_empty_function(f))
            {
                vtable const* f_vptr = get_vtable<T>();
                void* buffer = nullptr;
                if (vptr == f_vptr)
                {
                    HPX_ASSERT(object != nullptr);
                    // reuse object storage
                    buffer = object;
                    vtable::template get<T>(object).~T();
                }
                else
                {
                    destroy();
                    vptr = f_vptr;
                    buffer = vtable::template allocate<T>(
                        storage, function_storage_size);
                }
                object = ::new (buffer) T(std::forward<F>(f));
            }
            else
            {
                base_type::reset(get_empty_vtable());
            }
        }

        void reset() noexcept
        {
            base_type::reset(get_empty_vtable());
        }

        using base_type::empty;
        using base_type::swap;
        using base_type::operator bool;

        template <typename T>
        T* target() noexcept
        {
            using TD = typename std::remove_cv<T>::type;
            static_assert(traits::is_invocable_r<R, TD&, Ts...>::value,
                "T shall be Callable with the function signature");

            vtable const* f_vptr = get_vtable<TD>();
            if (vptr != f_vptr || empty())
                return nullptr;

            return &vtable::template get<TD>(object);
        }

        template <typename T>
        T const* target() const noexcept
        {
            using TD = typename std::remove_cv<T>::type;
            static_assert(traits::is_invocable_r<R, TD&, Ts...>::value,
                "T shall be Callable with the function signature");

            vtable const* f_vptr = get_vtable<TD>();
            if (vptr != f_vptr || empty())
                return nullptr;

            return &vtable::template get<TD>(object);
        }

        HPX_FORCEINLINE R operator()(Ts... vs) const
        {
            vtable const* vptr = static_cast<vtable const*>(base_type::vptr);
            return vptr->invoke(object, std::forward<Ts>(vs)...);
        }

        using base_type::get_function_address;
        using base_type::get_function_annotation;
        using base_type::get_function_annotation_itt;

    private:
        static HPX_CONSTEXPR vtable const* get_empty_vtable() noexcept
        {
            return detail::get_empty_function_vtable<R(Ts...)>();
        }

        template <typename T>
        static vtable const* get_vtable() noexcept
        {
            return detail::get_vtable<vtable, T>();
        }

    protected:
        using base_type::object;
        using base_type::storage;
        using base_type::vptr;
    };
}}}    // namespace hpx::util::detail

#endif
