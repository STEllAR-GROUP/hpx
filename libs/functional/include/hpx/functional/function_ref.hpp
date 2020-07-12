//  Copyright (c) 2018-2019 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/assert.hpp>
#include <hpx/functional/detail/empty_function.hpp>
#include <hpx/functional/detail/vtable/callable_vtable.hpp>
#include <hpx/functional/detail/vtable/vtable.hpp>
#include <hpx/functional/traits/get_function_address.hpp>
#include <hpx/functional/traits/get_function_annotation.hpp>
#include <hpx/functional/traits/is_invocable.hpp>

#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace util {
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    class function_ref;

    namespace detail {
        template <typename Sig>
        struct function_ref_vtable
          : callable_vtable<Sig>
          , callable_info_vtable
        {
            template <typename T>
            constexpr function_ref_vtable(construct_vtable<T>) noexcept
              : callable_vtable<Sig>(construct_vtable<T>())
              , callable_info_vtable(construct_vtable<T>())
            {
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename F>
        constexpr bool is_empty_function_ptr(F* fp) noexcept
        {
            return fp == nullptr;
        }

        template <typename T, typename C>
        constexpr bool is_empty_function_ptr(T C::*mp) noexcept
        {
            return mp == nullptr;
        }

        template <typename F>
        constexpr bool is_empty_function_ptr(F const& f) noexcept
        {
            return false;
        }
    }    // namespace detail

    ///////////////////////////////////////////////////////////////////////////
    template <typename R, typename... Ts>
    class function_ref<R(Ts...)>
    {
        using VTable = detail::function_ref_vtable<R(Ts...)>;

    public:
        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, function_ref>::value &&
                traits::is_invocable_r<R, F&, Ts...>::value>::type>
        function_ref(F&& f)
          : object(nullptr)
        {
            assign(std::forward<F>(f));
        }

        function_ref(function_ref const& other) noexcept
          : vptr(other.vptr)
          , object(other.object)
        {
        }

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, function_ref>::value &&
                traits::is_invocable_r<R, F&, Ts...>::value>::type>
        function_ref& operator=(F&& f)
        {
            assign(std::forward<F>(f));
            return *this;
        }

        // NOLINTNEXTLINE(bugprone-unhandled-self-assignment)
        function_ref& operator=(function_ref const& other) noexcept
        {
            vptr = other.vptr;
            object = other.object;
            return *this;
        }

        template <typename F,
            typename T = typename std::remove_reference<F>::type,
            typename Enable =
                typename std::enable_if<!std::is_pointer<T>::value>::type>
        void assign(F&& f)
        {
            HPX_ASSERT(!detail::is_empty_function_ptr(f));
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            vptr = get_vtable<T>();
#else
            vptr = get_vtable<T>()->invoke;
#endif
            object = reinterpret_cast<void*>(std::addressof(f));
        }

        template <typename T>
        void assign(std::reference_wrapper<T> f_ref) noexcept
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            vptr = get_vtable<T>();
#else
            vptr = get_vtable<T>()->invoke;
#endif
            object = reinterpret_cast<void*>(std::addressof(f_ref.get()));
        }

        template <typename T>
        void assign(T* f_ptr) noexcept
        {
            HPX_ASSERT(f_ptr != nullptr);
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            vptr = get_vtable<T>();
#else
            vptr = get_vtable<T>()->invoke;
#endif
            object = reinterpret_cast<void*>(f_ptr);
        }

        void swap(function_ref& f) noexcept
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object);    // swap
        }

        HPX_FORCEINLINE R operator()(Ts... vs) const
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            return vptr->invoke(object, std::forward<Ts>(vs)...);
#else
            return vptr(object, std::forward<Ts>(vs)...);
#endif
        }

        std::size_t get_function_address() const
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            return vptr->get_function_address(object);
#else
            return 0;
#endif
        }

        char const* get_function_annotation() const
        {
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
            return vptr->get_function_annotation(object);
#else
            return nullptr;
#endif
        }

        util::itt::string_handle get_function_annotation_itt() const
        {
#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
            return vptr->get_function_annotation_itt(object);
#else
            static util::itt::string_handle sh;
            return sh;
#endif
        }

    private:
        template <typename T>
        static VTable const* get_vtable() noexcept
        {
            return detail::get_vtable<VTable, T>();
        }

    protected:
#if defined(HPX_HAVE_THREAD_DESCRIPTION)
        VTable const* vptr;
#else
        R (*vptr)(void*, Ts&&...);
#endif
        void* object;
    };
}}    // namespace hpx::util

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits {
    template <typename Sig>
    struct get_function_address<util::function_ref<Sig>>
    {
        static std::size_t call(util::function_ref<Sig> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename Sig>
    struct get_function_annotation<util::function_ref<Sig>>
    {
        static char const* call(util::function_ref<Sig> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Sig>
    struct get_function_annotation_itt<util::function_ref<Sig>>
    {
        static util::itt::string_handle call(
            util::function_ref<Sig> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
}}    // namespace hpx::traits
#endif
