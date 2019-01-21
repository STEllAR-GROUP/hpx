//  Copyright (c) 2018-2019 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_FUNCTION_REF_HPP
#define HPX_UTIL_FUNCTION_REF_HPP

#include <hpx/config.hpp>
#include <hpx/traits/get_function_address.hpp>
#include <hpx/traits/get_function_annotation.hpp>
#include <hpx/traits/is_callable.hpp>
#include <hpx/util/assert.hpp>
#include <hpx/util/detail/empty_function.hpp>
#include <hpx/util/detail/vtable/callable_vtable.hpp>
#include <hpx/util/detail/vtable/vtable.hpp>

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>

namespace hpx { namespace util
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename Sig>
    class function_ref;

    template <typename R, typename ...Ts>
    class function_ref<R(Ts...)>
    {
        using VTable = detail::callable_vtable<R(Ts...)>;

    public:
        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, function_ref>::value
             && traits::is_invocable_r<R, F&, Ts...>::value
            >::type>
        function_ref(F&& f)
        {
            assign(std::forward<F>(f));
        }

        function_ref(function_ref const& other) noexcept
          : vptr(other.vptr)
          , object(other.object)
        {}

        template <typename F, typename FD = typename std::decay<F>::type,
            typename Enable = typename std::enable_if<
                !std::is_same<FD, function_ref>::value
             && traits::is_invocable_r<R, F&, Ts...>::value
            >::type>
        function_ref& operator=(F&& f)
        {
            assign(std::forward<F>(f));
            return *this;
        }

        function_ref& operator=(function_ref const& other) noexcept
        {
            vptr = other.vptr;
            object = other.object;
            return *this;
        }

        template <typename F>
        void assign(F&& f)
        {
            typedef typename std::remove_reference<F>::type target_type;

            HPX_ASSERT(!detail::is_empty_function(f));
            vptr = get_vtable<target_type>();
            object = reinterpret_cast<void*>(std::addressof(f));
        }

        void swap(function_ref& f) noexcept
        {
            std::swap(vptr, f.vptr);
            std::swap(object, f.object); // swap
        }

        HPX_FORCEINLINE R operator()(Ts... vs) const
        {
            return vptr->invoke(object, std::forward<Ts>(vs)...);
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
        VTable const *vptr;
        void* object;
    };
}}

#if defined(HPX_HAVE_THREAD_DESCRIPTION)
///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace traits
{
    template <typename Sig>
    struct get_function_address<util::function_ref<Sig>>
    {
        static std::size_t
            call(util::function_ref<Sig> const& f) noexcept
        {
            return f.get_function_address();
        }
    };

    template <typename Sig>
    struct get_function_annotation<util::function_ref<Sig>>
    {
        static char const*
            call(util::function_ref<Sig> const& f) noexcept
        {
            return f.get_function_annotation();
        }
    };

#if HPX_HAVE_ITTNOTIFY != 0 && !defined(HPX_HAVE_APEX)
    template <typename Sig>
    struct get_function_annotation_itt<util::function_ref<Sig>>
    {
        static util::itt::string_handle
            call(util::function_ref<Sig> const& f) noexcept
        {
            return f.get_function_annotation_itt();
        }
    };
#endif
}}
#endif

#endif
